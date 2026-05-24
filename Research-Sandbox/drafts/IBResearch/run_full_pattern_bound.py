from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from src.anthony_model import (
    DEFAULT_WEIGHTS,
    load_default_data,
    mip_objective_value,
    placement_from_timetable,
    prepare_anthony_model_data,
)
from src.full_heuristic import (
    _build_blocks,
    _generate_block_candidates,
    _pair_value_map,
    validate_full_solution,
)
from run_full_pattern_lns import (
    CrossCost,
    _assignment_signature,
    _cross_cost,
    _pair_objective,
    _var_name,
)


@dataclass(frozen=True)
class Pattern:
    subject: str
    index: int
    assignment: dict[str, tuple[pd.Timestamp, str]]
    internal_objective: float
    internal_same_slot: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a global subject-pattern relaxation for the full IB instance. "
            "The model keeps all subject patterns but only a selected set of "
            "inter-subject interactions, so the solver bound is a certified "
            "lower bound for the full nonnegative Furlong objective."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=Path, default=None)
    parser.add_argument("--nb-days", type=int, default=34)
    parser.add_argument("--objective-mode", choices=["formal", "anthony_appendix"], default="formal")
    parser.add_argument("--edge-count", type=int, default=40, help="Number of inter-subject edges to model exactly; use -1 for all edges.")
    parser.add_argument(
        "--edge-score",
        choices=["pair_mass", "start_objective"],
        default="pair_mass",
        help="Criterion used to choose sparse inter-subject edges.",
    )
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--cluster-cut-count", type=int, default=0)
    parser.add_argument("--cluster-size", type=int, default=6)
    parser.add_argument("--cluster-cut-time-limit", type=float, default=120.0)
    parser.add_argument(
        "--cluster-score",
        choices=["pair_mass", "start_objective"],
        default="pair_mass",
        help="Criterion used to choose dense clusters for combinatorial lower-bound cuts.",
    )
    parser.add_argument("--joint-cluster-count", type=int, default=0)
    parser.add_argument("--joint-max-tuples", type=int, default=500_000)
    parser.add_argument("--joint-cut-count", type=int, default=0)
    parser.add_argument("--joint-cut-max-tuples", type=int, default=500_000)
    parser.add_argument("--joint-cut-rounds", type=int, default=5)
    parser.add_argument("--joint-cut-max-cuts-per-round", type=int, default=3)
    parser.add_argument("--joint-cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--window-cut-count", type=int, default=0)
    parser.add_argument("--window-cluster-size", type=int, default=6)
    parser.add_argument("--window-length", type=int, default=7)
    parser.add_argument("--window-cut-limit", type=int, default=12)
    parser.add_argument("--window-cut-time-limit", type=float, default=60.0)
    parser.add_argument("--separated-window-cut-count", type=int, default=0)
    parser.add_argument("--separation-cluster-count", type=int, default=4)
    parser.add_argument("--separation-cluster-size", type=int, default=8)
    parser.add_argument("--separation-window-length", type=int, default=8)
    parser.add_argument("--separation-max-subproblems", type=int, default=24)
    parser.add_argument("--separation-cut-time-limit", type=float, default=45.0)
    parser.add_argument("--separation-min-activation", type=float, default=0.05)
    parser.add_argument("--mip-focus", type=int, default=None)
    parser.add_argument("--cuts", type=int, default=None)
    parser.add_argument("--symmetry", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument("--relax-lambda", action="store_true", help="Solve the LP relaxation of the pattern model.")
    parser.add_argument("--no-partial-clash-cap", action="store_true", help="Do not add the valid partial same-slot clash cap.")
    parser.add_argument("--summary-output", type=Path, default=Path("full_pattern_bound_summary.csv"))
    parser.add_argument("--progress-output", type=Path, default=Path("full_pattern_bound_progress.csv"))
    parser.add_argument("--log-output", type=Path, default=Path("full_pattern_bound.log"))
    parser.add_argument("--plot-output", type=Path, default=Path("full_pattern_bound_bounds.png"))
    args = parser.parse_args()

    started = time.perf_counter()
    result = solve_pattern_bound(args)
    elapsed = time.perf_counter() - started

    summary = dict(result["summary"])
    summary["elapsed_seconds"] = elapsed
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)
    result["progress"].to_csv(args.progress_output, index=False)
    _plot_progress(result["progress"], args.plot_output, title="Full IB sparse subject-pattern bound")

    print("Subjects:", summary["subject_count"])
    print("Patterns:", summary["pattern_count"])
    print("Selected inter-subject edges:", summary["edge_count"])
    print("Mu variables:", summary["mu_count"])
    print("Incompatible mu variables fixed to zero:", summary["incompatible_mu_count"])
    print("Relaxed model incumbent:", summary["relaxation_incumbent"])
    print("Certified full lower bound:", summary["certified_full_lower_bound"])
    print("Best incumbent full objective:", summary["start_full_objective"])
    print("Implied full gap vs start:", summary["gap_vs_start"])
    print("Status:", summary["status"])
    print("Nodes:", summary["nodes"])
    print("Iterations:", summary["iterations"])
    print("Work:", summary["work"])
    print("Runtime seconds:", summary["runtime_seconds"])
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved progress to {args.progress_output}")
    print(f"Saved plot to {args.plot_output}")
    print(f"Saved log to {args.log_output}")


def solve_pattern_bound(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("run_full_pattern_bound.py requires gurobipy.") from exc

    weights = dict(DEFAULT_WEIGHTS)
    if any(value < 0 for value in weights.values()):
        raise ValueError("Sparse lower bounds are valid only for nonnegative objective weights.")

    exams, days, pairs = load_default_data(args.data_dir)
    data = prepare_anthony_model_data(exams, days, pairs, nb_days=args.nb_days, recode_math_paper_three=True)
    pair_values = _pair_value_map(data.pairs)
    day_index = {date: idx for idx, date in enumerate(data.dates)}
    exam_lengths = dict(zip(data.exams["Full Name"], data.exams["Length"]))
    exam_subjects = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    blocks = _build_blocks(data.exams)
    subjects = sorted(blocks)

    start_timetable = None
    start_placement: dict[str, tuple[pd.Timestamp, str]] | None = None
    start_full_objective: float | None = None
    if args.start is not None:
        start_timetable = _load_timetable(args.start)
        validate_full_solution(start_timetable, data)
        start_placement = placement_from_timetable(start_timetable)
        start_full_objective = mip_objective_value(start_timetable, data.pairs, data.days, mode=args.objective_mode)

    patterns = _build_patterns(
        data=data,
        blocks=blocks,
        pair_values=pair_values,
        day_index=day_index,
        exam_lengths=exam_lengths,
        exam_subjects=exam_subjects,
        objective_mode=args.objective_mode,
        weights=weights,
        start_placement=start_placement,
    )
    selected_edges = _select_edges(
        data=data,
        subjects=subjects,
        edge_count=args.edge_count,
        edge_score=args.edge_score,
        objective_mode=args.objective_mode,
        weights=weights,
        start_placement=start_placement,
    )
    clusters = _select_clusters(
        data=data,
        subjects=subjects,
        cluster_count=args.cluster_cut_count,
        cluster_size=args.cluster_size,
        cluster_score=args.cluster_score,
        objective_mode=args.objective_mode,
        weights=weights,
        start_placement=start_placement,
    )
    joint_clusters = _select_joint_clusters(
        data=data,
        subjects=subjects,
        patterns=patterns,
        joint_cluster_count=args.joint_cluster_count,
        joint_max_tuples=args.joint_max_tuples,
        objective_mode=args.objective_mode,
        weights=weights,
        start_placement=start_placement,
    )
    window_clusters = _select_clusters(
        data=data,
        subjects=subjects,
        cluster_count=args.window_cut_count,
        cluster_size=args.window_cluster_size,
        cluster_score="pair_mass",
        objective_mode=args.objective_mode,
        weights=weights,
        start_placement=start_placement,
    )
    selected_edges = _with_cluster_edges(selected_edges, clusters + joint_clusters + window_clusters)
    cluster_cuts = _solve_cluster_cuts(
        clusters=clusters,
        patterns=patterns,
        pair_values=pair_values,
        day_index=day_index,
        exam_lengths=exam_lengths,
        objective_mode=args.objective_mode,
        weights=weights,
        time_limit=args.cluster_cut_time_limit,
    )
    window_cuts = _solve_window_cuts(
        clusters=window_clusters,
        patterns=patterns,
        dates=data.dates,
        pair_values=pair_values,
        day_index=day_index,
        exam_lengths=exam_lengths,
        objective_mode=args.objective_mode,
        weights=weights,
        window_length=args.window_length,
        cut_limit=args.window_cut_limit,
        time_limit=args.window_cut_time_limit,
    )
    separated_window_cuts = _separate_window_cuts(
        data=data,
        subjects=subjects,
        patterns=patterns,
        base_edges=selected_edges,
        pair_values=pair_values,
        day_index=day_index,
        exam_lengths=exam_lengths,
        objective_mode=args.objective_mode,
        weights=weights,
        start_placement=start_placement,
        cluster_count=args.separation_cluster_count,
        cluster_size=args.separation_cluster_size,
        window_length=args.separation_window_length,
        cut_count=args.separated_window_cut_count,
        max_subproblems=args.separation_max_subproblems,
        min_activation=args.separation_min_activation,
        cut_time_limit=args.separation_cut_time_limit,
        partial_clash_cap=not args.no_partial_clash_cap,
    )
    window_cuts.extend(separated_window_cuts)
    joint_cut_clusters = _select_joint_clusters(
        data=data,
        subjects=subjects,
        patterns=patterns,
        joint_cluster_count=args.joint_cut_count,
        joint_max_tuples=args.joint_cut_max_tuples,
        objective_mode=args.objective_mode,
        weights=weights,
        start_placement=start_placement,
    )
    selected_edges = _with_cluster_edges(
        selected_edges,
        [tuple(cut["cluster"]) for cut in separated_window_cuts] + joint_cut_clusters,
    )

    model = gp.Model("full_sparse_subject_pattern_bound")
    model.setParam("OutputFlag", args.output_flag)
    model.setParam("LogFile", str(args.log_output))
    model.setParam("TimeLimit", args.time_limit)
    if args.mip_focus is not None:
        model.setParam("MIPFocus", args.mip_focus)
    if args.cuts is not None:
        model.setParam("Cuts", args.cuts)
    if args.symmetry is not None:
        model.setParam("Symmetry", args.symmetry)
    if args.threads is not None:
        model.setParam("Threads", args.threads)

    lambda_type = GRB.CONTINUOUS if args.relax_lambda else GRB.BINARY
    lam: dict[tuple[str, int], Any] = {}
    for subject in subjects:
        for pattern in patterns[subject]:
            lam[(subject, pattern.index)] = model.addVar(
                vtype=lambda_type,
                lb=0.0,
                ub=1.0,
                name=_var_name("lambda", subject, pattern.index),
            )

    mu: dict[tuple[str, int, str, int], Any] = {}
    cross_costs: dict[tuple[str, int, str, int], CrossCost] = {}
    incompatible_mu_count = 0
    for left_subject, right_subject in selected_edges:
        left_patterns = patterns[left_subject]
        right_patterns = patterns[right_subject]
        for left_pattern in left_patterns:
            for right_pattern in right_patterns:
                cost = _cross_cost(
                    left_pattern.assignment,
                    right_pattern.assignment,
                    pair_values=pair_values,
                    day_index=day_index,
                    exam_lengths=exam_lengths,
                    objective_mode=args.objective_mode,
                    weights=weights,
                )
                key = (left_subject, left_pattern.index, right_subject, right_pattern.index)
                cross_costs[key] = cost
                ub = 0.0 if cost.daily_length_infeasible else 1.0
                if cost.daily_length_infeasible:
                    incompatible_mu_count += 1
                mu[key] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=ub,
                    name=_var_name("mu", left_subject, left_pattern.index, right_subject, right_pattern.index),
                )

    model.update()

    for subject in subjects:
        model.addConstr(
            gp.quicksum(lam[(subject, pattern.index)] for pattern in patterns[subject]) == 1,
            name=_var_name("choose", subject),
        )

    for left_subject, right_subject in selected_edges:
        for left_pattern in patterns[left_subject]:
            model.addConstr(
                gp.quicksum(
                    mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                    for right_pattern in patterns[right_subject]
                )
                == lam[(left_subject, left_pattern.index)],
                name=_var_name("mu_left", left_subject, left_pattern.index, right_subject),
            )
        for right_pattern in patterns[right_subject]:
            model.addConstr(
                gp.quicksum(
                    mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                    for left_pattern in patterns[left_subject]
                )
                == lam[(right_subject, right_pattern.index)],
                name=_var_name("mu_right", left_subject, right_subject, right_pattern.index),
            )

    if start_placement is not None:
        start_pattern = _set_start_values(lam, mu, patterns, blocks, start_placement)
        if len(start_pattern) != len(subjects):
            missing = sorted(set(subjects) - set(start_pattern))
            raise ValueError(f"Could not match start patterns for subjects: {missing}")

    objective = gp.LinExpr()
    same_slot_expr = gp.LinExpr()
    for subject in subjects:
        for pattern in patterns[subject]:
            var = lam[(subject, pattern.index)]
            if pattern.internal_objective:
                objective.addTerms(pattern.internal_objective, var)
            if pattern.internal_same_slot:
                same_slot_expr.addTerms(pattern.internal_same_slot, var)

    for key, var in mu.items():
        cost = cross_costs[key]
        if cost.objective:
            objective.addTerms(cost.objective, var)
        if cost.same_slot_clashes:
            same_slot_expr.addTerms(cost.same_slot_clashes, var)

    if not args.no_partial_clash_cap:
        model.addConstr(same_slot_expr <= 10_000, name="partial_same_slot_clash_cap")

    joint_stats = _add_joint_cluster_consistency(
        model=model,
        lam=lam,
        mu=mu,
        patterns=patterns,
        cross_costs=cross_costs,
        joint_clusters=joint_clusters,
    )

    _add_cluster_cuts(
        model=model,
        lam=lam,
        mu=mu,
        patterns=patterns,
        cross_costs=cross_costs,
        cluster_cuts=cluster_cuts,
    )
    _add_window_cuts(
        model=model,
        lam=lam,
        mu=mu,
        patterns=patterns,
        cross_costs=cross_costs,
        day_index=day_index,
        window_cuts=window_cuts,
    )

    model.setObjective(objective, GRB.MINIMIZE)
    model.update()
    joint_cut_stats = _generate_joint_consistency_cuts(
        model=model,
        gp=gp,
        patterns=patterns,
        cross_costs=cross_costs,
        lam=lam,
        mu=mu,
        joint_clusters=joint_cut_clusters,
        rounds=args.joint_cut_rounds,
        max_cuts_per_round=args.joint_cut_max_cuts_per_round,
        tolerance=args.joint_cut_tolerance,
    )

    progress: list[dict[str, float]] = []

    def callback(model_cb: Any, where: int) -> None:
        if where == GRB.Callback.MIP:
            incumbent = model_cb.cbGet(GRB.Callback.MIP_OBJBST)
            best_bound = model_cb.cbGet(GRB.Callback.MIP_OBJBND)
            if incumbent < GRB.INFINITY or best_bound < GRB.INFINITY:
                progress.append(
                    {
                        "time_seconds": float(model_cb.cbGet(GRB.Callback.RUNTIME)),
                        "incumbent": float(incumbent),
                        "best_bound": float(best_bound),
                        "node_count": float(model_cb.cbGet(GRB.Callback.MIP_NODCNT)),
                    }
                )

    model.optimize(callback)

    progress_df = _deduplicate_progress(pd.DataFrame.from_records(progress))
    relaxation_incumbent = _safe_model_attr(model, "ObjVal") if model.SolCount else None
    certified_bound = _safe_model_attr(model, "ObjBound")
    gap_vs_start = None
    if start_full_objective is not None and certified_bound is not None:
        gap_vs_start = (start_full_objective - certified_bound) / abs(start_full_objective)

    summary = {
        "data_dir": str(args.data_dir),
        "start": str(args.start) if args.start is not None else "",
        "nb_days": args.nb_days,
        "objective_mode": args.objective_mode,
        "edge_score": args.edge_score,
        "requested_edge_count": args.edge_count,
        "edge_count": len(selected_edges),
        "selected_edges": ";".join(f"{left}|{right}" for left, right in selected_edges),
        "cluster_cut_count": len(cluster_cuts),
        "cluster_size": args.cluster_size,
        "cluster_cuts": ";".join(
            f"{'|'.join(cut['cluster'])}:{cut['bound']}" for cut in cluster_cuts
        ),
        "joint_cluster_count": len(joint_clusters),
        "joint_clusters": ";".join("|".join(cluster) for cluster in joint_clusters),
        "joint_tuple_count": joint_stats["tuple_count"],
        "joint_constraint_count": joint_stats["constraint_count"],
        "joint_cut_count": len(joint_cut_stats["cuts"]),
        "joint_cut_clusters": ";".join("|".join(cut["cluster"]) for cut in joint_cut_stats["cuts"]),
        "joint_cut_max_violation": joint_cut_stats["max_violation"],
        "window_cut_count": len(window_cuts),
        "separated_window_cut_count": len(separated_window_cuts),
        "window_clusters": ";".join("|".join(cluster) for cluster in window_clusters),
        "window_cuts": ";".join(
            f"{'|'.join(cut['cluster'])}@{cut['start_index']}-{cut['end_index']}:{cut['bound']}"
            for cut in window_cuts
        ),
        "subject_count": len(subjects),
        "pattern_count": sum(len(subject_patterns) for subject_patterns in patterns.values()),
        "mu_count": len(mu),
        "incompatible_mu_count": incompatible_mu_count,
        "partial_clash_cap": not args.no_partial_clash_cap,
        "relax_lambda": args.relax_lambda,
        "start_full_objective": start_full_objective,
        "relaxation_incumbent": relaxation_incumbent,
        "certified_full_lower_bound": certified_bound,
        "gap_vs_start": gap_vs_start,
        "status": int(model.Status),
        "mip_gap": _safe_model_attr(model, "MIPGap") if model.SolCount else None,
        "nodes": float(model.NodeCount),
        "iterations": float(model.IterCount),
        "work": float(model.Work),
        "runtime_seconds": float(model.Runtime),
    }
    return {"summary": summary, "progress": progress_df}


def _build_patterns(
    *,
    data: Any,
    blocks: dict[str, dict[str, Any]],
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
    exam_subjects: dict[str, str],
    objective_mode: str,
    weights: dict[str, float],
    start_placement: dict[str, tuple[pd.Timestamp, str]] | None,
) -> dict[str, list[Pattern]]:
    first_half_subjects = {"BUSINESS MANAGEMENT", "HISTORY", "ENGLISH A LAL"}
    patterns: dict[str, list[Pattern]] = {}
    for subject in sorted(blocks):
        raw_candidates = _generate_block_candidates(
            block_id=subject,
            block=blocks[subject],
            dates=data.dates,
            days=data.days,
            slots=tuple(data.slots),
            exam_lengths=exam_lengths,
            exam_subjects=exam_subjects,
            max_afternoon_minutes=180,
            first_half_subjects=first_half_subjects,
        )
        subject_patterns: list[Pattern] = []
        seen_signatures: set[tuple[tuple[str, pd.Timestamp, str], ...]] = set()
        for candidate in raw_candidates:
            assignment = candidate["assignment"]
            signature = _assignment_signature(assignment)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            subject_patterns.append(
                Pattern(
                    subject=subject,
                    index=len(subject_patterns),
                    assignment=assignment,
                    internal_objective=_internal_objective(
                        assignment,
                        pair_values=pair_values,
                        day_index=day_index,
                        objective_mode=objective_mode,
                        weights=weights,
                    ),
                    internal_same_slot=_internal_same_slot(assignment, pair_values=pair_values),
                )
            )

        if start_placement is not None:
            start_assignment = {exam: start_placement[exam] for exam in blocks[subject]["exams"]}
            start_signature = _assignment_signature(start_assignment)
            if start_signature not in seen_signatures:
                subject_patterns.append(
                    Pattern(
                        subject=subject,
                        index=len(subject_patterns),
                        assignment=start_assignment,
                        internal_objective=_internal_objective(
                            start_assignment,
                            pair_values=pair_values,
                            day_index=day_index,
                            objective_mode=objective_mode,
                            weights=weights,
                        ),
                        internal_same_slot=_internal_same_slot(start_assignment, pair_values=pair_values),
                    )
                )

        if not subject_patterns:
            raise ValueError(f"No feasible patterns generated for subject {subject!r}.")
        patterns[subject] = subject_patterns
    return patterns


def _internal_objective(
    assignment: dict[str, tuple[pd.Timestamp, str]],
    *,
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    objective_mode: str,
    weights: dict[str, float],
) -> float:
    total = 0.0
    items = list(assignment.items())
    for pos, (exam_i, placement_i) in enumerate(items):
        for exam_j, placement_j in items[pos + 1 :]:
            total += _pair_objective(
                exam_i,
                placement_i,
                exam_j,
                placement_j,
                pair_values=pair_values,
                pairs=None,
                day_index=day_index,
                objective_mode=objective_mode,
                weights=weights,
            )
    return total


def _internal_same_slot(
    assignment: dict[str, tuple[pd.Timestamp, str]],
    *,
    pair_values: dict[tuple[str, str], float],
) -> float:
    total = 0.0
    items = list(assignment.items())
    for pos, (exam_i, placement_i) in enumerate(items):
        for exam_j, placement_j in items[pos + 1 :]:
            if placement_i == placement_j:
                total += pair_values[tuple(sorted((exam_i, exam_j)))]
    return total


def _select_edges(
    *,
    data: Any,
    subjects: list[str],
    edge_count: int,
    edge_score: str,
    objective_mode: str,
    weights: dict[str, float],
    start_placement: dict[str, tuple[pd.Timestamp, str]] | None,
) -> list[tuple[str, str]]:
    if edge_count == 0:
        return []
    if edge_score == "start_objective" and start_placement is None:
        raise ValueError("--edge-score start_objective requires --start.")

    all_edges = [
        (left, right)
        for left_pos, left in enumerate(subjects)
        for right in subjects[left_pos + 1 :]
    ]

    scores: dict[tuple[str, str], float] = {}
    exam_subject = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    day_index = {date: idx for idx, date in enumerate(data.dates)}
    exams = data.exam_names
    for pos, exam_i in enumerate(exams):
        subject_i = exam_subject[exam_i]
        for exam_j in exams[pos + 1 :]:
            subject_j = exam_subject[exam_j]
            if subject_i == subject_j:
                continue
            key = tuple(sorted((subject_i, subject_j)))
            if edge_score == "pair_mass":
                value = float(data.pairs.loc[exam_i, exam_j])
            else:
                assert start_placement is not None
                value = _pair_objective(
                    exam_i,
                    start_placement[exam_i],
                    exam_j,
                    start_placement[exam_j],
                    pair_values=None,
                    pairs=data.pairs,
                    day_index=day_index,
                    objective_mode=objective_mode,
                    weights=weights,
                )
            scores[key] = scores.get(key, 0.0) + value

    ranked = [(edge, scores.get(edge, 0.0)) for edge in all_edges]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    if edge_count < 0:
        return [edge for edge, _score in ranked]
    return [edge for edge, score in ranked if score > 0][:edge_count]


def _select_clusters(
    *,
    data: Any,
    subjects: list[str],
    cluster_count: int,
    cluster_size: int,
    cluster_score: str,
    objective_mode: str,
    weights: dict[str, float],
    start_placement: dict[str, tuple[pd.Timestamp, str]] | None,
) -> list[tuple[str, ...]]:
    if cluster_count <= 0:
        return []
    if cluster_size < 2:
        raise ValueError("--cluster-size must be at least 2.")
    if cluster_score == "start_objective" and start_placement is None:
        raise ValueError("--cluster-score start_objective requires --start.")

    edge_scores = _subject_edge_scores(
        data=data,
        objective_mode=objective_mode,
        weights=weights,
        score_mode=cluster_score,
        start_placement=start_placement,
    )
    subject_scores: dict[str, float] = {subject: 0.0 for subject in subjects}
    adjacency: dict[str, dict[str, float]] = {subject: {} for subject in subjects}
    for (left, right), score in edge_scores.items():
        if score <= 0:
            continue
        subject_scores[left] = subject_scores.get(left, 0.0) + 0.5 * score
        subject_scores[right] = subject_scores.get(right, 0.0) + 0.5 * score
        adjacency.setdefault(left, {})[right] = adjacency.setdefault(left, {}).get(right, 0.0) + score
        adjacency.setdefault(right, {})[left] = adjacency.setdefault(right, {}).get(left, 0.0) + score

    ranked_edges = [(edge, score) for edge, score in edge_scores.items() if score > 0]
    ranked_edges.sort(key=lambda item: (-item[1], item[0]))

    clusters: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for (left, right), _score in ranked_edges:
        selected = [left, right]
        while len(selected) < min(cluster_size, len(subjects)):
            candidates = [subject for subject in subjects if subject not in selected]
            candidates.sort(
                key=lambda subject: (
                    -sum(adjacency.get(subject, {}).get(chosen, 0.0) for chosen in selected),
                    -subject_scores.get(subject, 0.0),
                    subject,
                )
            )
            selected.append(candidates[0])
        signature = tuple(sorted(selected))
        if signature in seen:
            continue
        seen.add(signature)
        clusters.append(signature)
        if len(clusters) >= cluster_count:
            break
    return clusters


def _subject_edge_scores(
    *,
    data: Any,
    objective_mode: str,
    weights: dict[str, float],
    score_mode: str,
    start_placement: dict[str, tuple[pd.Timestamp, str]] | None,
) -> dict[tuple[str, str], float]:
    if score_mode == "start_objective" and start_placement is None:
        raise ValueError("start_placement is required for start_objective edge scores.")

    exam_subject = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    day_index = {date: idx for idx, date in enumerate(data.dates)}
    scores: dict[tuple[str, str], float] = {}
    exams = data.exam_names
    for pos, exam_i in enumerate(exams):
        subject_i = exam_subject[exam_i]
        for exam_j in exams[pos + 1 :]:
            subject_j = exam_subject[exam_j]
            if subject_i == subject_j:
                continue
            key = tuple(sorted((subject_i, subject_j)))
            if score_mode == "pair_mass":
                value = float(data.pairs.loc[exam_i, exam_j])
            else:
                assert start_placement is not None
                value = _pair_objective(
                    exam_i,
                    start_placement[exam_i],
                    exam_j,
                    start_placement[exam_j],
                    pair_values=None,
                    pairs=data.pairs,
                    day_index=day_index,
                    objective_mode=objective_mode,
                    weights=weights,
                )
            scores[key] = scores.get(key, 0.0) + value
    return scores


def _with_cluster_edges(
    selected_edges: list[tuple[str, str]],
    clusters: list[tuple[str, ...]],
) -> list[tuple[str, str]]:
    edge_set = {tuple(sorted(edge)) for edge in selected_edges}
    for cluster in clusters:
        for pos, left in enumerate(cluster):
            for right in cluster[pos + 1 :]:
                edge_set.add(tuple(sorted((left, right))))
    return sorted(edge_set)


def _select_joint_clusters(
    *,
    data: Any,
    subjects: list[str],
    patterns: dict[str, list[Pattern]],
    joint_cluster_count: int,
    joint_max_tuples: int,
    objective_mode: str,
    weights: dict[str, float],
    start_placement: dict[str, tuple[pd.Timestamp, str]] | None,
) -> list[tuple[str, ...]]:
    if joint_cluster_count <= 0:
        return []
    edge_scores = _subject_edge_scores(
        data=data,
        objective_mode=objective_mode,
        weights=weights,
        score_mode="pair_mass",
        start_placement=start_placement,
    )
    triples: list[tuple[tuple[str, ...], float, int]] = []
    for first_pos, first in enumerate(subjects):
        for second_pos, second in enumerate(subjects[first_pos + 1 :], start=first_pos + 1):
            for third in subjects[second_pos + 1 :]:
                cluster = tuple(sorted((first, second, third)))
                tuple_count = 1
                for subject in cluster:
                    tuple_count *= len(patterns[subject])
                if tuple_count > joint_max_tuples:
                    continue
                score = sum(
                    edge_scores.get(tuple(sorted((left, right))), 0.0)
                    for left_pos, left in enumerate(cluster)
                    for right in cluster[left_pos + 1 :]
                )
                if score <= 0:
                    continue
                triples.append((cluster, score, tuple_count))
    triples.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [cluster for cluster, _score, _tuple_count in triples[:joint_cluster_count]]


def _add_joint_cluster_consistency(
    *,
    model: Any,
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    patterns: dict[str, list[Pattern]],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    joint_clusters: list[tuple[str, ...]],
) -> dict[str, int]:
    if not joint_clusters:
        return {"tuple_count": 0, "constraint_count": 0}
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("Joint-cluster consistency requires gurobipy.") from exc

    tuple_count = 0
    constraint_count = 0
    for cluster_index, cluster in enumerate(joint_clusters, start=1):
        if len(cluster) != 3:
            raise ValueError("Joint-cluster consistency currently supports triples only.")

        cluster_patterns = [patterns[subject] for subject in cluster]
        rho: dict[tuple[int, int, int], Any] = {}
        by_subject_pattern: dict[tuple[str, int], list[Any]] = {}
        by_pair_pattern: dict[tuple[str, int, str, int], list[Any]] = {}

        for pattern_combo in product(*cluster_patterns):
            if _combo_daily_length_infeasible(pattern_combo, cross_costs):
                continue
            combo_key = tuple(pattern.index for pattern in pattern_combo)
            var = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=_var_name("rho", cluster_index, *combo_key),
            )
            rho[combo_key] = var
            tuple_count += 1

            for subject, pattern in zip(cluster, pattern_combo):
                by_subject_pattern.setdefault((subject, pattern.index), []).append(var)

            for left_pos, left_subject in enumerate(cluster):
                left_pattern = pattern_combo[left_pos]
                for right_pos in range(left_pos + 1, len(cluster)):
                    right_subject = cluster[right_pos]
                    right_pattern = pattern_combo[right_pos]
                    edge = tuple(sorted((left_subject, right_subject)))
                    if edge == (left_subject, right_subject):
                        key = (left_subject, left_pattern.index, right_subject, right_pattern.index)
                    else:
                        key = (right_subject, right_pattern.index, left_subject, left_pattern.index)
                    if key not in mu:
                        raise ValueError(f"Joint cluster requires missing edge {edge}.")
                    by_pair_pattern.setdefault(key, []).append(var)

        if not rho:
            raise ValueError(f"Joint cluster {cluster!r} has no feasible pattern tuples.")

        model.addConstr(
            gp.quicksum(rho.values()) == 1,
            name=_var_name("rho_sum", cluster_index),
        )
        constraint_count += 1

        for subject in cluster:
            for pattern in patterns[subject]:
                model.addConstr(
                    gp.quicksum(by_subject_pattern.get((subject, pattern.index), []))
                    == lam[(subject, pattern.index)],
                    name=_var_name("rho_lambda", cluster_index, subject, pattern.index),
                )
                constraint_count += 1

        for left_pos, left_subject in enumerate(cluster):
            for right_subject in cluster[left_pos + 1 :]:
                edge = tuple(sorted((left_subject, right_subject)))
                left, right = edge
                for left_pattern in patterns[left]:
                    for right_pattern in patterns[right]:
                        key = (left, left_pattern.index, right, right_pattern.index)
                        model.addConstr(
                            gp.quicksum(by_pair_pattern.get(key, [])) == mu[key],
                            name=_var_name("rho_mu", cluster_index, left, left_pattern.index, right, right_pattern.index),
                        )
                        constraint_count += 1
    return {"tuple_count": tuple_count, "constraint_count": constraint_count}


def _combo_daily_length_infeasible(
    pattern_combo: tuple[Pattern, ...],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
) -> bool:
    for left_pos, left_pattern in enumerate(pattern_combo):
        for right_pattern in pattern_combo[left_pos + 1 :]:
            edge = tuple(sorted((left_pattern.subject, right_pattern.subject)))
            if edge == (left_pattern.subject, right_pattern.subject):
                key = (left_pattern.subject, left_pattern.index, right_pattern.subject, right_pattern.index)
            else:
                key = (right_pattern.subject, right_pattern.index, left_pattern.subject, left_pattern.index)
            if cross_costs[key].daily_length_infeasible:
                return True
    return False


def _generate_joint_consistency_cuts(
    *,
    model: Any,
    gp: Any,
    patterns: dict[str, list[Pattern]],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    joint_clusters: list[tuple[str, ...]],
    rounds: int,
    max_cuts_per_round: int,
    tolerance: float,
) -> dict[str, Any]:
    if not joint_clusters or rounds <= 0:
        return {"cuts": [], "max_violation": 0.0}

    generated: list[dict[str, Any]] = []
    max_violation = 0.0
    original_output_flag = int(model.Params.OutputFlag)
    model.setParam("OutputFlag", 0)

    try:
        for round_index in range(1, rounds + 1):
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                break

            added_this_round = 0
            for cluster in joint_clusters:
                cut = _separate_joint_consistency_cut(
                    gp=gp,
                    patterns=patterns,
                    cross_costs=cross_costs,
                    lam=lam,
                    mu=mu,
                    cluster=cluster,
                    tolerance=tolerance,
                )
                if cut is None:
                    continue

                expression = gp.LinExpr()
                for key, coefficient in cut["coefficients"].items():
                    if key[0] == "lambda":
                        _kind, subject, pattern_index = key
                        expression.addTerms(coefficient, lam[(subject, pattern_index)])
                    else:
                        _kind, left_subject, left_index, right_subject, right_index = key
                        expression.addTerms(coefficient, mu[(left_subject, left_index, right_subject, right_index)])

                model.addConstr(
                    expression <= cut["rhs"] + 1e-7,
                    name=_var_name("joint_sep_cut", round_index, len(generated) + 1),
                )
                generated.append(cut)
                max_violation = max(max_violation, float(cut["violation"]))
                added_this_round += 1
                if added_this_round >= max_cuts_per_round:
                    break

            model.update()
            if added_this_round == 0:
                break
    finally:
        model.setParam("OutputFlag", original_output_flag)
        model.reset()

    return {"cuts": generated, "max_violation": max_violation}


def _separate_joint_consistency_cut(
    *,
    gp: Any,
    patterns: dict[str, list[Pattern]],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    cluster: tuple[str, ...],
    tolerance: float,
) -> dict[str, Any] | None:
    keys: list[tuple[Any, ...]] = []
    value_by_key: dict[tuple[Any, ...], float] = {}

    for subject in cluster:
        for pattern in patterns[subject]:
            key = ("lambda", subject, pattern.index)
            keys.append(key)
            value_by_key[key] = float(lam[(subject, pattern.index)].X)

    for left_pos, left_subject in enumerate(cluster):
        for right_subject in cluster[left_pos + 1 :]:
            left, right = tuple(sorted((left_subject, right_subject)))
            for left_pattern in patterns[left]:
                for right_pattern in patterns[right]:
                    key = ("mu", left, left_pattern.index, right, right_pattern.index)
                    keys.append(key)
                    value_by_key[key] = float(mu[(left, left_pattern.index, right, right_pattern.index)].X)

    key_index = {key: index for index, key in enumerate(keys)}
    sep = gp.Model("full_joint_consistency_separator")
    sep.setParam("OutputFlag", 0)
    y_plus = sep.addVars(len(keys), lb=0.0, name="yp")
    y_minus = sep.addVars(len(keys), lb=0.0, name="ym")
    delta_plus = sep.addVar(lb=0.0, name="dp")
    delta_minus = sep.addVar(lb=0.0, name="dm")
    delta = delta_plus - delta_minus

    sep.addConstr(
        gp.quicksum(y_plus[index] + y_minus[index] for index in range(len(keys)))
        + delta_plus
        + delta_minus
        <= 1.0,
        name="normalization",
    )

    sep.setObjective(
        gp.quicksum((y_plus[index] - y_minus[index]) * value_by_key[key] for key, index in key_index.items())
        - delta,
        gp.GRB.MAXIMIZE,
    )

    cluster_patterns = [patterns[subject] for subject in cluster]
    for combo_index, pattern_combo in enumerate(product(*cluster_patterns)):
        if _combo_daily_length_infeasible(pattern_combo, cross_costs):
            continue

        active_indices: list[int] = []
        for pattern in pattern_combo:
            active_indices.append(key_index[("lambda", pattern.subject, pattern.index)])

        for left_pos, left_pattern in enumerate(pattern_combo):
            for right_pattern in pattern_combo[left_pos + 1 :]:
                left, right = tuple(sorted((left_pattern.subject, right_pattern.subject)))
                if left == left_pattern.subject:
                    left_index = left_pattern.index
                    right_index = right_pattern.index
                else:
                    left_index = right_pattern.index
                    right_index = left_pattern.index
                active_indices.append(key_index[("mu", left, left_index, right, right_index)])

        sep.addConstr(
            gp.quicksum(y_plus[index] - y_minus[index] for index in active_indices) <= delta,
            name=f"joint_tuple[{combo_index}]",
        )

    sep.optimize()
    if sep.Status != gp.GRB.OPTIMAL:
        return None

    violation = float(sep.ObjVal)
    if violation <= tolerance:
        return None

    coefficients: dict[tuple[Any, ...], float] = {}
    for key, index in key_index.items():
        coefficient = float(y_plus[index].X - y_minus[index].X)
        if abs(coefficient) > 1e-8:
            coefficients[key] = coefficient

    rhs = float(delta_plus.X - delta_minus.X)
    return {
        "cluster": cluster,
        "coefficients": coefficients,
        "rhs": rhs,
        "violation": violation,
    }


def _solve_cluster_cuts(
    *,
    clusters: list[tuple[str, ...]],
    patterns: dict[str, list[Pattern]],
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
    objective_mode: str,
    weights: dict[str, float],
    time_limit: float,
) -> list[dict[str, Any]]:
    if not clusters:
        return []
    cuts: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        result = _solve_cluster_lower_bound(
            cluster=cluster,
            patterns=patterns,
            pair_values=pair_values,
            day_index=day_index,
            exam_lengths=exam_lengths,
            objective_mode=objective_mode,
            weights=weights,
            time_limit=time_limit,
        )
        bound = result["bound"]
        if bound is not None and math.isfinite(bound) and bound > 1e-9:
            cuts.append(
                {
                    "cluster_index": cluster_index,
                    "cluster": cluster,
                    "bound": float(bound),
                    "status": result["status"],
                    "objective": result["objective"],
                    "gap": result["gap"],
                    "runtime_seconds": result["runtime_seconds"],
                    "nodes": result["nodes"],
                    "mu_count": result["mu_count"],
                }
            )
    return cuts


def _solve_window_cuts(
    *,
    clusters: list[tuple[str, ...]],
    patterns: dict[str, list[Pattern]],
    dates: list[pd.Timestamp],
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
    objective_mode: str,
    weights: dict[str, float],
    window_length: int,
    cut_limit: int,
    time_limit: float,
) -> list[dict[str, Any]]:
    if not clusters or cut_limit <= 0:
        return []
    if window_length < 1:
        raise ValueError("--window-length must be positive.")

    candidates: list[dict[str, Any]] = []
    for cluster in clusters:
        for start_index in range(0, len(dates) - window_length + 1):
            end_index = start_index + window_length - 1
            restricted_patterns: dict[str, list[Pattern]] = {}
            for subject in cluster:
                kept = [
                    pattern
                    for pattern in patterns[subject]
                    if _pattern_inside_window(pattern, day_index=day_index, start_index=start_index, end_index=end_index)
                ]
                if not kept:
                    break
                restricted_patterns[subject] = kept
            if len(restricted_patterns) != len(cluster):
                continue

            result = _solve_cluster_lower_bound(
                cluster=cluster,
                patterns=restricted_patterns,
                pair_values=pair_values,
                day_index=day_index,
                exam_lengths=exam_lengths,
                objective_mode=objective_mode,
                weights=weights,
                time_limit=time_limit,
            )
            bound = result["bound"]
            if bound is None or not math.isfinite(bound) or bound <= 1e-9:
                continue
            candidates.append(
                {
                    "cluster": cluster,
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_date": dates[start_index],
                    "end_date": dates[end_index],
                    "bound": float(bound),
                    "status": result["status"],
                    "objective": result["objective"],
                    "gap": result["gap"],
                    "runtime_seconds": result["runtime_seconds"],
                    "nodes": result["nodes"],
                    "mu_count": result["mu_count"],
                    "restricted_pattern_count": sum(len(restricted_patterns[subject]) for subject in cluster),
                }
            )

    candidates.sort(
        key=lambda cut: (
            -float(cut["bound"]),
            int(cut["start_index"]),
            tuple(cut["cluster"]),
        )
    )
    return candidates[:cut_limit]


def _separate_window_cuts(
    *,
    data: Any,
    subjects: list[str],
    patterns: dict[str, list[Pattern]],
    base_edges: list[tuple[str, str]],
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
    objective_mode: str,
    weights: dict[str, float],
    start_placement: dict[str, tuple[pd.Timestamp, str]] | None,
    cluster_count: int,
    cluster_size: int,
    window_length: int,
    cut_count: int,
    max_subproblems: int,
    min_activation: float,
    cut_time_limit: float,
    partial_clash_cap: bool,
) -> list[dict[str, Any]]:
    if cut_count <= 0:
        return []
    candidate_clusters = _select_clusters(
        data=data,
        subjects=subjects,
        cluster_count=cluster_count,
        cluster_size=cluster_size,
        cluster_score="pair_mass",
        objective_mode=objective_mode,
        weights=weights,
        start_placement=start_placement,
    )
    if not candidate_clusters:
        return []

    lp_edges = _with_cluster_edges(base_edges, candidate_clusters)
    lp_result = _solve_relaxed_master_for_separation(
        patterns=patterns,
        selected_edges=lp_edges,
        pair_values=pair_values,
        day_index=day_index,
        exam_lengths=exam_lengths,
        objective_mode=objective_mode,
        weights=weights,
        partial_clash_cap=partial_clash_cap,
    )

    candidates: list[dict[str, Any]] = []
    for cluster in candidate_clusters:
        for start_index in range(0, len(data.dates) - window_length + 1):
            end_index = start_index + window_length - 1
            z_values = [
                sum(
                    lp_result["lambda_values"].get((subject, pattern.index), 0.0)
                    for pattern in patterns[subject]
                    if _pattern_inside_window(
                        pattern,
                        day_index=day_index,
                        start_index=start_index,
                        end_index=end_index,
                    )
                )
                for subject in cluster
            ]
            activation = sum(z_values) - len(cluster) + 1.0
            if activation <= min_activation:
                continue
            candidates.append(
                {
                    "cluster": cluster,
                    "start_index": start_index,
                    "end_index": end_index,
                    "activation": float(activation),
                    "min_z": float(min(z_values)),
                    "avg_z": float(sum(z_values) / len(z_values)),
                }
            )

    candidates.sort(key=lambda cut: (-cut["activation"], -cut["min_z"], tuple(cut["cluster"]), cut["start_index"]))
    separated: list[dict[str, Any]] = []
    for candidate in candidates[:max_subproblems]:
        cluster = candidate["cluster"]
        start_index = int(candidate["start_index"])
        end_index = int(candidate["end_index"])
        restricted_patterns: dict[str, list[Pattern]] = {}
        for subject in cluster:
            kept = [
                pattern
                for pattern in patterns[subject]
                if _pattern_inside_window(
                    pattern,
                    day_index=day_index,
                    start_index=start_index,
                    end_index=end_index,
                )
            ]
            if not kept:
                break
            restricted_patterns[subject] = kept
        if len(restricted_patterns) != len(cluster):
            continue

        result = _solve_cluster_lower_bound(
            cluster=cluster,
            patterns=restricted_patterns,
            pair_values=pair_values,
            day_index=day_index,
            exam_lengths=exam_lengths,
            objective_mode=objective_mode,
            weights=weights,
            time_limit=cut_time_limit,
        )
        bound = result["bound"]
        if bound is None or not math.isfinite(bound) or bound <= 1e-9:
            continue

        current_cost = _cluster_cost_from_relaxed_solution(
            cluster=cluster,
            patterns=patterns,
            lambda_values=lp_result["lambda_values"],
            mu_values=lp_result["mu_values"],
            cross_costs=lp_result["cross_costs"],
        )
        violation = float(bound) * float(candidate["activation"]) - current_cost
        if violation <= 1e-6:
            continue

        cut = {
            "cluster": cluster,
            "start_index": start_index,
            "end_index": end_index,
            "start_date": data.dates[start_index],
            "end_date": data.dates[end_index],
            "bound": float(bound),
            "activation": float(candidate["activation"]),
            "current_cost": current_cost,
            "violation": violation,
            "status": result["status"],
            "objective": result["objective"],
            "gap": result["gap"],
            "runtime_seconds": result["runtime_seconds"],
            "nodes": result["nodes"],
            "mu_count": result["mu_count"],
            "restricted_pattern_count": sum(len(restricted_patterns[subject]) for subject in cluster),
            "separated": True,
        }
        separated.append(cut)

    separated.sort(key=lambda cut: (-cut["violation"], -cut["activation"], -cut["bound"]))
    return separated[:cut_count]


def _solve_relaxed_master_for_separation(
    *,
    patterns: dict[str, list[Pattern]],
    selected_edges: list[tuple[str, str]],
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
    objective_mode: str,
    weights: dict[str, float],
    partial_clash_cap: bool,
) -> dict[str, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("Window-cut separation requires gurobipy.") from exc

    model = gp.Model("window_cut_separation_lp")
    model.setParam("OutputFlag", 0)
    subjects = sorted(patterns)
    lam: dict[tuple[str, int], Any] = {}
    for subject in subjects:
        for pattern in patterns[subject]:
            lam[(subject, pattern.index)] = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=_var_name("slambda", subject, pattern.index),
            )

    mu: dict[tuple[str, int, str, int], Any] = {}
    cross_costs: dict[tuple[str, int, str, int], CrossCost] = {}
    for left_subject, right_subject in selected_edges:
        for left_pattern in patterns[left_subject]:
            for right_pattern in patterns[right_subject]:
                cost = _cross_cost(
                    left_pattern.assignment,
                    right_pattern.assignment,
                    pair_values=pair_values,
                    day_index=day_index,
                    exam_lengths=exam_lengths,
                    objective_mode=objective_mode,
                    weights=weights,
                )
                key = (left_subject, left_pattern.index, right_subject, right_pattern.index)
                cross_costs[key] = cost
                mu[key] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=0.0 if cost.daily_length_infeasible else 1.0,
                    name=_var_name("smu", left_subject, left_pattern.index, right_subject, right_pattern.index),
                )

    model.update()
    for subject in subjects:
        model.addConstr(
            gp.quicksum(lam[(subject, pattern.index)] for pattern in patterns[subject]) == 1,
            name=_var_name("schoose", subject),
        )

    for left_subject, right_subject in selected_edges:
        for left_pattern in patterns[left_subject]:
            model.addConstr(
                gp.quicksum(
                    mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                    for right_pattern in patterns[right_subject]
                )
                == lam[(left_subject, left_pattern.index)],
                name=_var_name("smu_left", left_subject, left_pattern.index, right_subject),
            )
        for right_pattern in patterns[right_subject]:
            model.addConstr(
                gp.quicksum(
                    mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                    for left_pattern in patterns[left_subject]
                )
                == lam[(right_subject, right_pattern.index)],
                name=_var_name("smu_right", left_subject, right_subject, right_pattern.index),
            )

    objective = gp.LinExpr()
    same_slot_expr = gp.LinExpr()
    for subject in subjects:
        for pattern in patterns[subject]:
            var = lam[(subject, pattern.index)]
            if pattern.internal_objective:
                objective.addTerms(pattern.internal_objective, var)
            if pattern.internal_same_slot:
                same_slot_expr.addTerms(pattern.internal_same_slot, var)
    for key, var in mu.items():
        cost = cross_costs[key]
        if cost.objective:
            objective.addTerms(cost.objective, var)
        if cost.same_slot_clashes:
            same_slot_expr.addTerms(cost.same_slot_clashes, var)

    if partial_clash_cap:
        model.addConstr(same_slot_expr <= 10_000, name="separation_partial_same_slot_cap")

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
        raise ValueError(f"Separation LP ended with status {model.Status}.")

    return {
        "lambda_values": {key: float(var.X) for key, var in lam.items()},
        "mu_values": {key: float(var.X) for key, var in mu.items()},
        "cross_costs": cross_costs,
        "objective": _safe_model_attr(model, "ObjVal"),
        "bound": _safe_model_attr(model, "ObjBound"),
    }


def _cluster_cost_from_relaxed_solution(
    *,
    cluster: tuple[str, ...],
    patterns: dict[str, list[Pattern]],
    lambda_values: dict[tuple[str, int], float],
    mu_values: dict[tuple[str, int, str, int], float],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
) -> float:
    total = 0.0
    for subject in cluster:
        for pattern in patterns[subject]:
            total += pattern.internal_objective * lambda_values.get((subject, pattern.index), 0.0)
    for left_pos, left_subject in enumerate(cluster):
        for right_subject in cluster[left_pos + 1 :]:
            edge = tuple(sorted((left_subject, right_subject)))
            left, right = edge
            for left_pattern in patterns[left]:
                for right_pattern in patterns[right]:
                    key = (left, left_pattern.index, right, right_pattern.index)
                    total += cross_costs[key].objective * mu_values.get(key, 0.0)
    return float(total)


def _pattern_inside_window(
    pattern: Pattern,
    *,
    day_index: dict[pd.Timestamp, int],
    start_index: int,
    end_index: int,
) -> bool:
    return all(
        start_index <= day_index[pd.Timestamp(date).normalize()] <= end_index
        for date, _slot in pattern.assignment.values()
    )


def _solve_cluster_lower_bound(
    *,
    cluster: tuple[str, ...],
    patterns: dict[str, list[Pattern]],
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
    objective_mode: str,
    weights: dict[str, float],
    time_limit: float,
) -> dict[str, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("Cluster cuts require gurobipy.") from exc

    model = gp.Model(_var_name("cluster_pattern_bound", "_".join(cluster)))
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    lam: dict[tuple[str, int], Any] = {}
    for subject in cluster:
        for pattern in patterns[subject]:
            lam[(subject, pattern.index)] = model.addVar(
                vtype=GRB.BINARY,
                lb=0.0,
                ub=1.0,
                name=_var_name("clambda", subject, pattern.index),
            )

    mu: dict[tuple[str, int, str, int], Any] = {}
    cross_costs: dict[tuple[str, int, str, int], CrossCost] = {}
    for left_pos, left_subject in enumerate(cluster):
        for right_subject in cluster[left_pos + 1 :]:
            for left_pattern in patterns[left_subject]:
                for right_pattern in patterns[right_subject]:
                    cost = _cross_cost(
                        left_pattern.assignment,
                        right_pattern.assignment,
                        pair_values=pair_values,
                        day_index=day_index,
                        exam_lengths=exam_lengths,
                        objective_mode=objective_mode,
                        weights=weights,
                    )
                    key = (left_subject, left_pattern.index, right_subject, right_pattern.index)
                    cross_costs[key] = cost
                    mu[key] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=0.0,
                        ub=0.0 if cost.daily_length_infeasible else 1.0,
                        name=_var_name("cmu", left_subject, left_pattern.index, right_subject, right_pattern.index),
                    )

    model.update()

    for subject in cluster:
        model.addConstr(
            gp.quicksum(lam[(subject, pattern.index)] for pattern in patterns[subject]) == 1,
            name=_var_name("cchoose", subject),
        )

    for left_pos, left_subject in enumerate(cluster):
        for right_subject in cluster[left_pos + 1 :]:
            for left_pattern in patterns[left_subject]:
                model.addConstr(
                    gp.quicksum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for right_pattern in patterns[right_subject]
                    )
                    == lam[(left_subject, left_pattern.index)],
                    name=_var_name("cmu_left", left_subject, left_pattern.index, right_subject),
                )
            for right_pattern in patterns[right_subject]:
                model.addConstr(
                    gp.quicksum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for left_pattern in patterns[left_subject]
                    )
                    == lam[(right_subject, right_pattern.index)],
                    name=_var_name("cmu_right", left_subject, right_subject, right_pattern.index),
                )

    objective = gp.LinExpr()
    same_slot_expr = gp.LinExpr()
    for subject in cluster:
        for pattern in patterns[subject]:
            var = lam[(subject, pattern.index)]
            if pattern.internal_objective:
                objective.addTerms(pattern.internal_objective, var)
            if pattern.internal_same_slot:
                same_slot_expr.addTerms(pattern.internal_same_slot, var)
    for key, var in mu.items():
        cost = cross_costs[key]
        if cost.objective:
            objective.addTerms(cost.objective, var)
        if cost.same_slot_clashes:
            same_slot_expr.addTerms(cost.same_slot_clashes, var)

    model.addConstr(same_slot_expr <= 10_000, name="cluster_same_slot_cap")
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    return {
        "bound": _safe_model_attr(model, "ObjBound"),
        "objective": _safe_model_attr(model, "ObjVal") if model.SolCount else None,
        "status": int(model.Status),
        "gap": _safe_model_attr(model, "MIPGap") if model.SolCount else None,
        "runtime_seconds": float(model.Runtime),
        "nodes": float(model.NodeCount),
        "mu_count": len(mu),
    }


def _add_window_cuts(
    *,
    model: Any,
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    patterns: dict[str, list[Pattern]],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    day_index: dict[pd.Timestamp, int],
    window_cuts: list[dict[str, Any]],
) -> None:
    if not window_cuts:
        return
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise ImportError("Window cuts require gurobipy.") from exc

    for cut_index, cut in enumerate(window_cuts, start=1):
        cluster = cut["cluster"]
        start_index = int(cut["start_index"])
        end_index = int(cut["end_index"])
        cluster_cost = _cluster_cost_expression(
            cluster=cluster,
            lam=lam,
            mu=mu,
            patterns=patterns,
            cross_costs=cross_costs,
        )
        activation = gp.LinExpr(-len(cluster) + 1.0)
        for subject in cluster:
            for pattern in patterns[subject]:
                if _pattern_inside_window(
                    pattern,
                    day_index=day_index,
                    start_index=start_index,
                    end_index=end_index,
                ):
                    activation.addTerms(1.0, lam[(subject, pattern.index)])
        model.addConstr(
            cluster_cost >= float(cut["bound"]) * activation,
            name=_var_name("window_cluster_lb", cut_index),
        )


def _add_cluster_cuts(
    *,
    model: Any,
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    patterns: dict[str, list[Pattern]],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    cluster_cuts: list[dict[str, Any]],
) -> None:
    if not cluster_cuts:
        return

    for cut in cluster_cuts:
        cluster = cut["cluster"]
        expression = _cluster_cost_expression(
            cluster=cluster,
            lam=lam,
            mu=mu,
            patterns=patterns,
            cross_costs=cross_costs,
        )
        model.addConstr(
            expression >= float(cut["bound"]),
            name=_var_name("cluster_cost_lb", cut["cluster_index"]),
        )


def _cluster_cost_expression(
    *,
    cluster: tuple[str, ...],
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    patterns: dict[str, list[Pattern]],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
) -> Any:
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise ImportError("Cluster cuts require gurobipy.") from exc

    expression = gp.LinExpr()
    for subject in cluster:
        for pattern in patterns[subject]:
            if pattern.internal_objective:
                expression.addTerms(pattern.internal_objective, lam[(subject, pattern.index)])

    for left_pos, left_subject in enumerate(cluster):
        for right_subject in cluster[left_pos + 1 :]:
            edge = tuple(sorted((left_subject, right_subject)))
            left, right = edge
            for left_pattern in patterns[left]:
                for right_pattern in patterns[right]:
                    key = (left, left_pattern.index, right, right_pattern.index)
                    if key not in mu:
                        raise ValueError(f"Cluster cut requires missing edge {left!r}, {right!r}.")
                    cost = cross_costs[key]
                    if cost.objective:
                        expression.addTerms(cost.objective, mu[key])
    return expression


def _set_start_values(
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    patterns: dict[str, list[Pattern]],
    blocks: dict[str, dict[str, Any]],
    start_placement: dict[str, tuple[pd.Timestamp, str]],
) -> dict[str, int]:
    start_pattern: dict[str, int] = {}
    for subject, subject_patterns in patterns.items():
        start_assignment = {exam: start_placement[exam] for exam in blocks[subject]["exams"]}
        start_signature = _assignment_signature(start_assignment)
        for pattern in subject_patterns:
            is_current = _assignment_signature(pattern.assignment) == start_signature
            lam[(subject, pattern.index)].Start = 1.0 if is_current else 0.0
            if is_current:
                start_pattern[subject] = pattern.index

    for key, var in mu.items():
        left_subject, left_index, right_subject, right_index = key
        var.Start = 1.0 if (
            start_pattern.get(left_subject) == left_index
            and start_pattern.get(right_subject) == right_index
        ) else 0.0
    return start_pattern


def _load_timetable(path: Path) -> pd.DataFrame:
    timetable = pd.read_csv(path)
    timetable["Date"] = pd.to_datetime(timetable["Date"], dayfirst=True).dt.normalize()
    return timetable


def _deduplicate_progress(progress: pd.DataFrame) -> pd.DataFrame:
    if progress.empty:
        return progress
    progress = progress.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["incumbent", "best_bound"])
    progress["time_seconds"] = progress["time_seconds"].round(3)
    return progress.drop_duplicates(subset=["time_seconds", "incumbent", "best_bound"]).reset_index(drop=True)


def _plot_progress(progress: pd.DataFrame, output: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(10, 4.2))
    if not progress.empty:
        ax.plot(progress["time_seconds"], progress["incumbent"], label="Relaxation incumbent", color="#d62728", linewidth=3.0)
        ax.plot(progress["time_seconds"], progress["best_bound"], label="Certified lower bound", color="#1f77b4", linewidth=3.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_clock_time))
    ax.set_xlabel("Time (hh:mm:ss)")
    ax.set_ylabel("Objective")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if not progress.empty:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _format_clock_time(seconds: float, _pos: int | None = None) -> str:
    if pd.isna(seconds):
        return ""
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _safe_model_attr(model: Any, attr_name: str) -> float | None:
    try:
        return float(getattr(model, attr_name))
    except Exception:
        return None


if __name__ == "__main__":
    main()
