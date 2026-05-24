from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from src.anthony_model import DEFAULT_SLOTS, mip_objective_value, parse_day_series, prepare_toy_inputs, timetable_from_placement


WEIGHTS = {"a": 64, "b": 32, "c": 16, "d": 8, "e": 4, "f": 2, "g": 1}


@dataclass(frozen=True)
class Pattern:
    subject: str
    index: int
    assignment: dict[str, tuple[pd.Timestamp, str]]
    internal_objective: float
    internal_same_slot_clashes: float


@dataclass(frozen=True)
class CrossCost:
    objective: float
    same_slot_clashes: float
    daily_length_infeasible: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the toy instance with a subject/block pattern-pair formulation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--node-limit", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("toy_pattern_model_timetable.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("toy_pattern_model_summary.csv"))
    parser.add_argument("--progress-output", type=Path, default=Path("toy_pattern_model_progress.csv"))
    parser.add_argument("--log-output", type=Path, default=Path("toy_pattern_model.log"))
    parser.add_argument("--plot-output", type=Path, default=Path("toy_pattern_model_bounds.png"))
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument("--relax-lambda", action="store_true")
    parser.add_argument("--mu-binary", action="store_true")
    parser.add_argument("--joint-community-count", type=int, default=0)
    parser.add_argument("--joint-community-size", type=int, default=4)
    parser.add_argument("--joint-community-max-tuples", type=int, default=1_000_000)
    parser.add_argument("--joint-cut-count", type=int, default=0)
    parser.add_argument("--joint-cut-community-size", type=int, default=3)
    parser.add_argument("--joint-cut-max-tuples", type=int, default=250_000)
    parser.add_argument("--joint-cut-rounds", type=int, default=3)
    parser.add_argument("--joint-cut-max-cuts-per-round", type=int, default=3)
    parser.add_argument("--joint-cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--clique-cut-rounds", type=int, default=0)
    parser.add_argument("--clique-cut-max-cuts-per-round", type=int, default=10)
    parser.add_argument("--clique-cut-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    raw_exams = pd.read_csv(args.data_dir / "Toy exam list.csv")
    raw_pairs = pd.read_csv(args.data_dir / "Exam pairs.csv")
    exams, days, pairs = prepare_toy_inputs(raw_exams, raw_pairs)
    pairs = _clean_pairs(pairs)
    days = days.copy()
    days["Date"] = parse_day_series(days["Date"])

    started = time.perf_counter()
    result = solve_pattern_model(
        exams=exams,
        days=days,
        pairs=pairs,
        time_limit=args.time_limit,
        node_limit=args.node_limit,
        log_output=args.log_output,
        output_flag=args.output_flag,
        relax_lambda=args.relax_lambda,
        mu_binary=args.mu_binary,
        joint_community_count=args.joint_community_count,
        joint_community_size=args.joint_community_size,
        joint_community_max_tuples=args.joint_community_max_tuples,
        joint_cut_count=args.joint_cut_count,
        joint_cut_community_size=args.joint_cut_community_size,
        joint_cut_max_tuples=args.joint_cut_max_tuples,
        joint_cut_rounds=args.joint_cut_rounds,
        joint_cut_max_cuts_per_round=args.joint_cut_max_cuts_per_round,
        joint_cut_tolerance=args.joint_cut_tolerance,
        clique_cut_rounds=args.clique_cut_rounds,
        clique_cut_max_cuts_per_round=args.clique_cut_max_cuts_per_round,
        clique_cut_tolerance=args.clique_cut_tolerance,
        progress_output=args.progress_output,
        plot_output=args.plot_output,
    )
    elapsed = time.perf_counter() - started

    timetable = result["timetable"]
    timetable.to_csv(args.output, index=False)

    if args.relax_lambda:
        validation_value = None
        reported_same_slot_clashes = None
    else:
        validation_value = mip_objective_value(timetable, pairs, days, mode="anthony_appendix")
        reported_same_slot_clashes = result["same_slot_clashes"]
    summary = {
        "status": result["status"],
        "objective": result["objective"],
        "validation_objective": validation_value,
        "best_bound": result["best_bound"],
        "gap": result["gap"],
        "nodes": result["nodes"],
        "iterations": result["iterations"],
        "work": result["work"],
        "runtime_seconds": result["runtime_seconds"],
        "elapsed_seconds": elapsed,
        "pattern_count": result["pattern_count"],
        "mu_count": result["mu_count"],
        "theta_count": result["theta_count"],
        "joint_community_count": result["joint_community_count"],
        "joint_communities": result["joint_communities"],
        "joint_cut_count": result["joint_cut_count"],
        "joint_cut_communities": result["joint_cut_communities"],
        "joint_cut_max_violation": result["joint_cut_max_violation"],
        "clique_cut_count": result["clique_cut_count"],
        "clique_cut_max_violation": result["clique_cut_max_violation"],
        "incompatible_mu_count": result["incompatible_mu_count"],
        "same_slot_clashes": reported_same_slot_clashes,
        "max_clashes": 15,
        "relax_lambda": args.relax_lambda,
        "mu_binary": args.mu_binary,
        "joint_community_size": args.joint_community_size,
        "joint_community_max_tuples": args.joint_community_max_tuples,
        "joint_cut_community_size": args.joint_cut_community_size,
        "joint_cut_max_tuples": args.joint_cut_max_tuples,
        "joint_cut_rounds": args.joint_cut_rounds,
    }
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)

    print("Pattern model seconds:", round(elapsed, 6))
    print("Pattern model status:", result["status"])
    print("Pattern model objective:", result["objective"])
    print("Validation objective:", validation_value)
    if args.relax_lambda:
        print("Relaxation run: extracted timetable is only a rounded diagnostic, not a feasible incumbent.")
    print("Best bound:", result["best_bound"])
    print("Gap:", result["gap"])
    print("Nodes:", result["nodes"])
    print("Iterations:", result["iterations"])
    print("Work:", result["work"])
    print("Patterns:", result["pattern_count"])
    print("Mu variables:", result["mu_count"])
    print("Theta variables:", result["theta_count"])
    print("Joint communities:", result["joint_communities"])
    print("Generated joint cuts:", result["joint_cut_count"])
    print("Joint cut communities:", result["joint_cut_communities"])
    print("Generated clique cuts:", result["clique_cut_count"])
    print("Incompatible mu variables fixed to zero:", result["incompatible_mu_count"])
    print("Same-slot clashes:", result["same_slot_clashes"])
    print(f"Saved timetable to {args.output}")
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved progress to {args.progress_output}")
    print(f"Saved log to {args.log_output}")
    print(f"Saved bound plot to {args.plot_output}")


def solve_pattern_model(
    *,
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    time_limit: float | None,
    node_limit: float | None,
    log_output: Path,
    output_flag: int,
    relax_lambda: bool,
    mu_binary: bool,
    joint_community_count: int,
    joint_community_size: int,
    joint_community_max_tuples: int,
    joint_cut_count: int,
    joint_cut_community_size: int,
    joint_cut_max_tuples: int,
    joint_cut_rounds: int,
    joint_cut_max_cuts_per_round: int,
    joint_cut_tolerance: float,
    clique_cut_rounds: int,
    clique_cut_max_cuts_per_round: int,
    clique_cut_tolerance: float,
    progress_output: Path,
    plot_output: Path,
) -> dict[str, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("run_toy_pattern_model.py requires gurobipy.") from exc

    dates = parse_day_series(days["Date"]).tolist()
    date_index = {date: idx for idx, date in enumerate(dates)}
    exam_lengths = dict(zip(exams["Full Name"].astype(str), exams["Length"].astype(float)))
    blocks = _build_blocks(exams)
    patterns = _generate_all_patterns(
        blocks=blocks,
        exams=exams,
        days=days,
        pairs=pairs,
        dates=dates,
        date_index=date_index,
        exam_lengths=exam_lengths,
    )

    model = gp.Model("toy_subject_pattern_model")
    model.setParam("OutputFlag", output_flag)
    model.setParam("LogFile", str(log_output))
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if node_limit is not None:
        model.setParam("NodeLimit", node_limit)

    lambda_type = GRB.CONTINUOUS if relax_lambda else GRB.BINARY
    mu_type = GRB.BINARY if mu_binary else GRB.CONTINUOUS

    lam: dict[tuple[str, int], Any] = {}
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            lam[(subject, pattern.index)] = model.addVar(
                vtype=lambda_type,
                lb=0.0,
                ub=1.0,
                name=_var_name("lambda", subject, pattern.index),
            )

    subjects = list(patterns)
    mu: dict[tuple[str, int, str, int], Any] = {}
    cross_costs: dict[tuple[str, int, str, int], CrossCost] = {}
    incompatible_count = 0
    for left_pos, left_subject in enumerate(subjects):
        for right_subject in subjects[left_pos + 1 :]:
            for left_pattern in patterns[left_subject]:
                for right_pattern in patterns[right_subject]:
                    cost = _cross_cost(
                        left_pattern.assignment,
                        right_pattern.assignment,
                        pairs=pairs,
                        date_index=date_index,
                        exam_lengths=exam_lengths,
                    )
                    key = (left_subject, left_pattern.index, right_subject, right_pattern.index)
                    cross_costs[key] = cost
                    ub = 0.0 if cost.daily_length_infeasible else 1.0
                    if cost.daily_length_infeasible:
                        incompatible_count += 1
                    mu[key] = model.addVar(
                        vtype=mu_type,
                        lb=0.0,
                        ub=ub,
                        name=_var_name("mu", left_subject, left_pattern.index, right_subject, right_pattern.index),
                    )

    model.update()

    for subject, subject_patterns in patterns.items():
        model.addConstr(
            gp.quicksum(lam[(subject, pattern.index)] for pattern in subject_patterns) == 1,
            name=_var_name("choose_pattern", subject),
        )

    for left_pos, left_subject in enumerate(subjects):
        for right_subject in subjects[left_pos + 1 :]:
            right_patterns = patterns[right_subject]
            left_patterns = patterns[left_subject]
            for left_pattern in left_patterns:
                model.addConstr(
                    gp.quicksum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for right_pattern in right_patterns
                    )
                    == lam[(left_subject, left_pattern.index)],
                    name=_var_name("mu_left", left_subject, left_pattern.index, right_subject),
                )
            for right_pattern in right_patterns:
                model.addConstr(
                    gp.quicksum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for left_pattern in left_patterns
                    )
                    == lam[(right_subject, right_pattern.index)],
                    name=_var_name("mu_right", left_subject, right_subject, right_pattern.index),
                )

    joint_stats = _add_joint_community_constraints(
        model=model,
        gp=gp,
        patterns=patterns,
        blocks=blocks,
        pairs=pairs,
        cross_costs=cross_costs,
        lam=lam,
        mu=mu,
        subjects=subjects,
        community_count=joint_community_count,
        community_size=joint_community_size,
        max_tuples=joint_community_max_tuples,
    )

    clash_expr = gp.LinExpr()
    objective = gp.LinExpr()
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            var = lam[(subject, pattern.index)]
            if pattern.internal_objective:
                objective.addTerms(pattern.internal_objective, var)
            if pattern.internal_same_slot_clashes:
                clash_expr.addTerms(pattern.internal_same_slot_clashes, var)

    for key, var in mu.items():
        cost = cross_costs[key]
        if cost.objective:
            objective.addTerms(cost.objective, var)
        if cost.same_slot_clashes:
            clash_expr.addTerms(cost.same_slot_clashes, var)

    model.addConstr(clash_expr <= 15, name="max_same_slot_clashes")
    model.setObjective(objective, GRB.MINIMIZE)
    model.update()

    clique_cut_stats = _generate_incompatibility_clique_cuts(
        model=model,
        gp=gp,
        patterns=patterns,
        cross_costs=cross_costs,
        lam=lam,
        subjects=subjects,
        relax_lambda=relax_lambda,
        rounds=clique_cut_rounds,
        max_cuts_per_round=clique_cut_max_cuts_per_round,
        tolerance=clique_cut_tolerance,
    )

    joint_cut_stats = _generate_joint_consistency_cuts(
        model=model,
        gp=gp,
        patterns=patterns,
        blocks=blocks,
        pairs=pairs,
        cross_costs=cross_costs,
        lam=lam,
        mu=mu,
        subjects=subjects,
        relax_lambda=relax_lambda,
        mu_binary=mu_binary,
        community_count=joint_cut_count,
        community_size=joint_cut_community_size,
        max_tuples=joint_cut_max_tuples,
        rounds=joint_cut_rounds,
        max_cuts_per_round=joint_cut_max_cuts_per_round,
        tolerance=joint_cut_tolerance,
    )

    progress: list[dict[str, float]] = []

    def callback(cb_model: Any, where: int) -> None:
        if where == GRB.Callback.MIP:
            runtime = float(cb_model.cbGet(GRB.Callback.RUNTIME))
            node_count = float(cb_model.cbGet(GRB.Callback.MIP_NODCNT))
            incumbent = float(cb_model.cbGet(GRB.Callback.MIP_OBJBST))
            bound = float(cb_model.cbGet(GRB.Callback.MIP_OBJBND))
            if abs(incumbent) >= 1e100:
                gap = float("nan")
            else:
                gap = abs(incumbent - bound) / max(1.0, abs(incumbent))
            progress.append(
                {
                    "time_seconds": runtime,
                    "nodes": node_count,
                    "incumbent": incumbent,
                    "best_bound": bound,
                    "gap": gap,
                }
            )

    model.optimize(callback)

    progress_df = _deduplicate_progress(pd.DataFrame(progress))
    progress_df.to_csv(progress_output, index=False)
    _plot_progress(progress_df, plot_output)

    chosen: dict[str, Pattern] = {}
    for subject, subject_patterns in patterns.items():
        chosen_pattern = max(subject_patterns, key=lambda pattern: lam[(subject, pattern.index)].X)
        chosen[subject] = chosen_pattern

    placement: dict[str, tuple[pd.Timestamp, str]] = {}
    for pattern in chosen.values():
        placement.update(pattern.assignment)
    timetable = timetable_from_placement(placement, days)
    same_slot_clashes = _same_slot_clashes(placement, pairs)

    objective_value = _safe_model_attr(model, "ObjVal") if model.SolCount else None
    best_bound = _safe_model_attr(model, "ObjBound")
    if best_bound is None and objective_value is not None:
        best_bound = objective_value
    gap_value = _safe_model_attr(model, "MIPGap") if model.SolCount else None
    if gap_value is None and objective_value is not None and best_bound is not None:
        gap_value = abs(objective_value - best_bound) / max(1.0, abs(objective_value))

    return {
        "status": int(model.Status),
        "objective": objective_value,
        "best_bound": best_bound,
        "gap": gap_value,
        "nodes": float(model.NodeCount),
        "iterations": float(model.IterCount),
        "work": float(model.Work),
        "runtime_seconds": float(model.Runtime),
        "pattern_count": sum(len(subject_patterns) for subject_patterns in patterns.values()),
        "mu_count": len(mu),
        "theta_count": joint_stats["theta_count"],
        "joint_community_count": len(joint_stats["communities"]),
        "joint_communities": ";".join("|".join(community) for community in joint_stats["communities"]),
        "joint_cut_count": len(joint_cut_stats["cuts"]),
        "joint_cut_communities": ";".join("|".join(cut["community"]) for cut in joint_cut_stats["cuts"]),
        "joint_cut_max_violation": joint_cut_stats["max_violation"],
        "clique_cut_count": len(clique_cut_stats["cuts"]),
        "clique_cut_max_violation": clique_cut_stats["max_violation"],
        "incompatible_mu_count": incompatible_count,
        "same_slot_clashes": same_slot_clashes,
        "timetable": timetable,
    }


def _clean_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    result = pairs.copy()
    if result.columns[0].startswith("Unnamed") or result.columns[0] not in result.columns[1:]:
        result = result.set_index(result.columns[0])
    result.index = result.index.astype(str).str.strip()
    result.columns = result.columns.astype(str).str.strip()
    return result.apply(pd.to_numeric, errors="raise")


def _build_blocks(exams: pd.DataFrame) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    for subject, subject_df in exams.groupby("Subject", sort=False):
        subject_df = subject_df.sort_values("Exam")
        blocks[str(subject)] = subject_df["Full Name"].astype(str).tolist()
    return blocks


def _add_joint_community_constraints(
    *,
    model: Any,
    gp: Any,
    patterns: dict[str, list[Pattern]],
    blocks: dict[str, list[str]],
    pairs: pd.DataFrame,
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    subjects: list[str],
    community_count: int,
    community_size: int,
    max_tuples: int,
) -> dict[str, Any]:
    if community_count <= 0:
        return {"communities": [], "theta_count": 0}

    subject_order = {subject: pos for pos, subject in enumerate(subjects)}
    communities = _select_dense_subject_communities(
        blocks=blocks,
        pairs=pairs,
        patterns=patterns,
        count=community_count,
        size=community_size,
        max_tuples=max_tuples,
    )

    theta_count = 0
    for community_index, community in enumerate(communities, start=1):
        by_subject_pattern: dict[tuple[str, int], list[Any]] = {}
        by_pair_pattern: dict[tuple[str, int, str, int], list[Any]] = {}
        subject_patterns = [patterns[subject] for subject in community]

        for combo in itertools.product(*subject_patterns):
            signature = tuple(pattern.index for pattern in combo)
            if _combo_has_incompatible_pair(combo, community, cross_costs, subject_order):
                continue

            theta = model.addVar(
                vtype=gp.GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=_var_name("theta", community_index, *signature),
            )
            theta_count += 1

            for pattern in combo:
                by_subject_pattern.setdefault((pattern.subject, pattern.index), []).append(theta)

            for left_pos, left_pattern in enumerate(combo):
                for right_pattern in combo[left_pos + 1 :]:
                    key = _ordered_pattern_pair_key(
                        left_pattern.subject,
                        left_pattern.index,
                        right_pattern.subject,
                        right_pattern.index,
                        subject_order,
                    )
                    by_pair_pattern.setdefault(key, []).append(theta)

        model.update()

        for subject in community:
            for pattern in patterns[subject]:
                model.addConstr(
                    gp.quicksum(by_subject_pattern.get((subject, pattern.index), []))
                    == lam[(subject, pattern.index)],
                    name=_var_name("theta_lambda", community_index, subject, pattern.index),
                )

        for left_pos, left_subject in enumerate(community):
            for right_subject in community[left_pos + 1 :]:
                for left_pattern in patterns[left_subject]:
                    for right_pattern in patterns[right_subject]:
                        key = _ordered_pattern_pair_key(
                            left_subject,
                            left_pattern.index,
                            right_subject,
                            right_pattern.index,
                            subject_order,
                        )
                        model.addConstr(
                            gp.quicksum(by_pair_pattern.get(key, [])) == mu[key],
                            name=_var_name(
                                "theta_mu",
                                community_index,
                                left_subject,
                                left_pattern.index,
                                right_subject,
                                right_pattern.index,
                            ),
                        )

    return {"communities": communities, "theta_count": theta_count}


def _generate_joint_consistency_cuts(
    *,
    model: Any,
    gp: Any,
    patterns: dict[str, list[Pattern]],
    blocks: dict[str, list[str]],
    pairs: pd.DataFrame,
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    subjects: list[str],
    relax_lambda: bool,
    mu_binary: bool,
    community_count: int,
    community_size: int,
    max_tuples: int,
    rounds: int,
    max_cuts_per_round: int,
    tolerance: float,
) -> dict[str, Any]:
    if community_count <= 0 or rounds <= 0:
        return {"cuts": [], "max_violation": 0.0}

    subject_order = {subject: pos for pos, subject in enumerate(subjects)}
    communities = _select_dense_subject_communities(
        blocks=blocks,
        pairs=pairs,
        patterns=patterns,
        count=community_count,
        size=community_size,
        max_tuples=max_tuples,
    )
    if not communities:
        return {"cuts": [], "max_violation": 0.0}

    original_lambda_types = {key: var.VType for key, var in lam.items()}
    original_mu_types = {key: var.VType for key, var in mu.items()}
    if not relax_lambda:
        for var in lam.values():
            var.VType = gp.GRB.CONTINUOUS
    if mu_binary:
        for var in mu.values():
            var.VType = gp.GRB.CONTINUOUS
    model.update()

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
            for community in communities:
                cut = _separate_joint_consistency_cut(
                    gp=gp,
                    patterns=patterns,
                    cross_costs=cross_costs,
                    lam=lam,
                    mu=mu,
                    community=community,
                    subject_order=subject_order,
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
        for key, var_type in original_lambda_types.items():
            lam[key].VType = var_type
        for key, var_type in original_mu_types.items():
            mu[key].VType = var_type
        model.update()
        model.reset()

    return {"cuts": generated, "max_violation": max_violation}


def _generate_incompatibility_clique_cuts(
    *,
    model: Any,
    gp: Any,
    patterns: dict[str, list[Pattern]],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    lam: dict[tuple[str, int], Any],
    subjects: list[str],
    relax_lambda: bool,
    rounds: int,
    max_cuts_per_round: int,
    tolerance: float,
) -> dict[str, Any]:
    if rounds <= 0 or max_cuts_per_round <= 0:
        return {"cuts": [], "max_violation": 0.0}

    subject_order = {subject: pos for pos, subject in enumerate(subjects)}
    adjacency = _build_incompatibility_graph(
        patterns=patterns,
        cross_costs=cross_costs,
        subject_order=subject_order,
        max_same_slot_clashes=15.0,
    )
    if not adjacency:
        return {"cuts": [], "max_violation": 0.0}

    original_lambda_types = {key: var.VType for key, var in lam.items()}
    if not relax_lambda:
        for var in lam.values():
            var.VType = gp.GRB.CONTINUOUS
    model.update()

    generated: list[dict[str, Any]] = []
    generated_keys: set[frozenset[tuple[str, int]]] = set()
    max_violation = 0.0
    original_output_flag = int(model.Params.OutputFlag)
    model.setParam("OutputFlag", 0)

    try:
        for round_index in range(1, rounds + 1):
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                break

            weights = {node: float(lam[node].X) for node in adjacency}
            added_this_round = 0
            candidate_cliques = _exact_violated_cliques(
                gp=gp,
                adjacency=adjacency,
                weights=weights,
                max_cliques=max_cuts_per_round,
                tolerance=tolerance,
            )
            for clique in candidate_cliques:
                key = frozenset(clique)
                if key in generated_keys:
                    continue
                generated_keys.add(key)
                violation = sum(weights[node] for node in clique) - 1.0
                model.addConstr(
                    gp.quicksum(lam[node] for node in clique) <= 1.0,
                    name=_var_name("lifted_clique", round_index, len(generated) + 1),
                )
                generated.append({"clique": clique, "violation": violation})
                max_violation = max(max_violation, violation)
                added_this_round += 1
                if added_this_round >= max_cuts_per_round:
                    break

            model.update()
            if added_this_round == 0:
                break
    finally:
        model.setParam("OutputFlag", original_output_flag)
        for key, var_type in original_lambda_types.items():
            lam[key].VType = var_type
        model.update()
        model.reset()

    return {"cuts": generated, "max_violation": max_violation}


def _build_incompatibility_graph(
    *,
    patterns: dict[str, list[Pattern]],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    subject_order: dict[str, int],
    max_same_slot_clashes: float,
) -> dict[tuple[str, int], set[tuple[str, int]]]:
    nodes = [(subject, pattern.index) for subject, subject_patterns in patterns.items() for pattern in subject_patterns]
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {node: set() for node in nodes}
    subjects = list(patterns)

    for left_pos, left_subject in enumerate(subjects):
        for right_subject in subjects[left_pos + 1 :]:
            for left_pattern in patterns[left_subject]:
                for right_pattern in patterns[right_subject]:
                    key = _ordered_pattern_pair_key(
                        left_subject,
                        left_pattern.index,
                        right_subject,
                        right_pattern.index,
                        subject_order,
                    )
                    cost = cross_costs[key]
                    pair_is_impossible = (
                        cost.daily_length_infeasible
                        or cost.same_slot_clashes > max_same_slot_clashes
                    )
                    if not pair_is_impossible:
                        continue
                    left_node = (left_subject, left_pattern.index)
                    right_node = (right_subject, right_pattern.index)
                    adjacency[left_node].add(right_node)
                    adjacency[right_node].add(left_node)

    return {node: neighbors for node, neighbors in adjacency.items() if neighbors}


def _exact_violated_cliques(
    *,
    gp: Any,
    adjacency: dict[tuple[str, int], set[tuple[str, int]]],
    weights: dict[tuple[str, int], float],
    max_cliques: int,
    tolerance: float,
) -> list[tuple[tuple[str, int], ...]]:
    nodes = [node for node in adjacency if weights.get(node, 0.0) > tolerance]
    if len(nodes) < 2:
        return []

    cliques: list[tuple[tuple[str, int], ...]] = []
    forbidden: list[frozenset[tuple[str, int]]] = []
    for _ in range(max_cliques):
        sep = gp.Model("pattern_clique_separator")
        sep.setParam("OutputFlag", 0)
        z = sep.addVars(len(nodes), vtype=gp.GRB.BINARY, name="z")
        sep.setObjective(
            gp.quicksum(weights[nodes[index]] * z[index] for index in range(len(nodes))),
            gp.GRB.MAXIMIZE,
        )

        for i, left in enumerate(nodes):
            left_neighbors = adjacency.get(left, set())
            for j in range(i + 1, len(nodes)):
                right = nodes[j]
                if right not in left_neighbors:
                    sep.addConstr(z[i] + z[j] <= 1)

        for clique in forbidden:
            sep.addConstr(
                gp.quicksum(z[index] for index, node in enumerate(nodes) if node in clique) <= len(clique) - 1
            )

        sep.optimize()
        if sep.Status != gp.GRB.OPTIMAL or sep.ObjVal <= 1.0 + tolerance:
            break

        clique = tuple(sorted(nodes[index] for index in range(len(nodes)) if z[index].X > 0.5))
        if len(clique) < 2:
            break
        cliques.append(clique)
        forbidden.append(frozenset(clique))

    return cliques


def _greedy_violated_cliques(
    *,
    adjacency: dict[tuple[str, int], set[tuple[str, int]]],
    weights: dict[tuple[str, int], float],
    max_cliques: int,
    tolerance: float,
) -> list[tuple[tuple[str, int], ...]]:
    seeds = sorted(adjacency, key=lambda node: (-weights.get(node, 0.0), node))
    cliques: list[tuple[tuple[str, int], ...]] = []
    seen: set[frozenset[tuple[str, int]]] = set()

    for seed in seeds:
        if weights.get(seed, 0.0) <= tolerance:
            break
        clique = [seed]
        candidates = set(adjacency[seed])
        while candidates:
            next_node = max(candidates, key=lambda node: (weights.get(node, 0.0), node))
            if weights.get(next_node, 0.0) <= tolerance:
                break
            clique.append(next_node)
            candidates &= adjacency[next_node]

        clique_key = frozenset(clique)
        if clique_key in seen:
            continue
        seen.add(clique_key)
        if len(clique) >= 2 and sum(weights.get(node, 0.0) for node in clique) > 1.0 + tolerance:
            cliques.append(tuple(sorted(clique)))
            if len(cliques) >= max_cliques:
                break
    return cliques


def _separate_joint_consistency_cut(
    *,
    gp: Any,
    patterns: dict[str, list[Pattern]],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    lam: dict[tuple[str, int], Any],
    mu: dict[tuple[str, int, str, int], Any],
    community: tuple[str, ...],
    subject_order: dict[str, int],
    tolerance: float,
) -> dict[str, Any] | None:
    keys: list[tuple[Any, ...]] = []
    value_by_key: dict[tuple[Any, ...], float] = {}

    for subject in community:
        for pattern in patterns[subject]:
            key = ("lambda", subject, pattern.index)
            keys.append(key)
            value_by_key[key] = float(lam[(subject, pattern.index)].X)

    for left_pos, left_subject in enumerate(community):
        for right_subject in community[left_pos + 1 :]:
            for left_pattern in patterns[left_subject]:
                for right_pattern in patterns[right_subject]:
                    ordered_key = _ordered_pattern_pair_key(
                        left_subject,
                        left_pattern.index,
                        right_subject,
                        right_pattern.index,
                        subject_order,
                    )
                    key = ("mu", *ordered_key)
                    keys.append(key)
                    value_by_key[key] = float(mu[ordered_key].X)

    key_index = {key: index for index, key in enumerate(keys)}
    sep = gp.Model("joint_consistency_separator")
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

    objective = gp.quicksum(
        (y_plus[index] - y_minus[index]) * value_by_key[key]
        for key, index in key_index.items()
    ) - delta
    sep.setObjective(objective, gp.GRB.MAXIMIZE)

    subject_patterns = [patterns[subject] for subject in community]
    for combo_index, combo in enumerate(itertools.product(*subject_patterns)):
        if _combo_has_incompatible_pair(combo, community, cross_costs, subject_order):
            continue

        active_indices: list[int] = []
        for pattern in combo:
            active_indices.append(key_index[("lambda", pattern.subject, pattern.index)])

        for left_pos, left_pattern in enumerate(combo):
            for right_pattern in combo[left_pos + 1 :]:
                ordered_key = _ordered_pattern_pair_key(
                    left_pattern.subject,
                    left_pattern.index,
                    right_pattern.subject,
                    right_pattern.index,
                    subject_order,
                )
                active_indices.append(key_index[("mu", *ordered_key)])

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
        "community": community,
        "coefficients": coefficients,
        "rhs": rhs,
        "violation": violation,
    }


def _select_dense_subject_communities(
    *,
    blocks: dict[str, list[str]],
    pairs: pd.DataFrame,
    patterns: dict[str, list[Pattern]],
    count: int,
    size: int,
    max_tuples: int,
) -> list[tuple[str, ...]]:
    if size <= 1:
        return []

    candidates: list[tuple[float, int, tuple[str, ...]]] = []
    subjects = list(blocks)
    for community in itertools.combinations(subjects, size):
        tuple_count = 1
        for subject in community:
            tuple_count *= len(patterns[subject])
        if tuple_count > max_tuples:
            continue
        score = _community_pair_mass(community, blocks, pairs)
        if score > 0:
            candidates.append((score, tuple_count, community))

    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    selected: list[tuple[str, ...]] = []
    used: set[str] = set()
    for _score, _tuple_count, community in candidates:
        if any(subject in used for subject in community):
            continue
        selected.append(community)
        used.update(community)
        if len(selected) >= count:
            break
    return selected


def _community_pair_mass(community: tuple[str, ...], blocks: dict[str, list[str]], pairs: pd.DataFrame) -> float:
    total = 0.0
    for left_pos, left_subject in enumerate(community):
        for right_subject in community[left_pos + 1 :]:
            for exam_i in blocks[left_subject]:
                for exam_j in blocks[right_subject]:
                    total += float(pairs.loc[exam_i, exam_j])
    return total


def _combo_has_incompatible_pair(
    combo: tuple[Pattern, ...],
    community: tuple[str, ...],
    cross_costs: dict[tuple[str, int, str, int], CrossCost],
    subject_order: dict[str, int],
) -> bool:
    for left_pos, left_pattern in enumerate(combo):
        for right_pattern in combo[left_pos + 1 :]:
            key = _ordered_pattern_pair_key(
                left_pattern.subject,
                left_pattern.index,
                right_pattern.subject,
                right_pattern.index,
                subject_order,
            )
            if cross_costs[key].daily_length_infeasible:
                return True
    return False


def _ordered_pattern_pair_key(
    left_subject: str,
    left_pattern: int,
    right_subject: str,
    right_pattern: int,
    subject_order: dict[str, int],
) -> tuple[str, int, str, int]:
    if subject_order[left_subject] < subject_order[right_subject]:
        return (left_subject, left_pattern, right_subject, right_pattern)
    return (right_subject, right_pattern, left_subject, left_pattern)


def _generate_all_patterns(
    *,
    blocks: dict[str, list[str]],
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    dates: list[pd.Timestamp],
    date_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
) -> dict[str, list[Pattern]]:
    by_exam_subject = dict(zip(exams["Full Name"].astype(str), exams["Subject"].astype(str)))
    usable_dates = set(_usable_dates(days))
    second_half_dates = set(dates[round(len(dates) / 2) :])
    first_half_subjects = {"Finance", "Law and Ethics"}
    first_two_dates = set(dates[:2])
    result: dict[str, list[Pattern]] = {}

    for subject, exam_names in blocks.items():
        if len(exam_names) != 2:
            raise ValueError(f"Toy pattern model expects two exams per subject; {subject!r} has {len(exam_names)}.")
        subject_patterns: list[Pattern] = []
        for start_idx in range(len(dates) - 1):
            left_date = dates[start_idx]
            right_date = dates[start_idx + 1]
            if left_date not in usable_dates or right_date not in usable_dates:
                continue
            for first_exam, second_exam in itertools.permutations(exam_names, 2):
                for first_slot in _allowed_slots(first_exam, subject, left_date, exam_lengths):
                    for second_slot in _allowed_slots(second_exam, subject, right_date, exam_lengths):
                        assignment = {
                            first_exam: (left_date, first_slot),
                            second_exam: (right_date, second_slot),
                        }
                        if not _assignment_passes_toy_global_rules(
                            assignment=assignment,
                            subject=subject,
                            dates=dates,
                            first_two_dates=first_two_dates,
                            second_half_dates=second_half_dates,
                            first_half_subjects=first_half_subjects,
                        ):
                            continue
                        internal_objective, internal_clashes = _assignment_cost(
                            assignment,
                            pairs=pairs,
                            date_index=date_index,
                        )
                        subject_patterns.append(
                            Pattern(
                                subject=subject,
                                index=len(subject_patterns),
                                assignment=assignment,
                                internal_objective=internal_objective,
                                internal_same_slot_clashes=internal_clashes,
                            )
                        )
        if not subject_patterns:
            raise ValueError(f"No feasible subject patterns generated for {subject!r} ({exam_names}).")
        result[subject] = subject_patterns
    return result


def _usable_dates(days: pd.DataFrame) -> list[pd.Timestamp]:
    dates = parse_day_series(days["Date"])
    is_weekend = dates.dt.dayofweek >= 5
    is_may_first = dates.dt.month.eq(5) & dates.dt.day.eq(1)
    return dates.loc[~(is_weekend | is_may_first)].tolist()


def _allowed_slots(exam: str, subject: str, date: pd.Timestamp, exam_lengths: dict[str, float]) -> list[str]:
    slots = list(DEFAULT_SLOTS)
    if float(exam_lengths[exam]) > 180:
        slots = [slot for slot in slots if slot != "PM"]
    if subject.upper() == "LANGUAGE A LITERATURE" and pd.Timestamp(date).day_name() == "Friday":
        return []
    return slots


def _assignment_passes_toy_global_rules(
    *,
    assignment: dict[str, tuple[pd.Timestamp, str]],
    subject: str,
    dates: list[pd.Timestamp],
    first_two_dates: set[pd.Timestamp],
    second_half_dates: set[pd.Timestamp],
    first_half_subjects: set[str],
) -> bool:
    if subject.upper() == "SBS":
        if assignment.get("SBS Exam 1") != (dates[0], "AM"):
            return False
    else:
        if any(date in first_two_dates for date, _slot in assignment.values()):
            return False

    if subject in first_half_subjects:
        if any(date in second_half_dates for date, _slot in assignment.values()):
            return False
    return True


def _assignment_cost(
    assignment: dict[str, tuple[pd.Timestamp, str]],
    *,
    pairs: pd.DataFrame,
    date_index: dict[pd.Timestamp, int],
) -> tuple[float, float]:
    objective = 0.0
    clashes = 0.0
    exams = sorted(assignment)
    for pos, exam_i in enumerate(exams):
        for exam_j in exams[pos + 1 :]:
            contribution, same_slot = _pair_cost_and_clash(
                exam_i,
                assignment[exam_i],
                exam_j,
                assignment[exam_j],
                pairs=pairs,
                date_index=date_index,
            )
            objective += contribution
            clashes += same_slot
    return objective, clashes


def _cross_cost(
    left_assignment: dict[str, tuple[pd.Timestamp, str]],
    right_assignment: dict[str, tuple[pd.Timestamp, str]],
    *,
    pairs: pd.DataFrame,
    date_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
) -> CrossCost:
    objective = 0.0
    same_slot_clashes = 0.0
    daily_length_infeasible = False
    for exam_i, placement_i in left_assignment.items():
        for exam_j, placement_j in right_assignment.items():
            if placement_i[0] == placement_j[0] and exam_lengths[exam_i] + exam_lengths[exam_j] > 375:
                daily_length_infeasible = True
            contribution, clash = _pair_cost_and_clash(
                exam_i,
                placement_i,
                exam_j,
                placement_j,
                pairs=pairs,
                date_index=date_index,
            )
            objective += contribution
            same_slot_clashes += clash
    return CrossCost(objective=objective, same_slot_clashes=same_slot_clashes, daily_length_infeasible=daily_length_infeasible)


def _pair_cost_and_clash(
    exam_i: str,
    placement_i: tuple[pd.Timestamp, str],
    exam_j: str,
    placement_j: tuple[pd.Timestamp, str],
    *,
    pairs: pd.DataFrame,
    date_index: dict[pd.Timestamp, int],
) -> tuple[float, float]:
    date_i, slot_i = placement_i
    date_j, slot_j = placement_j
    cij = float(pairs.loc[exam_i, exam_j])
    gap = abs(date_index[pd.Timestamp(date_i).normalize()] - date_index[pd.Timestamp(date_j).normalize()])

    objective = 0.0
    clash = 0.0
    if gap == 0 and slot_i == slot_j:
        objective += cij * WEIGHTS["a"]
        clash += cij
    if gap == 0:
        objective += cij * WEIGHTS["b"]
    elif gap == 1 or gap == 5:
        objective += cij * WEIGHTS["c"]
    elif gap == 2:
        objective += cij * WEIGHTS["d"]
    elif gap == 3:
        objective += cij * WEIGHTS["e"]
    elif gap == 4:
        objective += cij * WEIGHTS["f"]
    return objective, clash


def _same_slot_clashes(placement: dict[str, tuple[pd.Timestamp, str]], pairs: pd.DataFrame) -> float:
    total = 0.0
    exams = sorted(placement)
    for pos, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[pos + 1 :]:
            date_j, slot_j = placement[exam_j]
            if date_i == date_j and slot_i == slot_j:
                total += float(pairs.loc[exam_i, exam_j])
    return total


def _deduplicate_progress(progress: pd.DataFrame) -> pd.DataFrame:
    if progress.empty:
        return progress
    progress = progress.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["incumbent", "best_bound"])
    progress["time_seconds"] = progress["time_seconds"].round(3)
    progress["nodes"] = progress["nodes"].round(0)
    return progress.drop_duplicates(subset=["time_seconds", "nodes", "incumbent", "best_bound"]).reset_index(drop=True)


def _plot_progress(progress: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if not progress.empty:
        ax.plot(progress["time_seconds"], progress["incumbent"], label="Incumbent", color="#d62728", linewidth=3.0)
        ax.plot(progress["time_seconds"], progress["best_bound"], label="Best bound", color="#1f77b4", linewidth=3.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_clock_time))
    ax.set_xlabel("Time (hh:mm:ss)")
    ax.set_ylabel("Objective")
    ax.set_title("Toy subject-pattern MILP incumbent and bound evolution")
    ax.grid(True, alpha=0.25)
    if not progress.empty:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _safe_model_attr(model: Any, attr_name: str) -> float | None:
    try:
        return float(getattr(model, attr_name))
    except Exception:
        return None


def _format_clock_time(seconds: float, _pos: int | None = None) -> str:
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def _var_name(prefix: str, *parts: object) -> str:
    clean = [str(part).replace(" ", "_").replace("[", "(").replace("]", ")").replace(",", "_") for part in parts]
    return prefix + "[" + ",".join(clean) + "]"


if __name__ == "__main__":
    main()
