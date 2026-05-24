from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_toy_pattern_model import (
    Pattern,
    _build_blocks,
    _clean_pairs,
    _cross_cost,
    _generate_all_patterns,
    _select_dense_subject_communities,
)
from src.anthony_model import parse_day_series, prepare_toy_inputs


@dataclass
class SubjectLayer:
    subject: str
    objective: np.ndarray
    clashes: np.ndarray


@dataclass
class CopyBlock:
    name: str
    subjects: tuple[str, ...]
    tuples: np.ndarray
    objective: np.ndarray
    clashes: np.ndarray
    multipliers: list[np.ndarray]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toy Lagrangian decomposition with exact dense-community blocks."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--community-count", type=int, default=1)
    parser.add_argument("--community-size", type=int, default=3)
    parser.add_argument("--community-max-tuples", type=int, default=250_000)
    parser.add_argument("--dual-mode", choices=["bundle", "full"], default="bundle")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--upper-bound", type=float, default=25190.0)
    parser.add_argument("--multiplier-bound", type=float, default=5000.0)
    parser.add_argument("--eta-bound", type=float, default=500.0)
    parser.add_argument("--proximal-weight", type=float, default=1e-3)
    parser.add_argument("--duality-gap-tolerance", type=float, default=1e-5)
    parser.add_argument("--summary-output", type=Path, default=Path("toy_lagrangian_community_poc_summary.csv"))
    parser.add_argument("--history-output", type=Path, default=Path("toy_lagrangian_community_poc_history.csv"))
    parser.add_argument("--plot-output", type=Path, default=Path("toy_lagrangian_community_poc_bounds.png"))
    args = parser.parse_args()

    raw_exams = pd.read_csv(args.data_dir / "Toy exam list.csv")
    raw_pairs = pd.read_csv(args.data_dir / "Exam pairs.csv")
    exams, days, pairs = prepare_toy_inputs(raw_exams, raw_pairs)
    pairs = _clean_pairs(pairs)
    days = days.copy()
    days["Date"] = parse_day_series(days["Date"])

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

    started = time.perf_counter()
    subject_layers = _build_subject_layers(patterns)
    copy_blocks, communities = _build_copy_blocks(
        patterns=patterns,
        blocks=blocks,
        pairs=pairs,
        date_index=date_index,
        exam_lengths=exam_lengths,
        community_count=args.community_count,
        community_size=args.community_size,
        community_max_tuples=args.community_max_tuples,
    )
    if args.dual_mode == "full":
        history, summary = solve_full_lagrangian_dual(
            subject_layers=subject_layers,
            copy_blocks=copy_blocks,
        )
    else:
        history, summary = cutting_plane_lagrangian_dual(
            subject_layers=subject_layers,
            copy_blocks=copy_blocks,
            iterations=args.iterations,
            multiplier_bound=args.multiplier_bound,
            eta_bound=args.eta_bound,
            proximal_weight=args.proximal_weight,
            tolerance=args.duality_gap_tolerance,
        )
    elapsed = time.perf_counter() - started

    history.to_csv(args.history_output, index=False)
    _plot_history(history, args.plot_output)

    summary.update(
        {
            "elapsed_seconds": elapsed,
            "subject_count": len(subject_layers),
            "pattern_count": sum(len(layer.objective) for layer in subject_layers.values()),
            "copy_block_count": len(copy_blocks),
            "community_count": len(communities),
            "communities": ";".join("|".join(community) for community in communities),
            "community_size": args.community_size,
            "community_max_tuples": args.community_max_tuples,
            "copy_tuple_count": sum(len(block.objective) for block in copy_blocks),
            "upper_bound": args.upper_bound,
            "pattern_lp_reference": 15370.0,
            "toy_optimum_reference": 25190.0,
            "dual_mode": args.dual_mode,
            "iterations_requested": args.iterations,
            "multiplier_bound": args.multiplier_bound,
            "eta_bound": args.eta_bound,
            "proximal_weight": args.proximal_weight,
        }
    )
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)

    print("Community Lagrangian POC seconds:", round(elapsed, 6))
    print("Communities:", summary["communities"])
    print("Copy blocks:", summary["copy_block_count"])
    print("Copy tuples:", summary["copy_tuple_count"])
    print("Best Lagrangian lower bound:", summary["best_bound"])
    print("Best iteration:", summary["best_iteration"])
    print("Final dual upper approximation:", summary.get("final_dual_upper_bound"))
    print("Final cutting-plane gap:", summary.get("final_cp_gap"))
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved history to {args.history_output}")
    print(f"Saved plot to {args.plot_output}")


def _build_subject_layers(patterns: dict[str, list[Pattern]]) -> dict[str, SubjectLayer]:
    return {
        subject: SubjectLayer(
            subject=subject,
            objective=np.array([pattern.internal_objective for pattern in subject_patterns], dtype=float),
            clashes=np.array([pattern.internal_same_slot_clashes for pattern in subject_patterns], dtype=float),
        )
        for subject, subject_patterns in patterns.items()
    }


def _build_copy_blocks(
    *,
    patterns: dict[str, list[Pattern]],
    blocks: dict[str, list[str]],
    pairs: pd.DataFrame,
    date_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
    community_count: int,
    community_size: int,
    community_max_tuples: int,
) -> tuple[list[CopyBlock], list[tuple[str, ...]]]:
    subjects = list(patterns)
    communities = _select_dense_subject_communities(
        blocks=blocks,
        pairs=pairs,
        patterns=patterns,
        count=community_count,
        size=community_size,
        max_tuples=community_max_tuples,
    )
    owned_pairs: dict[tuple[str, str], tuple[str, ...]] = {}
    for community in communities:
        for left_pos, left in enumerate(community):
            for right in community[left_pos + 1 :]:
                owned_pairs.setdefault(_pair_key(left, right), community)

    copy_blocks: list[CopyBlock] = []
    for community in communities:
        owner_edges = [
            _pair_key(left, right)
            for left_pos, left in enumerate(community)
            for right in community[left_pos + 1 :]
            if owned_pairs.get(_pair_key(left, right)) == community
        ]
        if owner_edges:
            copy_blocks.append(
                _make_joint_block(
                    name="community:" + "|".join(community),
                    subjects=community,
                    owner_edges=set(owner_edges),
                    patterns=patterns,
                    pairs=pairs,
                    date_index=date_index,
                    exam_lengths=exam_lengths,
                )
            )

    for left_pos, left in enumerate(subjects):
        for right in subjects[left_pos + 1 :]:
            key = _pair_key(left, right)
            if key in owned_pairs:
                continue
            copy_blocks.append(
                _make_joint_block(
                    name=f"edge:{left}|{right}",
                    subjects=(left, right),
                    owner_edges={key},
                    patterns=patterns,
                    pairs=pairs,
                    date_index=date_index,
                    exam_lengths=exam_lengths,
                )
            )
    return copy_blocks, communities


def _make_joint_block(
    *,
    name: str,
    subjects: tuple[str, ...],
    owner_edges: set[tuple[str, str]],
    patterns: dict[str, list[Pattern]],
    pairs: pd.DataFrame,
    date_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
) -> CopyBlock:
    tuple_rows: list[tuple[int, ...]] = []
    objective: list[float] = []
    clashes: list[float] = []
    subject_patterns = [patterns[subject] for subject in subjects]

    for combo in itertools.product(*subject_patterns):
        compatible = True
        combo_objective = 0.0
        combo_clashes = 0.0
        for left_pos, left_pattern in enumerate(combo):
            for right_pos in range(left_pos + 1, len(combo)):
                right_pattern = combo[right_pos]
                cost = _cross_cost(
                    left_pattern.assignment,
                    right_pattern.assignment,
                    pairs=pairs,
                    date_index=date_index,
                    exam_lengths=exam_lengths,
                )
                if cost.daily_length_infeasible:
                    compatible = False
                    break
                if _pair_key(left_pattern.subject, right_pattern.subject) in owner_edges:
                    combo_objective += cost.objective
                    combo_clashes += cost.same_slot_clashes
            if not compatible:
                break
        if not compatible:
            continue
        tuple_rows.append(tuple(pattern.index for pattern in combo))
        objective.append(combo_objective)
        clashes.append(combo_clashes)

    if not tuple_rows:
        raise ValueError(f"Copy block {name!r} has no feasible tuples.")

    return CopyBlock(
        name=name,
        subjects=subjects,
        tuples=np.array(tuple_rows, dtype=np.int32),
        objective=np.array(objective, dtype=float),
        clashes=np.array(clashes, dtype=float),
        multipliers=[np.zeros(len(patterns[subject]), dtype=float) for subject in subjects],
    )


def cutting_plane_lagrangian_dual(
    *,
    subject_layers: dict[str, SubjectLayer],
    copy_blocks: list[CopyBlock],
    iterations: int,
    multiplier_bound: float,
    eta_bound: float,
    proximal_weight: float,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise ImportError("run_toy_lagrangian_community_poc.py requires gurobipy.") from exc

    subject_incidence: dict[str, list[tuple[CopyBlock, int]]] = {subject: [] for subject in subject_layers}
    for block in copy_blocks:
        for pos, subject in enumerate(block.subjects):
            subject_incidence[subject].append((block, pos))

    master = gp.Model("toy_community_lagrangian_bundle")
    master.setParam("OutputFlag", 0)
    u_vars: dict[tuple[str, str, int], Any] = {}
    for block in copy_blocks:
        for pos, subject in enumerate(block.subjects):
            for pattern_index in range(len(subject_layers[subject].objective)):
                u_vars[(block.name, subject, pattern_index)] = master.addVar(
                    lb=-multiplier_bound,
                    ub=multiplier_bound,
                    name=_var_name("u", block.name, subject, pattern_index),
                )
    eta_var = master.addVar(lb=0.0, ub=eta_bound, name="eta")
    theta = master.addVar(lb=-1e9, ub=1e9, name="theta")
    master.setObjective(theta, gp.GRB.MAXIMIZE)
    master.update()

    current_u = {key: 0.0 for key in u_vars}
    current_eta = 0.0
    center_u = dict(current_u)
    center_eta = current_eta
    best_bound = -float("inf")
    best_iteration = 0
    best_u = dict(current_u)
    best_eta = current_eta
    history: list[dict[str, float]] = []
    final_upper = None

    for iteration in range(1, iterations + 1):
        value, gradient, state = _evaluate_split_dual(
            subject_layers=subject_layers,
            copy_blocks=copy_blocks,
            subject_incidence=subject_incidence,
            eta=current_eta,
        )
        if value > best_bound:
            best_bound = value
            best_iteration = iteration
            best_u = dict(current_u)
            best_eta = current_eta
            center_u = dict(current_u)
            center_eta = current_eta

        rhs = value
        expr = theta
        for key, grad_value in gradient.items():
            if key == ("eta",):
                expr -= grad_value * eta_var
                rhs -= grad_value * current_eta
            else:
                expr -= grad_value * u_vars[key]
                rhs -= grad_value * current_u[key]
        master.addConstr(expr <= rhs, name=_var_name("bundle_cut", iteration))
        _set_proximal_objective(
            master=master,
            gp=gp,
            theta=theta,
            u_vars=u_vars,
            eta_var=eta_var,
            center_u=center_u,
            center_eta=center_eta,
            proximal_weight=proximal_weight,
        )
        master.optimize()
        if master.Status != gp.GRB.OPTIMAL:
            break

        final_upper = float(theta.X)
        current_eta = float(eta_var.X)
        for key, var in u_vars.items():
            current_u[key] = float(var.X)
        _load_multipliers(copy_blocks, current_u)

        cp_gap = final_upper - best_bound
        history.append(
            {
                "iteration": float(iteration),
                "bound": float(value),
                "best_bound": float(best_bound),
                "dual_upper_bound": float(final_upper),
                "cp_gap": float(cp_gap),
                "eta": float(current_eta),
                "agreement_l1": float(state["agreement_l1"]),
                "clash_total": float(state["clash_total"]),
                "clash_residual": float(state["clash_residual"]),
            }
        )
        if cp_gap <= tolerance:
            break

    final = history[-1] if history else {}
    return pd.DataFrame(history), {
        "best_bound": best_bound,
        "best_iteration": best_iteration,
        "final_bound": final.get("bound"),
        "final_best_bound": final.get("best_bound"),
        "final_dual_upper_bound": final_upper,
        "final_cp_gap": final.get("cp_gap"),
        "final_eta": final.get("eta"),
        "final_agreement_l1": final.get("agreement_l1"),
        "final_clash_total": final.get("clash_total"),
        "final_clash_residual": final.get("clash_residual"),
        "iterations_completed": len(history),
        "best_eta": best_eta,
        "best_multiplier_linf": max((abs(value) for value in best_u.values()), default=0.0),
    }


def solve_full_lagrangian_dual(
    *,
    subject_layers: dict[str, SubjectLayer],
    copy_blocks: list[CopyBlock],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise ImportError("run_toy_lagrangian_community_poc.py requires gurobipy.") from exc

    model = gp.Model("toy_full_community_lagrangian_dual")
    model.setParam("OutputFlag", 0)
    eta = model.addVar(lb=0.0, name="eta")

    u_vars: dict[tuple[str, str, int], Any] = {}
    for block in copy_blocks:
        for subject in block.subjects:
            for pattern_index in range(len(subject_layers[subject].objective)):
                u_vars[(block.name, subject, pattern_index)] = model.addVar(
                    lb=-gp.GRB.INFINITY,
                    name=_var_name("u", block.name, subject, pattern_index),
                )

    theta_subject = {
        subject: model.addVar(lb=-gp.GRB.INFINITY, name=_var_name("theta", subject))
        for subject in subject_layers
    }
    theta_block = {
        block.name: model.addVar(lb=-gp.GRB.INFINITY, name=_var_name("phi", block.name))
        for block in copy_blocks
    }
    model.update()

    subject_incidence: dict[str, list[CopyBlock]] = {subject: [] for subject in subject_layers}
    for block in copy_blocks:
        for subject in block.subjects:
            subject_incidence[subject].append(block)

    for subject, layer in subject_layers.items():
        for pattern_index, objective in enumerate(layer.objective):
            rhs = objective + layer.clashes[pattern_index] * eta
            for block in subject_incidence[subject]:
                rhs += u_vars[(block.name, subject, pattern_index)]
            model.addConstr(
                theta_subject[subject] <= rhs,
                name=_var_name("subject_cut", subject, pattern_index),
            )

    for block in copy_blocks:
        for tuple_index, tuple_row in enumerate(block.tuples):
            rhs = block.objective[tuple_index] + block.clashes[tuple_index] * eta
            for pos, subject in enumerate(block.subjects):
                rhs -= u_vars[(block.name, subject, int(tuple_row[pos]))]
            model.addConstr(
                theta_block[block.name] <= rhs,
                name=_var_name("block_cut", block.name, tuple_index),
            )

    model.setObjective(
        gp.quicksum(theta_subject.values())
        + gp.quicksum(theta_block.values())
        - 15.0 * eta,
        gp.GRB.MAXIMIZE,
    )
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Full Lagrangian dual did not solve to optimality. Status={model.Status}")

    bound = float(model.ObjVal)
    history = pd.DataFrame(
        [
            {
                "iteration": 1.0,
                "bound": bound,
                "best_bound": bound,
                "dual_upper_bound": bound,
                "cp_gap": 0.0,
                "eta": float(eta.X),
                "agreement_l1": 0.0,
                "clash_total": np.nan,
                "clash_residual": np.nan,
            }
        ]
    )
    multiplier_values = [abs(var.X) for var in u_vars.values()]
    return history, {
        "best_bound": bound,
        "best_iteration": 1,
        "final_bound": bound,
        "final_best_bound": bound,
        "final_dual_upper_bound": bound,
        "final_cp_gap": 0.0,
        "final_eta": float(eta.X),
        "final_agreement_l1": 0.0,
        "final_clash_total": None,
        "final_clash_residual": None,
        "iterations_completed": 1,
        "best_eta": float(eta.X),
        "best_multiplier_linf": max(multiplier_values, default=0.0),
        "dual_rows": int(model.NumConstrs),
        "dual_cols": int(model.NumVars),
        "dual_iterations": float(model.IterCount),
        "dual_runtime_seconds": float(model.Runtime),
    }


def _set_proximal_objective(
    *,
    master: Any,
    gp: Any,
    theta: Any,
    u_vars: dict[tuple[str, str, int], Any],
    eta_var: Any,
    center_u: dict[tuple[str, str, int], float],
    center_eta: float,
    proximal_weight: float,
) -> None:
    objective = -theta
    if proximal_weight > 0:
        quad = gp.QuadExpr()
        for key, var in u_vars.items():
            diff = var - center_u[key]
            quad += diff * diff
        eta_diff = eta_var - center_eta
        quad += eta_diff * eta_diff
        objective += proximal_weight * quad
    master.setObjective(objective, gp.GRB.MINIMIZE)


def _evaluate_split_dual(
    *,
    subject_layers: dict[str, SubjectLayer],
    copy_blocks: list[CopyBlock],
    subject_incidence: dict[str, list[tuple[CopyBlock, int]]],
    eta: float,
) -> tuple[float, dict[tuple[str, str, int] | tuple[str], float], dict[str, Any]]:
    value = -15.0 * eta
    gradient: dict[tuple[str, str, int] | tuple[str], float] = {("eta",): -15.0}
    subject_choices: dict[str, int] = {}
    block_choices: dict[str, int] = {}
    agreement_l1 = 0.0
    clash_total = 0.0

    for subject, layer in subject_layers.items():
        adjusted = layer.objective + eta * layer.clashes
        adjusted = adjusted.copy()
        for block, pos in subject_incidence[subject]:
            adjusted += block.multipliers[pos]
        choice = int(np.argmin(adjusted))
        subject_choices[subject] = choice
        value += float(adjusted[choice])
        clash_total += float(layer.clashes[choice])
        gradient[("eta",)] = float(gradient[("eta",)] + layer.clashes[choice])

    for block in copy_blocks:
        adjusted = block.objective + eta * block.clashes
        adjusted = adjusted.copy()
        for pos, _subject in enumerate(block.subjects):
            adjusted -= block.multipliers[pos][block.tuples[:, pos]]
        choice = int(np.argmin(adjusted))
        block_choices[block.name] = choice
        value += float(adjusted[choice])
        clash_total += float(block.clashes[choice])
        gradient[("eta",)] = float(gradient[("eta",)] + block.clashes[choice])

        selected_tuple = block.tuples[choice]
        for pos, subject in enumerate(block.subjects):
            subject_choice = subject_choices[subject]
            block_choice = int(selected_tuple[pos])
            if subject_choice != block_choice:
                agreement_l1 += 2.0
            gradient[(_key_block(block.name), subject, subject_choice)] = (
                gradient.get((_key_block(block.name), subject, subject_choice), 0.0) + 1.0
            )
            gradient[(_key_block(block.name), subject, block_choice)] = (
                gradient.get((_key_block(block.name), subject, block_choice), 0.0) - 1.0
            )

    return value, gradient, {
        "subject_choices": subject_choices,
        "block_choices": block_choices,
        "agreement_l1": agreement_l1,
        "clash_total": clash_total,
        "clash_residual": clash_total - 15.0,
    }


def _load_multipliers(copy_blocks: list[CopyBlock], current_u: dict[tuple[str, str, int], float]) -> None:
    for block in copy_blocks:
        for pos, subject in enumerate(block.subjects):
            target = block.multipliers[pos]
            for pattern_index in range(len(target)):
                target[pattern_index] = current_u[(_key_block(block.name), subject, pattern_index)]


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _key_block(name: str) -> str:
    return name


def _var_name(*parts: Any) -> str:
    text = "_".join(str(part) for part in parts)
    return "".join(char if char.isalnum() or char == "_" else "_" for char in text)


def _plot_history(history: pd.DataFrame, output: Path) -> None:
    if history.empty:
        output.write_text("", encoding="utf-8")
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(history["iteration"], history["bound"], linewidth=1.2, alpha=0.45, label="evaluated bound")
    ax.plot(history["iteration"], history["best_bound"], linewidth=2.8, label="best lower bound")
    ax.plot(history["iteration"], history["dual_upper_bound"], linewidth=2.0, linestyle="--", label="bundle upper approx.")
    ax.axhline(15370.0, color="black", linewidth=2.0, linestyle=":", label="pattern LP 15370")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Bound")
    ax.set_title("Toy community Lagrangian variable-splitting bound")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
