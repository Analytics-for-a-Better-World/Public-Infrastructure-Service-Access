from __future__ import annotations

import argparse
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
)
from src.anthony_model import parse_day_series, prepare_toy_inputs


@dataclass
class SubjectBlock:
    subject: str
    patterns: list[Pattern]
    objective: np.ndarray
    clashes: np.ndarray


@dataclass
class EdgeBlock:
    left: str
    right: str
    objective: np.ndarray
    clashes: np.ndarray
    compatible: np.ndarray
    u_left: np.ndarray
    u_right: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toy proof-of-concept Lagrangian variable-splitting bound."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--upper-bound", type=float, default=25190.0)
    parser.add_argument("--step-scale", type=float, default=1.8)
    parser.add_argument("--max-step", type=float, default=1000.0)
    parser.add_argument("--edge-limit", type=int, default=-1)
    parser.add_argument("--summary-output", type=Path, default=Path("toy_lagrangian_poc_summary.csv"))
    parser.add_argument("--history-output", type=Path, default=Path("toy_lagrangian_poc_history.csv"))
    parser.add_argument("--plot-output", type=Path, default=Path("toy_lagrangian_poc_bounds.png"))
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
    subject_blocks = _build_subject_blocks(patterns)
    edge_blocks = _build_edge_blocks(
        subjects=list(patterns),
        patterns=patterns,
        pairs=pairs,
        date_index=date_index,
        exam_lengths=exam_lengths,
        edge_limit=args.edge_limit,
    )
    history, summary = lagrangian_bound(
        subject_blocks=subject_blocks,
        edge_blocks=edge_blocks,
        iterations=args.iterations,
        upper_bound=args.upper_bound,
        step_scale=args.step_scale,
        max_step=args.max_step,
    )
    elapsed = time.perf_counter() - started

    history.to_csv(args.history_output, index=False)
    _plot_history(history, args.plot_output)

    summary.update(
        {
            "elapsed_seconds": elapsed,
            "subject_count": len(subject_blocks),
            "pattern_count": sum(len(block.patterns) for block in subject_blocks.values()),
            "edge_count": len(edge_blocks),
            "upper_bound": args.upper_bound,
            "iterations": args.iterations,
            "step_scale": args.step_scale,
            "max_step": args.max_step,
        }
    )
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)

    print("Lagrangian POC seconds:", round(elapsed, 6))
    print("Best Lagrangian lower bound:", summary["best_bound"])
    print("Best iteration:", summary["best_iteration"])
    print("Final lower bound:", summary["final_bound"])
    print("Final eta:", summary["final_eta"])
    print("Final agreement residual L1:", summary["final_agreement_l1"])
    print("Final clash residual:", summary["final_clash_residual"])
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved history to {args.history_output}")
    print(f"Saved plot to {args.plot_output}")


def _build_subject_blocks(patterns: dict[str, list[Pattern]]) -> dict[str, SubjectBlock]:
    result: dict[str, SubjectBlock] = {}
    for subject, subject_patterns in patterns.items():
        result[subject] = SubjectBlock(
            subject=subject,
            patterns=subject_patterns,
            objective=np.array([pattern.internal_objective for pattern in subject_patterns], dtype=float),
            clashes=np.array([pattern.internal_same_slot_clashes for pattern in subject_patterns], dtype=float),
        )
    return result


def _build_edge_blocks(
    *,
    subjects: list[str],
    patterns: dict[str, list[Pattern]],
    pairs: pd.DataFrame,
    date_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
    edge_limit: int,
) -> list[EdgeBlock]:
    scored_edges: list[tuple[float, str, str]] = []
    for left_pos, left in enumerate(subjects):
        for right in subjects[left_pos + 1 :]:
            score = _subject_pair_mass(left, right, patterns, pairs)
            scored_edges.append((score, left, right))
    scored_edges.sort(key=lambda row: (-row[0], row[1], row[2]))
    if edge_limit > 0:
        scored_edges = scored_edges[:edge_limit]

    edge_blocks: list[EdgeBlock] = []
    for _score, left, right in scored_edges:
        left_patterns = patterns[left]
        right_patterns = patterns[right]
        objective = np.zeros((len(left_patterns), len(right_patterns)), dtype=float)
        clashes = np.zeros_like(objective)
        compatible = np.ones_like(objective, dtype=bool)
        for i, left_pattern in enumerate(left_patterns):
            for j, right_pattern in enumerate(right_patterns):
                cost = _cross_cost(
                    left_pattern.assignment,
                    right_pattern.assignment,
                    pairs=pairs,
                    date_index=date_index,
                    exam_lengths=exam_lengths,
                )
                objective[i, j] = cost.objective
                clashes[i, j] = cost.same_slot_clashes
                compatible[i, j] = not cost.daily_length_infeasible
        edge_blocks.append(
            EdgeBlock(
                left=left,
                right=right,
                objective=objective,
                clashes=clashes,
                compatible=compatible,
                u_left=np.zeros(len(left_patterns), dtype=float),
                u_right=np.zeros(len(right_patterns), dtype=float),
            )
        )
    return edge_blocks


def lagrangian_bound(
    *,
    subject_blocks: dict[str, SubjectBlock],
    edge_blocks: list[EdgeBlock],
    iterations: int,
    upper_bound: float,
    step_scale: float,
    max_step: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    incident: dict[str, list[tuple[EdgeBlock, str]]] = {subject: [] for subject in subject_blocks}
    for edge in edge_blocks:
        incident[edge.left].append((edge, "left"))
        incident[edge.right].append((edge, "right"))

    eta = 0.0
    best_bound = -float("inf")
    best_iteration = 0
    history: list[dict[str, float]] = []

    for iteration in range(1, iterations + 1):
        value, state = _evaluate_lagrangian(
            subject_blocks=subject_blocks,
            edge_blocks=edge_blocks,
            incident=incident,
            eta=eta,
        )
        if value > best_bound:
            best_bound = value
            best_iteration = iteration

        norm_sq = state["agreement_norm_sq"] + state["clash_residual"] ** 2
        if norm_sq > 1e-12 and upper_bound > value:
            step = min(max_step, step_scale * (upper_bound - value) / norm_sq)
        else:
            step = 0.0

        for edge in edge_blocks:
            left_choice, right_choice = state["edge_choices"][(edge.left, edge.right)]
            left_subject_choice = state["subject_choices"][edge.left]
            right_subject_choice = state["subject_choices"][edge.right]

            left_residual = np.zeros_like(edge.u_left)
            right_residual = np.zeros_like(edge.u_right)
            left_residual[left_subject_choice] += 1.0
            left_residual[left_choice] -= 1.0
            right_residual[right_subject_choice] += 1.0
            right_residual[right_choice] -= 1.0

            edge.u_left += step * left_residual
            edge.u_right += step * right_residual

        eta = max(0.0, eta + step * state["clash_residual"])

        history.append(
            {
                "iteration": float(iteration),
                "bound": float(value),
                "best_bound": float(best_bound),
                "step": float(step),
                "eta": float(eta),
                "agreement_l1": float(state["agreement_l1"]),
                "agreement_norm_sq": float(state["agreement_norm_sq"]),
                "clash_total": float(state["clash_total"]),
                "clash_residual": float(state["clash_residual"]),
            }
        )

    final = history[-1] if history else {}
    summary = {
        "best_bound": best_bound,
        "best_iteration": best_iteration,
        "final_bound": final.get("bound"),
        "final_best_bound": final.get("best_bound"),
        "final_step": final.get("step"),
        "final_eta": final.get("eta"),
        "final_agreement_l1": final.get("agreement_l1"),
        "final_clash_total": final.get("clash_total"),
        "final_clash_residual": final.get("clash_residual"),
    }
    return pd.DataFrame(history), summary


def _evaluate_lagrangian(
    *,
    subject_blocks: dict[str, SubjectBlock],
    edge_blocks: list[EdgeBlock],
    incident: dict[str, list[tuple[EdgeBlock, str]]],
    eta: float,
) -> tuple[float, dict[str, Any]]:
    value = -15.0 * eta
    subject_choices: dict[str, int] = {}
    edge_choices: dict[tuple[str, str], tuple[int, int]] = {}
    clash_total = -15.0

    for subject, block in subject_blocks.items():
        adjusted = block.objective + eta * block.clashes
        adjusted = adjusted.copy()
        for edge, side in incident[subject]:
            adjusted += edge.u_left if side == "left" else edge.u_right
        choice = int(np.argmin(adjusted))
        subject_choices[subject] = choice
        value += float(adjusted[choice])
        clash_total += float(block.clashes[choice])

    agreement_l1 = 0.0
    agreement_norm_sq = 0.0
    for edge in edge_blocks:
        adjusted = edge.objective + eta * edge.clashes - edge.u_left[:, None] - edge.u_right[None, :]
        adjusted = np.where(edge.compatible, adjusted, np.inf)
        flat_choice = int(np.argmin(adjusted))
        left_choice, right_choice = np.unravel_index(flat_choice, adjusted.shape)
        edge_choices[(edge.left, edge.right)] = (int(left_choice), int(right_choice))
        value += float(adjusted[left_choice, right_choice])
        clash_total += float(edge.clashes[left_choice, right_choice])

        left_subject_choice = subject_choices[edge.left]
        right_subject_choice = subject_choices[edge.right]
        left_l1 = 0.0 if left_choice == left_subject_choice else 2.0
        right_l1 = 0.0 if right_choice == right_subject_choice else 2.0
        agreement_l1 += left_l1 + right_l1
        if left_choice != left_subject_choice:
            agreement_norm_sq += 2.0
        if right_choice != right_subject_choice:
            agreement_norm_sq += 2.0

    return value, {
        "subject_choices": subject_choices,
        "edge_choices": edge_choices,
        "agreement_l1": agreement_l1,
        "agreement_norm_sq": agreement_norm_sq,
        "clash_total": clash_total + 15.0,
        "clash_residual": clash_total,
    }


def _subject_pair_mass(left: str, right: str, patterns: dict[str, list[Pattern]], pairs: pd.DataFrame) -> float:
    left_exams = set().union(*(pattern.assignment.keys() for pattern in patterns[left]))
    right_exams = set().union(*(pattern.assignment.keys() for pattern in patterns[right]))
    return sum(float(pairs.loc[exam_i, exam_j]) for exam_i in left_exams for exam_j in right_exams)


def _plot_history(history: pd.DataFrame, output: Path) -> None:
    if history.empty:
        output.write_text("", encoding="utf-8")
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(history["iteration"], history["bound"], linewidth=1.2, alpha=0.45, label="current bound")
    ax.plot(history["iteration"], history["best_bound"], linewidth=2.8, label="best bound")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Lower bound")
    ax.set_title("Toy Lagrangian variable-splitting bound")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
