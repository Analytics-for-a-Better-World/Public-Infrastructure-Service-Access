from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from run_full_pattern_lns import _pair_objective, improve_with_pattern_lns
from src.anthony_model import (
    DEFAULT_WEIGHTS,
    load_default_data,
    mip_objective_value,
    placement_from_timetable,
    prepare_anthony_model_data,
)
from src.full_heuristic import _pair_value_map, _same_slot_clashes, validate_full_solution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic subject-pattern LNS sweeps on the full IB instance."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("pattern_sweep_runs"))
    parser.add_argument("--prefix", default="pattern_sweep")
    parser.add_argument("--final-output", type=Path, default=None)
    parser.add_argument("--history-output", type=Path, default=None)
    parser.add_argument("--nb-days", type=int, default=34)
    parser.add_argument("--objective-mode", choices=["formal", "anthony_appendix"], default="formal")
    parser.add_argument("--band-size", type=int, default=8)
    parser.add_argument("--max-bands", type=int, default=4)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument(
        "--strategies",
        default="contribution_bands,interaction_edges",
        help="Comma-separated strategies: contribution_bands, interaction_edges.",
    )
    parser.add_argument("--edge-neighborhoods", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--output-flag", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_output = args.final_output or args.output_dir / f"{args.prefix}_final.csv"
    history_output = args.history_output or args.output_dir / f"{args.prefix}_history.csv"

    exams, days, pairs = load_default_data(args.data_dir)
    data = prepare_anthony_model_data(exams, days, pairs, nb_days=args.nb_days, recode_math_paper_three=True)
    current = _load_timetable(args.start)
    validate_full_solution(current, data)

    strategies = [strategy.strip() for strategy in args.strategies.split(",") if strategy.strip()]
    allowed = {"contribution_bands", "interaction_edges"}
    unknown = sorted(set(strategies) - allowed)
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")

    history: list[dict[str, Any]] = []
    step = 0
    total_started = time.perf_counter()
    for pass_index in range(1, args.passes + 1):
        for strategy in strategies:
            if strategy == "contribution_bands":
                step = _run_contribution_band_sweep(
                    current_ref={"timetable": current},
                    data=data,
                    step=step,
                    pass_index=pass_index,
                    args=args,
                    history=history,
                )
                current = current_ref_timetable(history, current)
            elif strategy == "interaction_edges":
                step = _run_interaction_edge_sweep(
                    current_ref={"timetable": current},
                    data=data,
                    step=step,
                    pass_index=pass_index,
                    args=args,
                    history=history,
                )
                current = current_ref_timetable(history, current)

    current.to_csv(final_output, index=False)
    pd.DataFrame(history).to_csv(history_output, index=False)

    final_objective = mip_objective_value(current, data.pairs, data.days, mode=args.objective_mode)
    final_same_slot = _same_slot_clashes(
        placement_from_timetable(current),
        data.pairs,
        pair_values=_pair_value_map(data.pairs),
    )
    elapsed = time.perf_counter() - total_started

    print("Start file:", args.start)
    print("Final output:", final_output)
    print("History output:", history_output)
    print("Final objective:", final_objective)
    print("Final same-slot clashes:", final_same_slot)
    print("Total accepted improvement:", _total_improvement(history))
    print("Neighborhoods solved:", len(history))
    print("Elapsed seconds:", round(elapsed, 6))


def _run_contribution_band_sweep(
    *,
    current_ref: dict[str, pd.DataFrame],
    data: Any,
    step: int,
    pass_index: int,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> int:
    subjects = sorted(data.exams["Subject"].unique())
    for band_index in range(args.max_bands):
        ranking = _subject_contribution_order(
            current_ref["timetable"],
            data=data,
            objective_mode=args.objective_mode,
        )
        start = band_index * args.band_size
        if start >= len(ranking):
            break
        if band_index == args.max_bands - 1:
            selected = ranking[start:]
        else:
            selected = ranking[start : start + args.band_size]
        selected = [subject for subject in selected if subject in subjects]
        if len(selected) < 2:
            continue
        step += 1
        current_ref["timetable"] = _solve_and_record(
            current_ref["timetable"],
            data=data,
            selected_subjects=selected,
            step=step,
            pass_index=pass_index,
            strategy="contribution_bands",
            args=args,
            history=history,
            label=f"band{band_index + 1}",
        )
    return step


def _run_interaction_edge_sweep(
    *,
    current_ref: dict[str, pd.DataFrame],
    data: Any,
    step: int,
    pass_index: int,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> int:
    neighborhoods = _interaction_neighborhoods(
        current_ref["timetable"],
        data=data,
        objective_mode=args.objective_mode,
        size=args.band_size,
        limit=args.edge_neighborhoods,
    )
    for edge_index, selected in enumerate(neighborhoods, start=1):
        step += 1
        current_ref["timetable"] = _solve_and_record(
            current_ref["timetable"],
            data=data,
            selected_subjects=selected,
            step=step,
            pass_index=pass_index,
            strategy="interaction_edges",
            args=args,
            history=history,
            label=f"edge{edge_index}",
        )
    return step


def _solve_and_record(
    current: pd.DataFrame,
    *,
    data: Any,
    selected_subjects: list[str],
    step: int,
    pass_index: int,
    strategy: str,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    label: str,
) -> pd.DataFrame:
    base = f"{args.prefix}_p{pass_index:02d}_s{step:03d}_{strategy}_{label}"
    output = args.output_dir / f"{base}.csv"
    log_output = args.output_dir / f"{base}.log"
    started = time.perf_counter()
    result = improve_with_pattern_lns(
        start_timetable=current,
        data=data,
        selected_subjects=selected_subjects,
        time_limit=args.time_limit,
        objective_mode=args.objective_mode,
        log_output=log_output,
        output_flag=args.output_flag,
    )
    elapsed = time.perf_counter() - started
    result["timetable"].to_csv(output, index=False)

    row = {
        "step": step,
        "pass": pass_index,
        "strategy": strategy,
        "label": label,
        "selected_subjects": ";".join(selected_subjects),
        "start_objective": result["start_objective"],
        "candidate_objective": result["candidate_objective"],
        "solver_candidate_objective": result["solver_candidate_objective"],
        "accepted": result["accepted"],
        "improvement": result["start_objective"] - result["candidate_objective"],
        "start_same_slot_clashes": result["start_same_slot_clashes"],
        "candidate_same_slot_clashes": result["candidate_same_slot_clashes"],
        "status": result["status"],
        "best_bound": result["best_bound"],
        "gap": result["gap"],
        "nodes": result["nodes"],
        "iterations": result["iterations"],
        "work": result["work"],
        "runtime_seconds": result["runtime_seconds"],
        "elapsed_seconds": elapsed,
        "pattern_count": result["pattern_count"],
        "mu_count": result["mu_count"],
        "incompatible_mu_count": result["incompatible_mu_count"],
        "output": str(output),
        "log_output": str(log_output),
    }
    history.append(row)
    print(
        f"step {step:03d} {strategy}/{label}: "
        f"{row['start_objective']:.0f} -> {row['candidate_objective']:.0f} "
        f"(gain {row['improvement']:.0f}, gap {row['gap']})"
    )
    return result["timetable"]


def current_ref_timetable(history: list[dict[str, Any]], current: pd.DataFrame) -> pd.DataFrame:
    if not history:
        return current
    last_output = Path(history[-1]["output"])
    if last_output.exists():
        return _load_timetable(last_output)
    return current


def _subject_contribution_order(
    timetable: pd.DataFrame,
    *,
    data: Any,
    objective_mode: str,
) -> list[str]:
    subject_values, _pair_values = _subject_contributions(
        timetable,
        data=data,
        objective_mode=objective_mode,
    )
    return [
        subject
        for subject, _value in sorted(subject_values.items(), key=lambda item: (-item[1], item[0]))
    ]


def _interaction_neighborhoods(
    timetable: pd.DataFrame,
    *,
    data: Any,
    objective_mode: str,
    size: int,
    limit: int,
) -> list[list[str]]:
    subject_values, pair_values = _subject_contributions(
        timetable,
        data=data,
        objective_mode=objective_mode,
    )
    cross_edges = [
        (subjects, value)
        for subjects, value in pair_values.items()
        if subjects[0] != subjects[1] and value > 0
    ]
    cross_edges.sort(key=lambda item: (-item[1], item[0]))

    adjacency: dict[str, dict[str, float]] = {}
    for (left, right), value in cross_edges:
        adjacency.setdefault(left, {})[right] = adjacency.setdefault(left, {}).get(right, 0.0) + value
        adjacency.setdefault(right, {})[left] = adjacency.setdefault(right, {}).get(left, 0.0) + value

    neighborhoods: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for (left, right), _value in cross_edges:
        selected = [left, right]
        while len(selected) < size:
            candidates = [
                subject
                for subject in subject_values
                if subject not in selected
            ]
            if not candidates:
                break
            candidates.sort(
                key=lambda subject: (
                    -sum(adjacency.get(subject, {}).get(chosen, 0.0) for chosen in selected),
                    -subject_values[subject],
                    subject,
                )
            )
            selected.append(candidates[0])
        signature = tuple(sorted(selected))
        if signature in seen:
            continue
        seen.add(signature)
        neighborhoods.append(selected)
        if len(neighborhoods) >= limit:
            break
    return neighborhoods


def _subject_contributions(
    timetable: pd.DataFrame,
    *,
    data: Any,
    objective_mode: str,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    placement = placement_from_timetable(timetable)
    day_index = {date: idx for idx, date in enumerate(data.dates)}
    exam_subject = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    subject_values = {subject: 0.0 for subject in data.exams["Subject"].unique()}
    pair_values: dict[tuple[str, str], float] = {}
    exams = sorted(placement)
    for left_pos, exam_i in enumerate(exams):
        for exam_j in exams[left_pos + 1 :]:
            value = _pair_objective(
                exam_i,
                placement[exam_i],
                exam_j,
                placement[exam_j],
                pair_values=None,
                pairs=data.pairs,
                day_index=day_index,
                objective_mode=objective_mode,
                weights=DEFAULT_WEIGHTS,
            )
            if value == 0:
                continue
            subject_i = exam_subject[exam_i]
            subject_j = exam_subject[exam_j]
            key = tuple(sorted((subject_i, subject_j)))
            pair_values[key] = pair_values.get(key, 0.0) + value
            if subject_i == subject_j:
                subject_values[subject_i] += value
            else:
                subject_values[subject_i] += 0.5 * value
                subject_values[subject_j] += 0.5 * value
    return subject_values, pair_values


def _load_timetable(path: Path) -> pd.DataFrame:
    timetable = pd.read_csv(path)
    timetable["Date"] = pd.to_datetime(timetable["Date"], dayfirst=True).dt.normalize()
    return timetable


def _total_improvement(history: list[dict[str, Any]]) -> float:
    return sum(float(row["improvement"]) for row in history)


if __name__ == "__main__":
    main()
