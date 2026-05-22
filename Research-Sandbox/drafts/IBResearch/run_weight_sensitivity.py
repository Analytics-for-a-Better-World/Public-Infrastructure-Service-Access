from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.anthony_model import (
    DEFAULT_WEIGHTS,
    apply_timetable_start,
    build_anthony_mip_model,
    load_default_data,
    mip_objective_value,
    prepare_toy_inputs,
    timetable_from_solution,
)
from src.full_heuristic import validate_full_solution


WEIGHT_VARIANTS: dict[str, dict[str, float]] = {
    "anthony": {"a": 64, "b": 32, "c": 16, "d": 8, "e": 4, "f": 2, "g": 1},
    "flat_proximity": {"a": 64, "b": 16, "c": 16, "d": 16, "e": 16, "f": 16, "g": 16},
    "gentle_halving": {"a": 32, "b": 16, "c": 8, "d": 4, "e": 2, "f": 1, "g": 1},
    "same_slot_heavy": {"a": 128, "b": 32, "c": 16, "d": 8, "e": 4, "f": 2, "g": 1},
    "near_gap_heavy": {"a": 64, "b": 48, "c": 32, "d": 8, "e": 4, "f": 2, "g": 1},
    "long_gap_light": {"a": 64, "b": 32, "c": 16, "d": 4, "e": 2, "f": 1, "g": 0},
    "julien_experiment_e": {"a": 0, "b": -0.5, "c": -1, "d": -2, "e": -3, "f": -4, "g": -5},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test sensitivity of MILP gap to objective weight choices.")
    parser.add_argument("--instance", choices=["toy", "full"], default="toy")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("weight_sensitivity_summary.csv"))
    parser.add_argument("--timetable-dir", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--nb-days", type=int, default=None)
    parser.add_argument("--objective-mode", choices=["formal", "anthony_appendix"], default=None)
    parser.add_argument("--variants", nargs="*", default=list(WEIGHT_VARIANTS))
    parser.add_argument("--custom-weights-json", type=Path, default=None)
    parser.add_argument("--y-binary", action="store_true")
    parser.add_argument("--strengthen-y-upper-bounds", action="store_true")
    parser.add_argument("--proximity-at-most-one", action="store_true")
    parser.add_argument("--enforce-subject-exam-order", action="store_true")
    parser.add_argument("--symmetry", type=int, default=None)
    args = parser.parse_args()

    variants = dict(WEIGHT_VARIANTS)
    if args.custom_weights_json is not None:
        with args.custom_weights_json.open("r", encoding="utf-8") as file:
            custom = json.load(file)
        variants.update({str(name): _normalize_weights(weights) for name, weights in custom.items()})

    rows = []
    timetable_dir = args.timetable_dir
    if timetable_dir is not None:
        timetable_dir.mkdir(parents=True, exist_ok=True)

    for variant_name in args.variants:
        if variant_name not in variants:
            raise ValueError(f"Unknown weight variant {variant_name!r}. Available: {sorted(variants)}")
        weights = variants[variant_name]
        row = _run_variant(
            variant_name=variant_name,
            weights=weights,
            instance=args.instance,
            data_dir=args.data_dir,
            start_path=args.start,
            time_limit=args.time_limit,
            nb_days=args.nb_days,
            objective_mode=args.objective_mode,
            y_binary=args.y_binary,
            strengthen_y_upper_bounds=args.strengthen_y_upper_bounds,
            proximity_at_most_one=args.proximity_at_most_one,
            enforce_subject_exam_order=args.enforce_subject_exam_order,
            symmetry=args.symmetry,
            timetable_dir=timetable_dir,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(pd.DataFrame([row]).to_string(index=False))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output, index=False)
    print(f"Saved summary to {args.output}")


def _run_variant(
    *,
    variant_name: str,
    weights: dict[str, float],
    instance: str,
    data_dir: Path,
    start_path: Path | None,
    time_limit: float,
    nb_days: int | None,
    objective_mode: str | None,
    y_binary: bool,
    strengthen_y_upper_bounds: bool,
    proximity_at_most_one: bool,
    enforce_subject_exam_order: bool,
    symmetry: int | None,
    timetable_dir: Path | None,
) -> dict[str, object]:
    if instance == "toy":
        objective_mode = objective_mode or "anthony_appendix"
        toy_exams = pd.read_csv(data_dir / "Toy exam list.csv")
        toy_pairs = pd.read_csv(data_dir / "Exam pairs.csv")
        exams, days, pairs = prepare_toy_inputs(toy_exams, toy_pairs)
        model_kwargs = {
            "nb_days": len(days),
            "max_clashes": 15,
            "max_afternoon_minutes": 180,
            "max_daily_minutes": 375,
            "forbid_weekends": True,
            "forbid_may_first": True,
            "forbid_language_fridays": True,
            "forbid_language_friday_afternoons": False,
            "consecutive_subject_exams": True,
            "consecutive_usable_subject_exams": False,
            "first_half_subjects": {"Finance", "Law and Ethics"},
            "recode_math_paper_three": False,
        }
        start = pd.read_csv(start_path) if start_path is not None else None
    else:
        objective_mode = objective_mode or "formal"
        exams, days, pairs = load_default_data(data_dir)
        model_kwargs = {
            "nb_days": nb_days or 34,
            "max_clashes": 10_000,
            "max_afternoon_minutes": 180,
            "max_daily_minutes": 385,
            "forbid_weekends": True,
            "forbid_may_first": True,
            "forbid_language_fridays": True,
            "forbid_language_friday_afternoons": False,
            "consecutive_subject_exams": True,
            "consecutive_usable_subject_exams": False,
            "first_half_subjects": None,
            "recode_math_paper_three": True,
        }
        start = pd.read_csv(start_path) if start_path is not None else None

    built = build_anthony_mip_model(
        exams=exams,
        days=days,
        pairs=pairs,
        weights=weights,
        y_binary=y_binary,
        strengthen_y_upper_bounds=strengthen_y_upper_bounds,
        proximity_at_most_one=proximity_at_most_one,
        enforce_subject_exam_order=enforce_subject_exam_order,
        objective_mode=objective_mode,
        output_flag=0,
        model_name=f"{instance}_weight_sensitivity_{variant_name}",
        **model_kwargs,
    )
    built.model.setParam("TimeLimit", time_limit)
    if symmetry is not None:
        built.model.setParam("Symmetry", symmetry)

    start_objective = None
    if start is not None:
        if instance == "full":
            validate_full_solution(start, built.data)
        apply_timetable_start(built, start)
        start_objective = mip_objective_value(start, built.data.pairs, built.data.days, weights=weights, mode=objective_mode)

    t0 = time.perf_counter()
    built.model.optimize()
    seconds = time.perf_counter() - t0

    incumbent_objective = None
    timetable_path = None
    if built.model.SolCount:
        timetable = timetable_from_solution(built)
        incumbent_objective = mip_objective_value(
            timetable,
            built.data.pairs,
            built.data.days,
            weights=weights,
            mode=objective_mode,
        )
        if timetable_dir is not None:
            timetable_path = timetable_dir / f"{instance}_{variant_name}_timetable.csv"
            timetable.to_csv(timetable_path, index=False)

    bound = float(built.model.ObjBound) if built.model.SolCount or built.model.Status else None
    gap = float(built.model.MIPGap) if built.model.SolCount else None
    return {
        "instance": instance,
        "variant": variant_name,
        "weights": json.dumps(weights, sort_keys=True),
        "objective_mode": objective_mode,
        "time_limit": time_limit,
        "seconds": seconds,
        "status": int(built.model.Status),
        "start_objective": start_objective,
        "incumbent": incumbent_objective,
        "best_bound": bound,
        "gap": gap,
        "node_count": float(built.model.NodeCount),
        "iteration_count": float(built.model.IterCount),
        "timetable": str(timetable_path) if timetable_path is not None else None,
    }


def _normalize_weights(weights: dict[str, int | float]) -> dict[str, float]:
    missing = set(DEFAULT_WEIGHTS).difference(weights)
    if missing:
        raise ValueError(f"Custom weight vector is missing categories: {sorted(missing)}")
    return {category: float(weights[category]) for category in DEFAULT_WEIGHTS}


if __name__ == "__main__":
    main()
