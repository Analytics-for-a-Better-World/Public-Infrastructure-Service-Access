from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
APPROX_SRC = SCRIPT_DIR.parents[2] / "approximated_tradeoff" / "src"
sys.path.insert(0, str(APPROX_SRC))
sys.path.insert(0, str(SCRIPT_DIR))

import mc_heuristics as mch  # noqa: E402
from vietnam_grasp_heuristics import run_grasp  # noqa: E402

OUTPUT_ROOT = Path(r"C:\local\Parvathy\Vietnam")


def load_instance(path: Path) -> tuple[mch.MaxCoverInstance, dict, np.lib.npyio.NpzFile]:
    data = np.load(path, allow_pickle=False)
    instance = mch.MaxCoverInstance(
        w=data["w"],
        ij_indptr=data["ij_indptr"],
        ij_indices=data["ij_indices"],
        ji_indptr=data["ji_indptr"],
        ji_indices=data["ji_indices"],
    )
    metadata = json.loads(str(data["metadata_json"]))
    return instance, metadata, data


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scalable fresh-PISA Vietnam max-cover heuristics.")
    parser.add_argument("--instance-npz", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--constructor", choices=["greedy", "randomized", "sample", "random_plus"], default="randomized")
    parser.add_argument("--rcl-size", type=int, default=25)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--random-plus-fraction", type=float, default=0.15)
    parser.add_argument("--local-search", choices=["first", "first_sparse", "none"], default="first_sparse")
    parser.add_argument("--path-relinking", type=parse_bool, default=True)
    parser.add_argument("--path-relinking-method", choices=["fast", "original"], default="fast")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pool", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "grasp_latest")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance, metadata, data = load_instance(args.instance_npz)
    scale = float(metadata.get("weight_scale", 1.0))
    baseline = float(metadata.get("baseline_covered_population", 0.0))
    diagnostics = {
        "instance_npz": str(args.instance_npz),
        "n_population": int(instance.n_households),
        "n_candidates": int(instance.n_facilities),
        "threshold_m": metadata.get("threshold_m"),
        "baseline_covered_population": baseline,
        "incremental_population_available": float(instance.w.sum() / scale),
        "total_population": metadata.get("total_population"),
        "budget": args.budget,
    }
    print(json.dumps(diagnostics, indent=2))
    if args.dry_run:
        return

    best, records = run_grasp(
        instance,
        args.budget,
        time_limit_seconds=args.time_limit_seconds,
        max_iterations=args.max_iterations,
        constructor=args.constructor,
        rcl_size=args.rcl_size,
        sample_size=args.sample_size,
        random_plus_fraction=args.random_plus_fraction,
        local_search=args.local_search,
        path_relinking=args.path_relinking,
        path_relinking_method=args.path_relinking_method,
        seed=args.seed,
        max_pool=args.max_pool,
    )

    candidate_ids = data["candidate_source_ids"].astype(str)
    selected_source_ids = [str(candidate_ids[int(j)]) for j in best.solution if int(j) < len(candidate_ids)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.instance_npz.stem
    run_name = (
        f"{stem}_p{args.budget}_{args.constructor}_rcl{args.rcl_size}_"
        f"{args.local_search}_pr{str(args.path_relinking).lower()}_seed{args.seed}"
    )
    rows = []
    for rec in records:
        row = asdict(rec)
        row["selected_solution"] = json.dumps(row["selected_solution"])
        row["best_incremental_population"] = row["best_objective"] / scale
        row["best_total_covered_population"] = baseline + row["best_incremental_population"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_dir / f"{run_name}_trace.csv", index=False)
    summary = {
        "run_name": run_name,
        "instance_npz": str(args.instance_npz),
        "budget": args.budget,
        "constructor": args.constructor,
        "rcl_size": args.rcl_size,
        "sample_size": args.sample_size,
        "random_plus_fraction": args.random_plus_fraction,
        "local_search": args.local_search,
        "path_relinking": args.path_relinking,
        "path_relinking_method": args.path_relinking_method,
        "seed": args.seed,
        "best_objective_weight_units": int(best.objective),
        "best_incremental_population": float(best.objective / scale),
        "baseline_covered_population": baseline,
        "best_total_covered_population": float(baseline + best.objective / scale),
        "selected_candidate_indices": [int(x) for x in best.solution],
        "selected_candidate_source_ids": selected_source_ids,
        "metadata": metadata,
    }
    (args.output_dir / f"{run_name}_best_solution.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_incremental_population": summary["best_incremental_population"],
        "best_total_covered_population": summary["best_total_covered_population"],
        "iterations": len(records),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
