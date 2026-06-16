from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter as pc

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
APPROX_SRC = SCRIPT_DIR.parents[2] / "approximated_tradeoff" / "src"
sys.path.insert(0, str(APPROX_SRC))
sys.path.insert(0, str(SCRIPT_DIR))

import mc_heuristics as mch  # noqa: E402
from vietnam_grasp_heuristics import budgeted_construct, improve_local_search  # noqa: E402
from vietnam_sparse_local_search import SparseSwapLocalSearch  # noqa: E402

OUTPUT_ROOT = Path(r"C:\local\Parvathy\Vietnam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original and sparse first-swap local search.")
    parser.add_argument(
        "--instances",
        type=Path,
        nargs="+",
        default=sorted((OUTPUT_ROOT / "optimization").glob("vietnam_10kmgrid_*km_threshold.npz")),
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[20])
    parser.add_argument("--constructor", choices=["greedy", "randomized", "sample", "random_plus"], default="greedy")
    parser.add_argument("--rcl-size", type=int, default=25)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_ROOT / "line_profiles" / "local_search_variant_benchmark.csv")
    return parser.parse_args()


def load_instance(path: Path) -> tuple[mch.MaxCoverInstance, dict]:
    data = np.load(path, allow_pickle=False)
    instance = mch.MaxCoverInstance(
        w=data["w"],
        ij_indptr=data["ij_indptr"],
        ij_indices=data["ij_indices"],
        ji_indptr=data["ji_indptr"],
        ji_indices=data["ji_indices"],
    )
    metadata = json.loads(str(data["metadata_json"]))
    return instance, metadata


def run_one(
    *,
    instance: mch.MaxCoverInstance,
    metadata: dict,
    instance_path: Path,
    budget: int,
    constructor: str,
    rcl_size: int,
    sample_size: int,
    seed: int,
) -> list[dict]:
    scale = float(metadata.get("weight_scale", 1.0))
    threshold_km = float(metadata.get("threshold_m", 0.0)) / 1000.0
    constructed = budgeted_construct(
        instance,
        budget,
        constructor=constructor,
        rcl_size=rcl_size,
        sample_size=sample_size,
        seed=seed,
    )

    rows: list[dict] = []
    for local_search in ["first", "first_sparse"]:
        start = pc()
        sparse_index = SparseSwapLocalSearch.from_instance(instance) if local_search == "first_sparse" else None
        improved = improve_local_search(
            instance,
            constructed,
            local_search=local_search,
            sparse_local_search=sparse_index,
        )
        wall_seconds = pc() - start
        rows.append(
            {
                "instance": instance_path.name,
                "threshold_km": threshold_km,
                "budget": int(budget),
                "constructor": constructor,
                "local_search": local_search,
                "objective_weight_units": int(improved.objective),
                "incremental_population": float(improved.objective / scale),
                "solution_size": int(len(improved.solution)),
                "moves": max(0, len(improved.objectives) - 1),
                "reported_seconds": float(improved.total_time),
                "wall_seconds_including_setup": float(wall_seconds),
                "selected_solution": json.dumps([int(x) for x in improved.solution]),
            }
        )
    rows[0]["objective_matches_sparse"] = rows[0]["objective_weight_units"] == rows[1]["objective_weight_units"]
    rows[1]["objective_matches_sparse"] = rows[0]["objective_matches_sparse"]
    rows[0]["solution_matches_sparse"] = rows[0]["selected_solution"] == rows[1]["selected_solution"]
    rows[1]["solution_matches_sparse"] = rows[0]["solution_matches_sparse"]
    return rows


def main() -> None:
    args = parse_args()
    all_rows: list[dict] = []
    for instance_path in args.instances:
        instance, metadata = load_instance(instance_path)
        for budget in args.budgets:
            print(f"Benchmarking {instance_path.name}, p={budget}")
            all_rows.extend(
                run_one(
                    instance=instance,
                    metadata=metadata,
                    instance_path=instance_path,
                    budget=budget,
                    constructor=args.constructor,
                    rcl_size=args.rcl_size,
                    sample_size=args.sample_size,
                    seed=args.seed,
                )
            )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(args.output_csv, index=False)
    print(json.dumps({"output_csv": str(args.output_csv), "rows": int(len(frame))}, indent=2))


if __name__ == "__main__":
    main()
