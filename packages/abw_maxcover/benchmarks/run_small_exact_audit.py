from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np
from environment import runtime_metadata, write_manifest

from abw_maxcover import HeuristicConfig, approximate_pareto_curve, build_instance
from abw_maxcover._incremental_core import compute_coverage_and_objective


def random_instance(seed: int, n_demand: int = 30, n_facilities: int = 14):
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 50, size=n_demand, dtype=np.int64)
    ij: list[list[int]] = [[] for _ in range(n_demand)]
    ji: list[list[int]] = [[] for _ in range(n_facilities)]
    for facility in range(n_facilities):
        demand = np.flatnonzero(rng.random(n_demand) < 0.22).astype(np.int32)
        if not demand.size:
            demand = np.asarray([facility % n_demand], dtype=np.int32)
        ji[facility] = demand.tolist()
        for item in demand:
            ij[int(item)].append(facility)
    return build_instance(weights, ij, ji, name=f"audit_{seed}", validate_consistency=True)


def brute_force(instance, budget: int) -> int:
    best = 0
    for size in range(min(budget, instance.n_facilities) + 1):
        for solution in itertools.combinations(range(instance.n_facilities), size):
            _, objective = compute_coverage_and_objective(instance, list(solution))
            best = max(best, objective)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for seed in range(args.seeds):
        instance = random_instance(seed)
        budgets = [1, 2, 3, 4]
        curve = approximate_pareto_curve(
            instance,
            budgets,
            config=HeuristicConfig(
                constructors=("greedy", "compact", "regreedy"),
                randomized_repeats=0,
                local_search="first_sparse",
                use_path_relinking=False,
            ),
        )
        for result in curve.results:
            optimum = brute_force(instance, result.budget)
            ratio = 1.0 if optimum == 0 else float(result.objective or 0) / optimum
            rows.append(
                {
                    "seed": seed,
                    "budget": result.budget,
                    "heuristic": result.objective,
                    "optimum": optimum,
                    "ratio": ratio,
                    "guarantee_satisfied": ratio + 1e-12 >= 1.0 - 1.0 / np.e,
                }
            )
    csv_path = args.output / "small_exact_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not all(bool(row["guarantee_satisfied"]) for row in rows):
        raise RuntimeError("greedy approximation guarantee failed")
    write_manifest(
        args.output / "small_exact_audit_manifest.json",
        {
            "runtime": runtime_metadata(),
            "seeds": args.seeds,
            "rows": len(rows),
            "minimum_ratio": min(float(row["ratio"]) for row in rows),
            "wall_seconds": time.perf_counter() - start,
            "results": str(csv_path.resolve()),
        },
    )


if __name__ == "__main__":
    main()
