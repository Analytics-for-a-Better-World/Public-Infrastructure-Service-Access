from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from environment import runtime_metadata, write_manifest

from abw_maxcover import HeuristicConfig, approximate_pareto_curve, build_instance


@dataclass(frozen=True, slots=True)
class ScalingCase:
    name: str
    n_demand: int
    n_facilities: int
    degree: int
    max_budget: int
    seed: int


CASES = (
    ScalingCase("square_1k", 1_000, 1_000, 12, 100, 101),
    ScalingCase("square_10k", 10_000, 10_000, 24, 500, 102),
    ScalingCase("asymmetric_100k", 100_000, 100, 4, 100, 103),
    ScalingCase("asymmetric_1m", 1_000_000, 100, 4, 100, 104),
)


def build_case(case: ScalingCase):
    rng = np.random.default_rng(case.seed)
    weights = rng.integers(1, 101, size=case.n_demand, dtype=np.int64)
    ij: list[list[int]] = [[] for _ in range(case.n_demand)]
    ji: list[list[int]] = [[] for _ in range(case.n_facilities)]
    degree = min(case.degree, case.n_facilities)
    for demand in range(case.n_demand):
        facilities = rng.choice(case.n_facilities, size=degree, replace=False)
        for facility in facilities:
            facility_i = int(facility)
            ij[demand].append(facility_i)
            ji[facility_i].append(demand)
    return build_instance(
        weights,
        ij,
        ji,
        name=case.name,
        assume_unique_sorted=True,
        validate_consistency=False,
        metadata={"generator": asdict(case)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--skip-million", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    campaign_start = time.perf_counter()
    for case in CASES:
        if args.skip_million and case.n_demand >= 1_000_000:
            continue
        build_start = time.perf_counter()
        instance = build_case(case)
        build_seconds = time.perf_counter() - build_start
        budgets = sorted({0, case.max_budget // 4, case.max_budget // 2, case.max_budget})
        solve_start = time.perf_counter()
        curve = approximate_pareto_curve(
            instance,
            budgets,
            config=HeuristicConfig(
                constructors=("greedy", "compact", "regreedy"),
                randomized_repeats=0,
                local_search="none",
                use_path_relinking=False,
            ),
        )
        solve_seconds = time.perf_counter() - solve_start
        rows.append(
            {
                **asdict(case),
                "incidences": int(instance.ji_indices.size),
                "build_seconds": build_seconds,
                "frontier_seconds": solve_seconds,
                "last_objective": int(curve.results[-1].objective or 0),
            }
        )
        print(f"{case.name}: {instance.ji_indices.size:,} incidences, {solve_seconds:.3f}s")

    csv_path = args.output / "scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_manifest(
        args.output / "scaling_manifest.json",
        {
            "runtime": runtime_metadata(),
            "cases": [asdict(case) for case in CASES],
            "rows": len(rows),
            "wall_seconds": time.perf_counter() - campaign_start,
            "results": str(csv_path.resolve()),
        },
    )


if __name__ == "__main__":
    main()
