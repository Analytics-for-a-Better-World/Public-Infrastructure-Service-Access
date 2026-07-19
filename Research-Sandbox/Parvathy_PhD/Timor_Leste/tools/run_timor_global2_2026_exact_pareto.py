from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


ROOT = Path(
    os.environ.get(
        "PARVATHY_REPLICATION_ROOT",
        r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis",
    )
)
REPOSITORY = Path(
    os.environ.get(
        "PISA_REPOSITORY",
        str(Path(__file__).resolve().parents[4]),
    )
)
ABW_SRC = REPOSITORY / "packages" / "abw_maxcover" / "src"
RUN_OUT = Path(
    os.environ.get(
        "TIMOR_GLOBAL2_2026_PIPELINE_OUTPUT",
        str(
            ROOT
            / "runs"
            / "timor_global2_2026_20260717_clean"
            / "east-timor_data"
            / "outputs"
        ),
    )
)
OUT = Path(
    os.environ.get(
        "TIMOR_GLOBAL2_2026_PARETO_OUTPUT",
        str(ROOT / "outputs" / "timor_global2_2026_exact_pareto_20260717"),
    )
)
WEIGHT_SCALE = 1_000

sys.path.insert(0, str(ABW_SRC))

from abw_maxcover import (  # noqa: E402
    GurobiConfig,
    build_instance_from_facility_map,
    exact_pareto_curve,
)
from abw_maxcover._incremental_core import (  # noqa: E402
    budgeted_construct,
    compute_coverage_and_objective,
)
from abw_maxcover.results import MaxCoverResult  # noqa: E402


@dataclass(frozen=True, slots=True)
class Case:
    spacing_m: int
    threshold_m: int

    @property
    def case_id(self) -> str:
        return (
            f"timor_global2_2026_grid_{self.spacing_m // 1000}km_"
            f"threshold_{self.threshold_m // 1000}km"
        )


CASES = [
    *(Case(10_000, threshold) for threshold in (2_000, 5_000, 10_000)),
    *(Case(5_000, threshold) for threshold in (2_000, 5_000, 10_000)),
    *(Case(1_000, threshold) for threshold in (10_000, 5_000, 2_000)),
]


class SaturationReached(RuntimeError):
    pass


class NonOptimalBudget(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def clock(seconds: float) -> str:
    milliseconds = int(round(seconds * 1_000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def package_versions() -> dict[str, str]:
    import gurobipy
    import pandas
    import pyarrow

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "gurobipy": ".".join(str(part) for part in gurobipy.gurobi.version()),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate_manifests() -> dict[int, tuple[Path, dict[str, Any]]]:
    manifests: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(RUN_OUT.glob("run_manifest*.yaml"), key=lambda p: p.stat().st_mtime):
        manifest = load_yaml(path)
        resolved = manifest.get("parameters", {}).get("resolved", {})
        settings = manifest.get("parameters", {}).get("runtime_settings", {})
        if not resolved.get("has_candidates") or settings.get("network_profile") != "driving":
            continue
        spacing = int(float(resolved["candidate_grid_spacing_m"]))
        manifests[spacing] = (path, manifest)
    missing = {case.spacing_m for case in CASES} - manifests.keys()
    if missing:
        raise RuntimeError(f"Missing candidate manifests for spacings {sorted(missing)}")
    return manifests


def normalize_population(population: pd.DataFrame) -> pd.DataFrame:
    population = population.copy()
    if "target_id" not in population.columns:
        ids = population["ID"].astype(str)
        population["target_id"] = ids.where(
            ids.str.startswith("target_population_"),
            "target_population_" + ids,
        )
    return population


def build_case(
    case: Case,
    manifest_path: Path,
    manifest: dict[str, Any],
):
    outputs = manifest["outputs"]
    population_path = Path(outputs["population"]["path"])
    amenity_matrix_path = Path(
        outputs["distance_matrix_src_amenities_dst_population"]["path"]
    )
    candidate_matrix_path = Path(
        outputs["distance_matrix_src_candidates_dst_population"]["path"]
    )
    sources_path = Path(outputs["sources"]["path"])

    population = normalize_population(pd.read_parquet(population_path))
    demand_ids = population["target_id"].astype(str).tolist()
    demand_index = {demand_id: index for index, demand_id in enumerate(demand_ids)}
    weights = np.rint(
        population["population"].to_numpy(dtype=float) * WEIGHT_SCALE
    ).astype(np.int64)

    amenity = pd.read_parquet(
        amenity_matrix_path, columns=["target_id", "total_dist"]
    )
    existing_ids = set(
        amenity.loc[
            amenity["total_dist"] <= case.threshold_m, "target_id"
        ].astype(str)
    )
    existing_covered = {
        demand_index[demand_id]
        for demand_id in existing_ids
        if demand_id in demand_index
    }

    candidates = pd.read_parquet(
        candidate_matrix_path,
        columns=["source_id", "target_id", "total_dist"],
        filters=[("total_dist", "<=", float(case.threshold_m))],
    )
    candidates = candidates[candidates["total_dist"] <= case.threshold_m]
    candidate_ids = sorted(candidates["source_id"].astype(str).unique().tolist())
    facility_index = {
        facility_id: index for index, facility_id in enumerate(candidate_ids)
    }
    facility_to_demand: dict[int, list[int]] = {
        facility_index[facility_id]: [] for facility_id in candidate_ids
    }
    for facility_id, target_id in candidates[
        ["source_id", "target_id"]
    ].itertuples(index=False, name=None):
        demand_i = demand_index.get(str(target_id))
        if demand_i is None or demand_i in existing_covered:
            continue
        facility_to_demand[facility_index[str(facility_id)]].append(int(demand_i))

    instance = build_instance_from_facility_map(
        facility_to_demand,
        weights,
        covered=existing_covered,
        n_facilities=len(candidate_ids),
        assume_unique_sorted=False,
        metadata={
            "case_id": case.case_id,
            "spacing_m": case.spacing_m,
            "service_threshold_m": case.threshold_m,
            "weight_scale": WEIGHT_SCALE,
            "candidate_ids": candidate_ids,
            "manifest_path": str(manifest_path),
        },
        name=case.case_id,
    )
    _, all_objective = compute_coverage_and_objective(
        instance, list(range(instance.n_facilities))
    )
    baseline = int(weights[list(existing_covered)].sum()) if existing_covered else 0
    total = int(weights.sum())

    sources = pd.read_parquet(sources_path)
    source_id_column = "ID" if "ID" in sources.columns else "source_id"
    sources[source_id_column] = sources[source_id_column].astype(str)
    catalog = sources[
        sources[source_id_column].isin(candidate_ids)
    ].copy()
    catalog = catalog.set_index(source_id_column).reindex(candidate_ids).reset_index()
    catalog = catalog.rename(columns={source_id_column: "candidate_id"})
    catalog.insert(0, "candidate_index", range(len(catalog)))

    metadata = {
        "case_id": case.case_id,
        "spacing_m": case.spacing_m,
        "service_threshold_m": case.threshold_m,
        "weight_scale": WEIGHT_SCALE,
        "candidate_count_all_grid": int(
            (sources["source_type"] == "candidates").sum()
        ),
        "candidate_count_active": int(instance.n_facilities),
        "population_points": int(len(population)),
        "total_population": total / WEIGHT_SCALE,
        "total_population_scaled": total,
        "baseline_covered_population": baseline / WEIGHT_SCALE,
        "baseline_covered_population_scaled": baseline,
        "baseline_coverage_pct": 100.0 * baseline / total,
        "all_candidate_incremental_population": int(all_objective) / WEIGHT_SCALE,
        "all_candidate_incremental_population_scaled": int(all_objective),
        "all_candidate_coverage_pct": 100.0 * (baseline + int(all_objective)) / total,
        "n_demand": int(instance.n_demand),
        "n_arcs": int(instance.ji_indices.size),
        "candidate_matrix_rows_all_distances": int(
            pq.ParquetFile(candidate_matrix_path).metadata.num_rows
        ),
        "candidate_matrix_rows_within_threshold": int(len(candidates)),
        "manifest_path": str(manifest_path),
        "population_path": str(population_path),
        "population_sha256": sha256(population_path),
        "candidate_matrix_path": str(candidate_matrix_path),
        "candidate_matrix_sha256": sha256(candidate_matrix_path),
        "amenity_matrix_path": str(amenity_matrix_path),
        "amenity_matrix_sha256": sha256(amenity_matrix_path),
    }
    return instance, candidate_ids, catalog, metadata


def first_saturation_upper_bound(instance, all_objective: int) -> tuple[int, float]:
    started = perf_counter()
    greedy = budgeted_construct(
        instance,
        instance.n_facilities,
        constructor="greedy",
        seed=42,
    )
    elapsed = perf_counter() - started
    for budget, objective in enumerate(greedy.objectives):
        if int(objective) >= int(all_objective):
            return int(budget), elapsed
    raise RuntimeError("Greedy construction did not attain the all-candidate ceiling")


def latest_optimal_records(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if record.get("status") == "optimal":
            latest[int(record["budget"])] = record
    return latest


def write_frontier_csv(case_dir: Path) -> None:
    records = sorted(
        latest_optimal_records(case_dir / "frontier.jsonl").values(),
        key=lambda record: int(record["budget"]),
    )
    if not records:
        return
    fields = sorted({field for record in records for field in record})
    with (case_dir / "frontier.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_case(
    case: Case,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    time_limit_seconds: float,
    mip_gap: float,
    dry_run: bool,
) -> None:
    case_dir = OUT / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    event_path = case_dir / "events.jsonl"
    append_jsonl(
        event_path,
        {"event": "instance_build_started", "timestamp": now()},
    )
    build_started = perf_counter()
    instance, candidate_ids, catalog, metadata = build_case(
        case, manifest_path, manifest
    )
    build_seconds = perf_counter() - build_started
    catalog.to_csv(case_dir / "candidate_catalog.csv", index=False)
    append_jsonl(
        event_path,
        {
            "event": "instance_build_completed",
            "timestamp": now(),
            "elapsed_seconds": build_seconds,
            "elapsed_clock": clock(build_seconds),
        },
    )

    all_objective = int(metadata["all_candidate_incremental_population_scaled"])
    append_jsonl(
        event_path,
        {"event": "greedy_saturation_started", "timestamp": now()},
    )
    saturation_upper, greedy_seconds = first_saturation_upper_bound(
        instance, all_objective
    )
    metadata.update(
        {
            "instance_build_seconds": build_seconds,
            "instance_build_clock": clock(build_seconds),
            "greedy_saturation_upper_budget": saturation_upper,
            "greedy_saturation_seconds": greedy_seconds,
            "greedy_saturation_clock": clock(greedy_seconds),
            "pipeline_git_commit_used_for_matrices": manifest["code"]["git_commit"],
            "abw_maxcover_git_commit": git_commit(REPOSITORY),
            "created_at": now(),
        }
    )
    (case_dir / "instance_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    append_jsonl(
        event_path,
        {
            "event": "greedy_saturation_completed",
            "timestamp": now(),
            "upper_budget": saturation_upper,
            "elapsed_seconds": greedy_seconds,
            "elapsed_clock": clock(greedy_seconds),
        },
    )
    print(
        f"{case.case_id}: candidates={instance.n_facilities:,}, "
        f"arcs={instance.ji_indices.size:,}, baseline={metadata['baseline_coverage_pct']:.3f}%, "
        f"ceiling={metadata['all_candidate_coverage_pct']:.3f}%, "
        f"greedy upper={saturation_upper:,}",
        flush=True,
    )
    if dry_run:
        return

    existing = latest_optimal_records(case_dir / "frontier.jsonl")
    saturated = [
        record
        for record in existing.values()
        if int(record["incremental_objective_scaled"]) >= all_objective
    ]
    if saturated:
        exact_budget = min(int(record["budget"]) for record in saturated)
        print(f"  already complete at exact saturation budget {exact_budget}", flush=True)
        return
    next_budget = max(existing, default=-1) + 1
    budgets = list(range(next_budget, saturation_upper + 1))
    if not budgets:
        raise RuntimeError(
            f"No remaining budgets but saturation was not recorded for {case.case_id}"
        )

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    session_started = perf_counter()
    prior_solve_seconds = sum(
        float(record["solve_seconds"]) for record in existing.values()
    )
    session_solve_seconds = 0.0
    callback_count = 0
    session_model_seconds: float | None = None
    append_jsonl(
        event_path,
        {
            "event": "exact_session_started",
            "timestamp": now(),
            "run_id": run_id,
            "first_budget": budgets[0],
            "last_budget_upper_bound": budgets[-1],
            "time_limit_seconds_per_budget": time_limit_seconds,
            "mip_gap": mip_gap,
        },
    )

    def checkpoint(result: MaxCoverResult) -> None:
        nonlocal callback_count, session_model_seconds, session_solve_seconds
        callback_count += 1
        session_solve_seconds += float(result.solve_seconds)
        if session_model_seconds is None:
            session_model_seconds = float(result.model_seconds)
            append_jsonl(
                event_path,
                {
                    "event": "gurobi_model_built",
                    "timestamp": now(),
                    "run_id": run_id,
                    "model_seconds": session_model_seconds,
                    "model_clock": clock(session_model_seconds),
                },
            )
        objective = 0 if result.objective is None else int(result.objective)
        cumulative_solve = prior_solve_seconds + session_solve_seconds
        total_covered = int(metadata["baseline_covered_population_scaled"]) + objective
        frontier_record = {
            "case_id": case.case_id,
            "spacing_m": case.spacing_m,
            "threshold_m": case.threshold_m,
            "budget": int(result.budget),
            "status": result.status,
            "incremental_objective_scaled": objective,
            "incremental_population": objective / WEIGHT_SCALE,
            "total_covered_population": total_covered / WEIGHT_SCALE,
            "coverage_pct": 100.0
            * total_covered
            / int(metadata["total_population_scaled"]),
            "upper_bound": result.upper_bound,
            "mip_gap": result.mip_gap,
            "selected_count": len(result.solution),
            "model_seconds_this_session": float(result.model_seconds),
            "solve_seconds": float(result.solve_seconds),
            "solve_clock": clock(float(result.solve_seconds)),
            "solver_total_seconds": float(result.total_seconds),
            "session_wall_seconds": perf_counter() - session_started,
            "cumulative_solve_seconds": cumulative_solve,
            "warm_start_source": result.metadata.get("warm_start_source"),
            "warm_start_objective": result.metadata.get("warm_start_objective"),
            "warm_start_selected_count": result.metadata.get(
                "warm_start_selected_count"
            ),
            "recorded_at": now(),
            "run_id": run_id,
        }
        solution_record = {
            "case_id": case.case_id,
            "budget": int(result.budget),
            "status": result.status,
            "objective_scaled": objective,
            "solution_indices": [int(index) for index in result.solution],
            "candidate_ids": [candidate_ids[int(index)] for index in result.solution],
            "recorded_at": frontier_record["recorded_at"],
            "run_id": run_id,
        }
        append_jsonl(case_dir / "frontier.jsonl", frontier_record)
        append_jsonl(case_dir / "solutions.jsonl", solution_record)
        if callback_count == 1 or callback_count % 10 == 0:
            write_frontier_csv(case_dir)
        if int(result.budget) < 10 or int(result.budget) % 10 == 0:
            print(
                f"  p={result.budget:4d}  coverage={frontier_record['coverage_pct']:.6f}%  "
                f"solve={clock(float(result.solve_seconds))}  {result.status}",
                flush=True,
            )
        if result.status != "optimal":
            raise NonOptimalBudget(
                f"{case.case_id} budget {result.budget}: {result.status}"
            )
        if objective >= all_objective:
            append_jsonl(
                event_path,
                {
                    "event": "exact_saturation_reached",
                    "timestamp": now(),
                    "run_id": run_id,
                    "budget": int(result.budget),
                    "objective_scaled": objective,
                },
            )
            raise SaturationReached(
                f"{case.case_id} saturated at budget {result.budget}"
            )

    try:
        exact_pareto_curve(
            instance,
            budgets,
            gurobi_config=GurobiConfig(
                time_limit_seconds=time_limit_seconds,
                mip_gap=mip_gap,
                trace=False,
                warm_start=True,
            ),
            result_callback=checkpoint,
        )
    except SaturationReached as reached:
        print(f"  {reached}", flush=True)
    finally:
        wall_seconds = perf_counter() - session_started
        append_jsonl(
            event_path,
            {
                "event": "exact_session_ended",
                "timestamp": now(),
                "run_id": run_id,
                "wall_seconds": wall_seconds,
                "wall_clock": clock(wall_seconds),
                "budgets_recorded": callback_count,
                "model_seconds": session_model_seconds,
            },
        )
        write_frontier_csv(case_dir)


def write_campaign_manifest(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": now(),
        "purpose": "Exact Gurobipy Pareto frontiers for latest WorldPop Timor-Leste replication",
        "cases": [
            {
                "case_id": case.case_id,
                "spacing_m": case.spacing_m,
                "threshold_m": case.threshold_m,
            }
            for case in CASES
        ],
        "solver": "gurobipy only",
        "time_limit_seconds_per_budget": args.time_limit,
        "mip_gap": args.mip_gap,
        "weight_scale": WEIGHT_SCALE,
        "repository": str(REPOSITORY),
        "repository_git_commit": git_commit(REPOSITORY),
        "pipeline_matrix_commit": "1be52321f1f218f0a11f7faf420ae007fcd67d72",
        "versions": package_versions(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    (OUT / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=3_600.0)
    parser.add_argument("--mip-gap", type=float, default=1e-9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Optional exact case_id; repeat to select multiple cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_campaign_manifest(args)
    manifests = candidate_manifests()
    selected = [case for case in CASES if not args.case or case.case_id in args.case]
    unknown = set(args.case) - {case.case_id for case in CASES}
    if unknown:
        raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
    for case in selected:
        manifest_path, manifest = manifests[case.spacing_m]
        run_case(
            case,
            manifest_path,
            manifest,
            time_limit_seconds=args.time_limit,
            mip_gap=args.mip_gap,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
