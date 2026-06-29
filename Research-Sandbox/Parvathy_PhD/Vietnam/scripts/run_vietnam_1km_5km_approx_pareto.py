from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ABW_SRC = (
    Path(r"C:\github\Public-Infrastructure-Service-Access")
    / "Research-Sandbox"
    / "abw_maxcover"
    / "src"
)
sys.path.insert(0, str(ABW_SRC))

from abw_maxcover import MaxCoverInstance  # noqa: E402
from abw_maxcover._incremental_core import (  # noqa: E402
    budgeted_construct,
    drop_redundant_facilities,
)
from abw_maxcover.io import save_instance_npz, load_instance_npz  # noqa: E402


RUN_ROOT = ROOT / "runs" / "vietnam_170_agg5_20260624_s20" / "vietnam_data" / "outputs"
OUT_DIR = ROOT / "outputs" / "vietnam_1km_5km_approx_pareto_20260629"
INSTANCE_NPZ = OUT_DIR / "vietnam_170_1km_5km_component01_maxcover_instance.npz"
METADATA_JSON = OUT_DIR / "vietnam_170_1km_5km_component01_instance_metadata.json"

POPULATION_PATH = (
    RUN_ROOT
    / "population_pop_1_sample_1_seed_42_max_none_agg_5_maxdist_20000_amenity_amenity_all-dst_population-s_b7d2ceeff8d2.parquet"
)
CANDIDATE_PARTS = (
    RUN_ROOT
    / "distance_matrix_src_candidates_dst_population_pop_1_sample_1_seed_42_max_none_agg_5_maxdist_20000_a_15fe840aed2e.parquet_parts"
)
EXISTING_PARTS = (
    RUN_ROOT
    / "distance_matrix_src_table_dst_population_pop_1_sample_1_seed_42_max_none_agg_5_maxdist_20000_amenit_edf2ee14a7da.parquet_parts"
)

THRESHOLD_M = 5_000.0
WEIGHT_SCALE = 1_000.0


def clock(seconds: float) -> str:
    millis = int(round(float(seconds) * 1000))
    h, rem = divmod(millis, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric_suffix(values: np.ndarray, prefix: str) -> np.ndarray:
    result = np.empty(values.size, dtype=np.int32)
    n = len(prefix)
    for i, value in enumerate(values):
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if not text.startswith(prefix):
            raise ValueError(f"unexpected id {text!r}; expected prefix {prefix!r}")
        result[i] = int(text[n:])
    return result


def read_population() -> tuple[np.ndarray, np.ndarray, float]:
    table = pq.read_table(POPULATION_PATH, columns=["ID", "population"], use_threads=False)
    ids = table["ID"].to_numpy(zero_copy_only=False)
    raw_ids = numeric_suffix(ids, "target_population_")
    pop = table["population"].to_numpy(zero_copy_only=False).astype(float)
    if raw_ids.size and (raw_ids.min() != 0 or raw_ids.max() != raw_ids.size - 1):
        raise ValueError("population target IDs are not contiguous from zero")
    weights = np.rint(pop * WEIGHT_SCALE).astype(np.int64)
    return raw_ids, weights, float(pop.sum())


def iter_part_files(path: Path) -> list[Path]:
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parquet parts found under {path}")
    success = path / "_SUCCESS.json"
    if not success.exists():
        raise FileNotFoundError(f"missing success marker: {success}")
    return parts


def filter_part(path: Path, columns: list[str]) -> pq.Table:
    table = pq.read_table(path, columns=columns, use_threads=False)
    mask = pc.less_equal(table["total_dist"], THRESHOLD_M)
    return table.filter(mask)


def baseline_mask_from_existing(n_demand: int, weights: np.ndarray) -> tuple[np.ndarray, int, int, float]:
    baseline = np.zeros(n_demand, dtype=bool)
    rows = 0
    start = perf_counter()
    for i, part in enumerate(iter_part_files(EXISTING_PARTS), start=1):
        filtered = filter_part(part, ["target_id", "total_dist"])
        rows += filtered.num_rows
        if filtered.num_rows:
            target_raw = numeric_suffix(filtered["target_id"].to_numpy(zero_copy_only=False), "target_population_")
            baseline[target_raw] = True
        print(f"baseline part {i}: kept {filtered.num_rows:,} rows", flush=True)
    return baseline, int(rows), int(weights[baseline].sum()), float(perf_counter() - start)


def build_instance() -> tuple[MaxCoverInstance, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if INSTANCE_NPZ.exists() and METADATA_JSON.exists():
        instance = load_instance_npz(INSTANCE_NPZ)
        metadata = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
        return instance, metadata

    t0 = perf_counter()
    _, all_weights, total_population = read_population()
    baseline, baseline_rows, baseline_weight, baseline_seconds = baseline_mask_from_existing(
        all_weights.size,
        all_weights,
    )
    effective_positive = (~baseline) & (all_weights > 0)

    target_chunks: list[np.ndarray] = []
    candidate_chunks: list[np.ndarray] = []
    kept_rows = 0
    dropped_baseline_or_zero = 0
    scan_start = perf_counter()
    parts = iter_part_files(CANDIDATE_PARTS)
    for i, part in enumerate(parts, start=1):
        filtered = filter_part(part, ["target_id", "source_id", "total_dist"])
        if filtered.num_rows:
            target_raw = numeric_suffix(filtered["target_id"].to_numpy(zero_copy_only=False), "target_population_")
            candidate_raw = numeric_suffix(filtered["source_id"].to_numpy(zero_copy_only=False), "source_candidates_")
            keep = effective_positive[target_raw]
            dropped_baseline_or_zero += int((~keep).sum())
            target_raw = target_raw[keep]
            candidate_raw = candidate_raw[keep]
            if target_raw.size:
                target_chunks.append(target_raw.astype(np.int32, copy=False))
                candidate_chunks.append(candidate_raw.astype(np.int32, copy=False))
                kept_rows += int(target_raw.size)
        print(
            f"candidate part {i:02d}/{len(parts)}: <=5km {filtered.num_rows:,}, "
            f"kept incremental {kept_rows:,}",
            flush=True,
        )

    if not target_chunks:
        raise RuntimeError("no candidate-demand rows survived the 5 km threshold and baseline filter")

    target_raw_all = np.concatenate(target_chunks).astype(np.int32, copy=False)
    candidate_raw_all = np.concatenate(candidate_chunks).astype(np.int32, copy=False)
    del target_chunks, candidate_chunks

    map_start = perf_counter()
    unique_targets, target_compact = np.unique(target_raw_all, return_inverse=True)
    unique_candidates, candidate_compact = np.unique(candidate_raw_all, return_inverse=True)
    target_compact = target_compact.astype(np.int32, copy=False)
    candidate_compact = candidate_compact.astype(np.int32, copy=False)
    del target_raw_all, candidate_raw_all

    n_demand = int(unique_targets.size)
    n_facilities = int(unique_candidates.size)
    pair_key = target_compact.astype(np.int64) * np.int64(n_facilities) + candidate_compact.astype(np.int64)
    del target_compact, candidate_compact
    pair_key = np.unique(pair_key)
    demand_ids = (pair_key // np.int64(n_facilities)).astype(np.int32)
    facility_ids = (pair_key % np.int64(n_facilities)).astype(np.int32)
    del pair_key

    ij_counts = np.bincount(demand_ids, minlength=n_demand).astype(np.int64)
    ij_indptr = np.empty(n_demand + 1, dtype=np.int64)
    ij_indptr[0] = 0
    np.cumsum(ij_counts, out=ij_indptr[1:])
    ij_indices = facility_ids.astype(np.int32, copy=True)

    order = np.argsort(facility_ids, kind="stable")
    sorted_facilities = facility_ids[order]
    ji_indices = demand_ids[order].astype(np.int32, copy=False)
    ji_counts = np.bincount(sorted_facilities, minlength=n_facilities).astype(np.int64)
    ji_indptr = np.empty(n_facilities + 1, dtype=np.int64)
    ji_indptr[0] = 0
    np.cumsum(ji_counts, out=ji_indptr[1:])
    del order, sorted_facilities, demand_ids, facility_ids

    weights = all_weights[unique_targets]
    instance = MaxCoverInstance(
        weights=weights,
        ij_indptr=ij_indptr.astype(np.int32),
        ij_indices=ij_indices.astype(np.int32, copy=False),
        ji_indptr=ji_indptr.astype(np.int32),
        ji_indices=ji_indices.astype(np.int32, copy=False),
        name="vietnam_170_1km_5km_component01",
        metadata={
            "threshold_m": THRESHOLD_M,
            "weight_scale": WEIGHT_SCALE,
            "candidate_raw_ids": unique_candidates.tolist(),
            "target_raw_ids": unique_targets.tolist(),
        },
    )

    metadata = {
        "instance_name": instance.name,
        "threshold_m": THRESHOLD_M,
        "threshold_km": THRESHOLD_M / 1000.0,
        "weight_scale": WEIGHT_SCALE,
        "component_policy": "snap_components_0_1",
        "candidate_grid_spacing_m": 1000,
        "network_profile": "driving",
        "simplify_network": False,
        "aggregate_factor": 5,
        "total_population": total_population,
        "total_weight": int(all_weights.sum()),
        "baseline_weight": int(baseline_weight),
        "baseline_population": baseline_weight / WEIGHT_SCALE,
        "baseline_percent": 100.0 * baseline_weight / int(all_weights.sum()),
        "baseline_rows_within_threshold": int(baseline_rows),
        "baseline_seconds": baseline_seconds,
        "candidate_rows_after_threshold_and_baseline": int(kept_rows),
        "candidate_rows_dropped_baseline_or_zero": int(dropped_baseline_or_zero),
        "n_coverable_incremental_demand": instance.n_demand,
        "n_candidate_sources_with_incremental_coverage": instance.n_facilities,
        "n_arcs": int(instance.ji_indices.size),
        "all_candidate_incremental_population": instance.total_weight / WEIGHT_SCALE,
        "all_candidate_covered_population": (baseline_weight + instance.total_weight) / WEIGHT_SCALE,
        "all_candidate_coverage_percent": 100.0
        * (baseline_weight + instance.total_weight)
        / int(all_weights.sum()),
        "candidate_scan_seconds": float(perf_counter() - scan_start),
        "mapping_seconds": float(perf_counter() - map_start),
        "build_seconds": float(perf_counter() - t0),
    }
    save_instance_npz(instance, INSTANCE_NPZ)
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return instance, metadata


def curve_rows(
    *,
    stage: str,
    objectives: list[int],
    times: list[float],
    metadata: dict,
) -> list[dict]:
    rows = []
    total_weight = float(metadata["total_weight"])
    baseline_weight = float(metadata["baseline_weight"])
    for budget, objective in enumerate(objectives):
        covered_weight = baseline_weight + float(objective)
        elapsed = float(times[budget]) if budget < len(times) else math.nan
        rows.append(
            {
                "stage": stage,
                "budget": int(budget),
                "incremental_population": float(objective) / WEIGHT_SCALE,
                "covered_population": covered_weight / WEIGHT_SCALE,
                "coverage_percent_total_population": 100.0 * covered_weight / total_weight,
                "seconds_from_stage_start": elapsed if math.isfinite(elapsed) else "",
                "clock_from_stage_start": clock(elapsed) if math.isfinite(elapsed) else "",
            }
        )
    return rows


def pointwise_max_objectives(*objective_lists: list[int]) -> list[int]:
    max_len = max(len(values) for values in objective_lists)
    padded = []
    for values in objective_lists:
        if not values:
            raise ValueError("objective curves must not be empty")
        padded.append(list(values) + [values[-1]] * (max_len - len(values)))
    return [max(values[budget] for values in padded) for budget in range(max_len)]


def first_saturation_budget(objectives: list[int], saturation_objective: int) -> int:
    for budget, objective in enumerate(objectives):
        if objective >= saturation_objective:
            return int(budget)
    return int(len(objectives) - 1)


def run_curve() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    instance, metadata = build_instance()

    print(
        f"Instance {instance.name}: {instance.n_facilities:,} candidates, "
        f"{instance.n_demand:,} demand, {instance.ji_indices.size:,} arcs",
        flush=True,
    )

    stage_rows: list[dict] = []
    t = perf_counter()
    greedy = budgeted_construct(instance, instance.n_facilities, constructor="greedy", seed=42)
    greedy_wall = perf_counter() - t
    stage_rows.append(
        {
            "stage": "greedy_to_saturation",
            "selected_count": len(greedy.solution),
            "objective": int(greedy.objective),
            "incremental_population": greedy.objective / WEIGHT_SCALE,
            "covered_population": (metadata["baseline_weight"] + greedy.objective) / WEIGHT_SCALE,
            "coverage_percent_total_population": 100.0
            * (metadata["baseline_weight"] + greedy.objective)
            / metadata["total_weight"],
            "seconds": greedy_wall,
            "clock": clock(greedy_wall),
        }
    )
    print(f"greedy saturated at p={len(greedy.solution):,} in {clock(greedy_wall)}", flush=True)

    t = perf_counter()
    compact = drop_redundant_facilities(
        instance,
        greedy.solution,
        coverage=greedy.coverage,
        objective=greedy.objective,
    )
    compact_wall = perf_counter() - t
    stage_rows.append(
        {
            "stage": "drop_redundant_zero_loss",
            "selected_count": len(compact.solution),
            "objective": int(compact.objective),
            "incremental_population": compact.objective / WEIGHT_SCALE,
            "covered_population": (metadata["baseline_weight"] + compact.objective) / WEIGHT_SCALE,
            "coverage_percent_total_population": 100.0
            * (metadata["baseline_weight"] + compact.objective)
            / metadata["total_weight"],
            "seconds": compact_wall,
            "clock": clock(compact_wall),
        }
    )
    print(
        f"drop redundant kept p={len(compact.solution):,} "
        f"(removed {len(greedy.solution) - len(compact.solution):,}) in {clock(compact_wall)}",
        flush=True,
    )

    t = perf_counter()
    regreedy = budgeted_construct(
        instance,
        len(compact.solution),
        constructor="greedy",
        candidate_pool=compact.solution,
        seed=42,
    )
    regreedy_wall = perf_counter() - t
    stage_rows.append(
        {
            "stage": "regreedy_on_compacted_candidate_set",
            "selected_count": len(regreedy.solution),
            "objective": int(regreedy.objective),
            "incremental_population": regreedy.objective / WEIGHT_SCALE,
            "covered_population": (metadata["baseline_weight"] + regreedy.objective) / WEIGHT_SCALE,
            "coverage_percent_total_population": 100.0
            * (metadata["baseline_weight"] + regreedy.objective)
            / metadata["total_weight"],
            "seconds": regreedy_wall,
            "clock": clock(regreedy_wall),
        }
    )
    print(f"regreedy saturated at p={len(regreedy.solution):,} in {clock(regreedy_wall)}", flush=True)

    envelope_objectives = pointwise_max_objectives(greedy.objectives, regreedy.objectives)
    envelope_saturation_budget = first_saturation_budget(
        envelope_objectives,
        int(instance.total_weight),
    )

    metadata = {
        **metadata,
        "greedy_selected_count": len(greedy.solution),
        "compact_selected_count": len(compact.solution),
        "regreedy_selected_count": len(regreedy.solution),
        "pointwise_max_selected_count_at_saturation": envelope_saturation_budget,
        "pointwise_max_objective_at_saturation": int(envelope_objectives[envelope_saturation_budget]),
        "greedy_seconds": greedy_wall,
        "compact_seconds": compact_wall,
        "regreedy_seconds": regreedy_wall,
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_csv(OUT_DIR / "vietnam_1km_5km_approx_pareto_stage_times.csv", stage_rows)
    rows = []
    rows.extend(curve_rows(stage="greedy", objectives=greedy.objectives, times=greedy.times, metadata=metadata))
    rows.extend(curve_rows(stage="regreedy_restricted", objectives=regreedy.objectives, times=regreedy.times, metadata=metadata))
    rows.extend(
        curve_rows(
            stage="pointwise_max",
            objectives=envelope_objectives,
            times=[math.nan] * len(envelope_objectives),
            metadata=metadata,
        )
    )
    write_csv(OUT_DIR / "vietnam_1km_5km_approx_pareto_curve.csv", rows)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    greedy_x = np.arange(len(greedy.objectives))
    greedy_y = [
        100.0 * (metadata["baseline_weight"] + objective) / metadata["total_weight"]
        for objective in greedy.objectives
    ]
    regreedy_x = np.arange(len(regreedy.objectives))
    regreedy_y = [
        100.0 * (metadata["baseline_weight"] + objective) / metadata["total_weight"]
        for objective in regreedy.objectives
    ]
    envelope_x = np.arange(len(envelope_objectives))
    envelope_y = [
        100.0 * (metadata["baseline_weight"] + objective) / metadata["total_weight"]
        for objective in envelope_objectives
    ]
    ax.plot(greedy_x, greedy_y, color="#1f77b4", linewidth=1.4, alpha=0.55, label="Greedy")
    ax.plot(
        regreedy_x,
        regreedy_y,
        color="#d62728",
        linewidth=1.4,
        alpha=0.55,
        label="Regreedy after zero-loss drop",
    )
    ax.plot(
        envelope_x,
        envelope_y,
        color="#111111",
        linewidth=2.4,
        label="Approximate Pareto: pointwise maximum",
    )
    ax.scatter(
        [envelope_saturation_budget],
        [
            100.0
            * (metadata["baseline_weight"] + envelope_objectives[envelope_saturation_budget])
            / metadata["total_weight"]
        ],
        color="#2ca02c",
        s=46,
        zorder=4,
        label=f"Saturation at p={envelope_saturation_budget:,}",
    )
    ax.axhline(metadata["all_candidate_coverage_percent"], color="0.55", linewidth=1.0, linestyle="--")
    ax.set_xlabel("New candidate sites selected (p)")
    ax.set_ylabel("Population covered within 5 km (%)")
    ax.set_title("Vietnam 1 km candidate grid, road-driving distance, 5 km threshold")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_xlim(0, len(envelope_objectives) * 1.03)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "vietnam_1km_5km_approx_pareto_curve.png", dpi=220)
    fig.savefig(OUT_DIR / "vietnam_1km_5km_approx_pareto_curve.pdf")
    plt.close(fig)

    print(f"wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    run_curve()
