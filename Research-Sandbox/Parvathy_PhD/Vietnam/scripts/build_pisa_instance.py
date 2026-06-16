from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

APPROX_SRC = Path(__file__).resolve().parents[3] / "approximated_tradeoff" / "src"
sys.path.insert(0, str(APPROX_SRC))

import mc_heuristics as mch  # noqa: E402


def find_one(outputs_dir: Path, pattern: str) -> Path:
    matches = sorted(outputs_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def read_parquet_columns(path: Path, wanted: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=wanted)
    except Exception:
        df = pd.read_parquet(path)
        return df[[col for col in wanted if col in df.columns]]


def load_long_distance_matrices(outputs_dir: Path, marker: str, source_type_by_id: dict[str, str]) -> pd.DataFrame:
    split = sorted(outputs_dir.glob(f"distance_matrix_src_*_dst_population_*{marker}*.parquet"))
    if split:
        paths = split
    else:
        paths = [
            path for path in sorted(outputs_dir.glob(f"distance_matrix_*{marker}*.parquet"))
            if "dense" not in path.name.lower()
        ]
    if not paths:
        raise FileNotFoundError(
            f"No sparse long-form distance matrix parquet found with marker {marker!r} in {outputs_dir}. "
            "Run the PISA pipeline with --matrix-output-mode split --matrix-shape sparse."
        )

    frames: list[pd.DataFrame] = []
    wanted = ["source_id", "target_id", "total_dist", "source_type", "target_type"]
    for path in paths:
        df = read_parquet_columns(path, wanted)
        missing = {"source_id", "target_id", "total_dist"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is not a supported sparse matrix; missing {sorted(missing)}")
        if "target_type" in df.columns:
            df = df.loc[df["target_type"].astype(str) == "population"].copy()
        if "source_type" not in df.columns:
            df["source_type"] = df["source_id"].astype(str).map(source_type_by_id)
        frames.append(df[["source_id", "target_id", "total_dist", "source_type"]])
    return pd.concat(frames, ignore_index=True)


def parse_list(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fresh Vietnam max-cover CSR instance from PISA parquets.")
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--run-tag-marker", required=True)
    parser.add_argument("--threshold-m", type=float, required=True)
    parser.add_argument("--existing-source-types", default="table,existing")
    parser.add_argument("--candidate-source-types", default="candidates,candidate")
    parser.add_argument("--weight-scale", type=float, default=1000.0)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--validate-consistency", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_npz is None and args.output_prefix is None:
        raise ValueError("Provide --output-npz or --output-prefix")
    output_npz = args.output_npz or args.output_prefix.with_suffix(".npz")
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    population_path = find_one(args.outputs_dir, f"population_*{args.run_tag_marker}*.parquet")
    sources_path = find_one(args.outputs_dir, f"sources_*{args.run_tag_marker}*.parquet")
    population = pd.read_parquet(population_path).reset_index(drop=True)
    sources = pd.read_parquet(sources_path).reset_index(drop=True)

    required_pop_cols = {"ID", "population"}
    if not required_pop_cols.issubset(population.columns):
        raise ValueError(f"Population parquet missing {sorted(required_pop_cols - set(population.columns))}")
    if not {"ID", "source_type"}.issubset(sources.columns):
        raise ValueError("Sources parquet must contain ID and source_type")

    population_ids = population["ID"].astype(str).to_numpy()
    pop_id_to_idx = {pid: idx for idx, pid in enumerate(population_ids)}
    raw_population = population["population"].to_numpy(dtype=float)
    full_weights = np.rint(raw_population * float(args.weight_scale)).astype(np.int64)

    source_type_by_id = dict(zip(sources["ID"].astype(str), sources["source_type"].astype(str)))
    matrix = load_long_distance_matrices(args.outputs_dir, args.run_tag_marker, source_type_by_id)
    matrix["source_id"] = matrix["source_id"].astype(str)
    matrix["target_id"] = matrix["target_id"].astype(str)
    matrix["source_type"] = matrix["source_type"].astype(str)
    retained = matrix.loc[np.isfinite(matrix["total_dist"]) & (matrix["total_dist"] <= args.threshold_m)].copy()

    existing_types = parse_list(args.existing_source_types)
    candidate_types = parse_list(args.candidate_source_types)

    existing_targets = retained.loc[retained["source_type"].isin(existing_types), "target_id"].unique()
    baseline_mask = np.zeros(len(population_ids), dtype=bool)
    for target_id in existing_targets:
        idx = pop_id_to_idx.get(str(target_id))
        if idx is not None:
            baseline_mask[idx] = True

    effective_weights = full_weights.copy()
    effective_weights[baseline_mask] = 0

    cand = retained.loc[retained["source_type"].isin(candidate_types), ["source_id", "target_id"]].copy()
    candidate_ids = sorted(cand["source_id"].dropna().astype(str).unique().tolist())
    candidate_id_to_j = {source_id: j for j, source_id in enumerate(candidate_ids)}

    ji_lists: list[np.ndarray] = []
    ij_lists: list[list[int]] = [[] for _ in range(len(population_ids))]

    grouped = cand.groupby("source_id", sort=False)
    candidate_groups = {str(source_id): group for source_id, group in grouped}
    for source_id in candidate_ids:
        group = candidate_groups.get(source_id)
        if group is None:
            arr = np.empty(0, dtype=np.int32)
        else:
            pop_indices = sorted({
                pop_id_to_idx[str(target_id)]
                for target_id in group["target_id"]
                if str(target_id) in pop_id_to_idx
            })
            arr = np.asarray(pop_indices, dtype=np.int32)
        ji_lists.append(arr)
        j = candidate_id_to_j[source_id]
        for pop_idx in arr:
            ij_lists[int(pop_idx)].append(j)

    ij_arrays = [np.asarray(sorted(vals), dtype=np.int32) for vals in ij_lists]
    instance = mch.build_instance(
        effective_weights,
        ij_arrays,
        ji_lists,
        assume_unique_sorted=True,
        validate_consistency=args.validate_consistency,
    )

    candidate_sources = sources.loc[sources["ID"].astype(str).isin(candidate_ids)].copy()
    candidate_sources = candidate_sources.set_index(candidate_sources["ID"].astype(str))
    candidate_lon = np.full(len(candidate_ids), np.nan)
    candidate_lat = np.full(len(candidate_ids), np.nan)
    if "Longitude" in candidate_sources.columns:
        for j, source_id in enumerate(candidate_ids):
            if source_id in candidate_sources.index:
                candidate_lon[j] = float(candidate_sources.at[source_id, "Longitude"])
    if "Latitude" in candidate_sources.columns:
        for j, source_id in enumerate(candidate_ids):
            if source_id in candidate_sources.index:
                candidate_lat[j] = float(candidate_sources.at[source_id, "Latitude"])

    metadata = {
        "schema": "pisa_maxcover_csr_v1",
        "outputs_dir": str(args.outputs_dir),
        "run_tag_marker": args.run_tag_marker,
        "population_path": str(population_path),
        "sources_path": str(sources_path),
        "threshold_m": float(args.threshold_m),
        "weight_scale": float(args.weight_scale),
        "existing_source_types": sorted(existing_types),
        "candidate_source_types": sorted(candidate_types),
        "n_population": int(instance.n_households),
        "n_candidates": int(instance.n_facilities),
        "distance_rows_loaded": int(len(matrix)),
        "distance_rows_retained": int(len(retained)),
        "candidate_distance_rows_retained": int(len(cand)),
        "baseline_covered_points": int(baseline_mask.sum()),
        "baseline_covered_population": float(raw_population[baseline_mask].sum()),
        "total_population": float(raw_population.sum()),
        "incremental_weight_units": int(instance.w.sum()),
        "output_npz": str(output_npz),
    }

    np.savez(
        output_npz,
        w=instance.w,
        ij_indptr=instance.ij_indptr,
        ij_indices=instance.ij_indices,
        ji_indptr=instance.ji_indptr,
        ji_indices=instance.ji_indices,
        raw_population=raw_population,
        full_weights=full_weights,
        baseline_covered_mask=baseline_mask,
        population_ids=population_ids.astype("U"),
        candidate_source_ids=np.asarray(candidate_ids, dtype="U"),
        candidate_longitude=candidate_lon,
        candidate_latitude=candidate_lat,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    metadata_path = output_npz.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
