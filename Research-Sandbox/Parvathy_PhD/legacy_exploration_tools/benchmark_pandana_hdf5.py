"""Benchmark Pandana HDF5 persistence on a cached pipeline network."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl


def clock_ms(seconds: float) -> str:
    millis = int(round(seconds * 1000.0))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}.{secs:02d}.{ms:03d}"


def timed(label: str, fn):
    start = perf_counter()
    value = fn()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.3f}s", flush=True)
    return value, elapsed


def shortest_path_lengths(network, target_nodes: np.ndarray, source_nodes: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(
            network.shortest_path_lengths(
                target_nodes,
                source_nodes,
                imp_name="length",
            ),
            dtype=np.float64,
        )
    except Exception:
        return np.asarray(
            network.shortest_path_lengths(target_nodes, source_nodes),
            dtype=np.float64,
        )


def load_sample_pairs(cache_dir: Path, sample_size: int, seed: int) -> pd.DataFrame:
    rows: list[pl.DataFrame] = []
    row_count = 0
    for path in sorted(cache_dir.glob("bucket=*/*.parquet")):
        frame = pl.read_parquet(path).select(
            "target_nearest_node",
            "source_nearest_node",
            "road_distance",
        )
        rows.append(frame)
        row_count += frame.height
        if row_count >= sample_size * 5:
            break
    if not rows:
        raise FileNotFoundError(f"No node-pair cache parquet files under {cache_dir}")
    pairs = pl.concat(rows, how="vertical").unique(
        subset=["target_nearest_node", "source_nearest_node"],
        keep="first",
    )
    if pairs.height > sample_size:
        pairs = pairs.sample(n=sample_size, seed=seed, shuffle=True)
    return pairs.to_pandas()


def summarize_differences(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return {
            "finite_pairs": 0,
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
        }
    diff = np.abs(a[finite] - b[finite])
    return {
        "finite_pairs": int(finite.sum()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--nodes-pkl", type=Path, required=True)
    parser.add_argument("--edges-pkl", type=Path, required=True)
    parser.add_argument("--node-pair-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-cols", nargs="+", default=["length"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_tag = "_".join(str(col) for col in args.weight_cols)
    hdf5_path = args.output_dir / f"vietnam_unsimplified_driving_{weight_tag}_pandana.h5"
    summary_path = (
        args.output_dir
        / f"vietnam_unsimplified_driving_{weight_tag}_pandana_hdf5_benchmark.json"
    )

    import sys

    sys.path.insert(0, str(args.pipeline_dir.resolve()))
    from distance_pipeline.network import build_pandana_network
    import pandana

    nodes, nodes_load_seconds = timed("load nodes pickle", lambda: pd.read_pickle(args.nodes_pkl))
    edges, edges_load_seconds = timed("load edges pickle", lambda: pd.read_pickle(args.edges_pkl))
    pairs, sample_load_seconds = timed(
        "load sample cached node pairs",
        lambda: load_sample_pairs(args.node_pair_cache_dir, args.sample_size, args.seed),
    )

    target_nodes = pairs["target_nearest_node"].to_numpy(dtype=np.int64)
    source_nodes = pairs["source_nearest_node"].to_numpy(dtype=np.int64)
    cached_distances = pairs["road_distance"].to_numpy(dtype=np.float64)

    fresh_network, fresh_build_seconds = timed(
        "build fresh Pandana network",
        lambda: build_pandana_network(
            nodes=nodes,
            edges=edges,
            weight_cols=tuple(args.weight_cols),
        ),
    )
    fresh_distances, fresh_query_seconds = timed(
        f"query fresh network ({len(pairs):,} pairs)",
        lambda: shortest_path_lengths(fresh_network, target_nodes, source_nodes),
    )

    if args.overwrite and hdf5_path.exists():
        hdf5_path.unlink()
    if not hdf5_path.exists():
        _, hdf5_save_seconds = timed(
            "save Pandana network to HDF5",
            lambda: fresh_network.save_hdf5(str(hdf5_path)),
        )
    else:
        hdf5_save_seconds = None
        print(f"using existing HDF5: {hdf5_path}", flush=True)

    del fresh_network
    gc.collect()

    hdf5_network, hdf5_load_seconds = timed(
        "load Pandana network from HDF5",
        lambda: pandana.Network.from_hdf5(str(hdf5_path)),
    )
    hdf5_distances, hdf5_query_seconds = timed(
        f"query HDF5-loaded network ({len(pairs):,} pairs)",
        lambda: shortest_path_lengths(hdf5_network, target_nodes, source_nodes),
    )

    summary = {
        "nodes_pickle": str(args.nodes_pkl),
        "edges_pickle": str(args.edges_pkl),
        "node_pair_cache_dir": str(args.node_pair_cache_dir),
        "hdf5_path": str(hdf5_path),
        "hdf5_size_bytes": hdf5_path.stat().st_size,
        "hdf5_size_gib": hdf5_path.stat().st_size / (1024**3),
        "pandana_version": getattr(pandana, "__version__", "unknown"),
        "weight_cols": list(args.weight_cols),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "sample_size": int(len(pairs)),
        "timings_seconds": {
            "nodes_pickle_load": nodes_load_seconds,
            "edges_pickle_load": edges_load_seconds,
            "sample_pair_load": sample_load_seconds,
            "fresh_build": fresh_build_seconds,
            "fresh_query": fresh_query_seconds,
            "hdf5_save": hdf5_save_seconds,
            "hdf5_load": hdf5_load_seconds,
            "hdf5_query": hdf5_query_seconds,
        },
        "timings_clock_ms": {
            key: None if value is None else clock_ms(float(value))
            for key, value in {
                "nodes_pickle_load": nodes_load_seconds,
                "edges_pickle_load": edges_load_seconds,
                "sample_pair_load": sample_load_seconds,
                "fresh_build": fresh_build_seconds,
                "fresh_query": fresh_query_seconds,
                "hdf5_save": hdf5_save_seconds,
                "hdf5_load": hdf5_load_seconds,
                "hdf5_query": hdf5_query_seconds,
            }.items()
        },
        "differences": {
            "fresh_vs_cached": summarize_differences(fresh_distances, cached_distances),
            "hdf5_vs_cached": summarize_differences(hdf5_distances, cached_distances),
            "hdf5_vs_fresh": summarize_differences(hdf5_distances, fresh_distances),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
