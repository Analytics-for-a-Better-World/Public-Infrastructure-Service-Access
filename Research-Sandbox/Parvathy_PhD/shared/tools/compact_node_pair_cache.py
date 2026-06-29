from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl


KEY_COLUMNS = ["target_nearest_node", "source_nearest_node"]
VALUE_COLUMNS = [*KEY_COLUMNS, "road_distance"]


def compact_bucket(bucket_dir: Path, *, dry_run: bool) -> dict[str, object]:
    files = sorted(bucket_dir.glob("node_pairs_*.parquet"))
    if len(files) <= 1:
        return {
            "bucket": bucket_dir.name,
            "input_files": len(files),
            "output_rows": None,
            "removed_files": 0,
            "skipped": True,
        }

    if dry_run:
        return {
            "bucket": bucket_dir.name,
            "input_files": len(files),
            "output_rows": None,
            "removed_files": 0,
            "skipped": False,
        }

    t0 = time.perf_counter()
    compacted = (
        pl.scan_parquet([str(path) for path in files])
        .select(VALUE_COLUMNS)
        .unique(subset=KEY_COLUMNS, keep="last")
        .collect()
    )

    token = time.time_ns()
    tmp_path = bucket_dir / f"node_pairs_compacted_{token}.tmp.parquet"
    final_path = bucket_dir / f"node_pairs_compacted_{token}_{compacted.height}.parquet"
    compacted.write_parquet(tmp_path)
    tmp_path.replace(final_path)

    bucket_root = bucket_dir.resolve()
    removed = 0
    for path in files:
        resolved = path.resolve()
        if not str(resolved).lower().startswith(str(bucket_root).lower()):
            raise RuntimeError(f"Refusing to delete outside bucket: {resolved}")
        path.unlink()
        removed += 1

    return {
        "bucket": bucket_dir.name,
        "input_files": len(files),
        "output_rows": compacted.height,
        "removed_files": removed,
        "output_file": final_path.name,
        "elapsed_seconds": time.perf_counter() - t0,
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    if not cache_dir.exists():
        raise SystemExit(f"Cache directory does not exist: {cache_dir}")

    bucket_dirs = sorted(
        path for path in cache_dir.glob("bucket=*") if path.is_dir()
    )
    total_before = 0
    total_removed = 0
    total_rows = 0
    t0 = time.perf_counter()

    for bucket_dir in bucket_dirs:
        result = compact_bucket(bucket_dir, dry_run=args.dry_run)
        total_before += int(result["input_files"])
        total_removed += int(result["removed_files"])
        if result["output_rows"] is not None:
            total_rows += int(result["output_rows"])
        if result["input_files"] > 1:
            print(
                f"{result['bucket']}: "
                f"{result['input_files']} files -> "
                f"{result.get('output_rows', 'dry-run')} rows; "
                f"removed {result['removed_files']} "
                f"in {result.get('elapsed_seconds', 0):.3f}s"
            )

    total_after = sum(1 for _ in cache_dir.glob("bucket=*/node_pairs_*.parquet"))
    print(
        "summary: "
        f"buckets={len(bucket_dirs)} "
        f"files_before={total_before} "
        f"files_after={total_after} "
        f"removed={total_removed} "
        f"rows_written={total_rows} "
        f"elapsed_seconds={time.perf_counter() - t0:.3f} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
