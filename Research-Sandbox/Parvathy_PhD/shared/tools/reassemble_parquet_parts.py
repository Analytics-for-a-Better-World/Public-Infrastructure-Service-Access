"""Reassemble a `.parquet_parts` directory into one Parquet file.

The script streams record batches through pyarrow and never loads the full
matrix in memory. It validates the output row count against `_SUCCESS.json`
when that manifest is present.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PART_RE = re.compile(r"part-(\d+)\.parquet$")


def _part_sort_key(path: Path) -> tuple[int, str]:
    match = PART_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def _load_manifest(parts_dir: Path) -> dict[str, Any]:
    manifest_path = parts_dir / "_SUCCESS.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _parts(parts_dir: Path, max_parts: int | None) -> list[Path]:
    paths = sorted(parts_dir.glob("*.parquet"), key=_part_sort_key)
    if max_parts is not None:
        paths = paths[: max(0, int(max_parts))]
    return paths


def reassemble(
    parts_dir: Path,
    output_path: Path,
    *,
    batch_size: int,
    compression: str,
    force: bool,
    max_parts: int | None,
) -> dict[str, Any]:
    parts_dir = parts_dir.resolve()
    output_path = output_path.resolve()
    manifest = _load_manifest(parts_dir)
    part_paths = _parts(parts_dir, max_parts)
    if not part_paths:
        raise FileNotFoundError(f"no parquet part files found in {parts_dir}")
    if output_path.exists() and not force:
        raise FileExistsError(f"output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = pq.ParquetFile(part_paths[0])
    schema = first.schema_arrow
    expected_rows = None if max_parts is not None else manifest.get("row_count")
    expected_parts = None if max_parts is not None else manifest.get("part_count")
    if expected_parts is not None and int(expected_parts) != len(part_paths):
        raise RuntimeError(
            f"manifest part_count={expected_parts} but found {len(part_paths)} part files"
        )

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    started = time.perf_counter()
    rows_written = 0
    batches_written = 0
    bytes_read = 0
    try:
        with pq.ParquetWriter(tmp_path, schema=schema, compression=compression) as writer:
            for part_index, part_path in enumerate(part_paths):
                part = pq.ParquetFile(part_path)
                if not part.schema_arrow.equals(schema):
                    raise RuntimeError(f"schema mismatch in {part_path}")
                bytes_read += part_path.stat().st_size
                for batch in part.iter_batches(batch_size=batch_size):
                    writer.write_batch(batch)
                    rows_written += batch.num_rows
                    batches_written += 1
                print(
                    f"[{part_index + 1:04d}/{len(part_paths):04d}] "
                    f"{part_path.name}: rows_written={rows_written:,}",
                    flush=True,
                )
        if expected_rows is not None and int(expected_rows) != rows_written:
            raise RuntimeError(
                f"manifest row_count={expected_rows} but wrote {rows_written}"
            )
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    elapsed = time.perf_counter() - started
    summary = {
        "parts_dir": str(parts_dir),
        "output_path": str(output_path),
        "part_count": len(part_paths),
        "rows_written": rows_written,
        "batches_written": batches_written,
        "batch_size": int(batch_size),
        "compression": compression,
        "input_bytes": bytes_read,
        "output_bytes": output_path.stat().st_size,
        "elapsed_seconds": elapsed,
        "expected_rows": expected_rows,
        "manifest_path": str(parts_dir / "_SUCCESS.json") if manifest else None,
        "limited_test_run": max_parts is not None,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".reassembly.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-parts", type=int, default=None)
    args = parser.parse_args()
    summary = reassemble(
        args.parts_dir,
        args.output_path,
        batch_size=args.batch_size,
        compression=args.compression,
        force=args.force,
        max_parts=args.max_parts,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
