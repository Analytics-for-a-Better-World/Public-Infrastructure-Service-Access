"""Run the country pipeline with safe resume for partitioned sparse matrices.

This launcher leaves the upstream pipeline untouched. It monkey-patches only
``write_distances_polars_partitioned`` before executing ``run_pipeline.py``.
The patch is intentionally conservative: it resumes only when existing part
files form an exact contiguous prefix of target chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from time import perf_counter as pc

import polars as pl


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run run_pipeline.py with conservative partition resume."
    )
    parser.add_argument(
        "--pipeline-dir",
        required=True,
        help="Path to Research-Sandbox/general_distances_per_country.",
    )
    return parser.parse_known_args()


def _target_suffix_expr() -> pl.Expr:
    return pl.col("target_id").cast(pl.Utf8).str.extract(r"(\d+)$").cast(pl.Int64)


def _validate_existing_prefix(
    output_dir: Path,
    *,
    target_chunk_size: int,
) -> tuple[int, int, int, pl.DataFrame | None]:
    """Return ``(resume_start, part_count, row_count, preview)``.

    Parts are sorted by their emitted file number. We require monotone,
    non-overlapping target ranges and continue from one plus the largest target
    id suffix already present. This permits increasing the target chunk size
    after a conservative interrupted run.
    """
    part_files = sorted(output_dir.glob("part-*.parquet"))
    if not part_files:
        return 0, 0, 0, None

    row_count = 0
    previous_target_max = -1
    preview: pl.DataFrame | None = None
    for part_index, part_path in enumerate(part_files):
        try:
            stats = (
                pl.scan_parquet(part_path)
                .select(
                    _target_suffix_expr().min().alias("target_min"),
                    _target_suffix_expr().max().alias("target_max"),
                    pl.len().alias("rows"),
                )
                .collect()
                .to_dicts()[0]
            )
        except Exception as exc:  # pragma: no cover - operational guard
            raise RuntimeError(f"Cannot read existing part {part_path}: {exc}") from exc

        target_min = stats["target_min"]
        target_max = stats["target_max"]
        rows = int(stats["rows"])
        if target_min is None or target_max is None or rows <= 0:
            raise RuntimeError(f"Existing part has no usable target rows: {part_path}")
        if target_min <= previous_target_max:
            raise RuntimeError(
                "Existing partitioned matrix is not an exact chunk prefix: "
                f"{part_path.name} overlaps an earlier part with target range "
                f"{target_min}:{target_max + 1}."
            )

        row_count += rows
        previous_target_max = target_max
        if preview is None:
            preview = pl.read_parquet(part_path).head()

    return previous_target_max + 1, len(part_files), row_count, preview


def _install_resume_writer() -> None:
    import distance_pipeline.distance_matrix as dm

    original_writer = dm.write_distances_polars_partitioned

    def adopt_resume_source(output_dir: Path) -> None:
        source_env = os.environ.get("PISA_PARTITION_RESUME_SOURCE_DIR")
        if not source_env or output_dir.exists():
            return
        if "src_candidates_dst_population" not in output_dir.name:
            return

        source_dir = Path(source_env)
        if not source_dir.exists():
            raise FileNotFoundError(
                "PISA_PARTITION_RESUME_SOURCE_DIR does not exist: "
                f"{source_dir}"
            )
        if source_dir.resolve() == output_dir.resolve():
            return

        action = os.environ.get("PISA_PARTITION_RESUME_ADOPT", "copy").lower()
        if action not in {"copy", "move"}:
            raise ValueError("PISA_PARTITION_RESUME_ADOPT must be 'copy' or 'move'")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if action == "move":
            shutil.move(str(source_dir), str(output_dir))
        else:
            shutil.copytree(source_dir, output_dir)

    def write_distances_polars_partitioned_resumable(
        targets,
        sources,
        distance_threshold_largest,
        network,
        output_dir,
        *,
        max_total_dist=None,
        node_pair_cache_dir=None,
        imp_name=None,
        road_cost_col="road_distance",
        total_cost_col="total_dist",
        stitch_cost_factor=1.0,
        target_chunk_size=10_000,
        overwrite=False,
        verbose=False,
    ):
        if overwrite:
            return original_writer(
                targets=targets,
                sources=sources,
                distance_threshold_largest=distance_threshold_largest,
                network=network,
                output_dir=output_dir,
                max_total_dist=max_total_dist,
                node_pair_cache_dir=node_pair_cache_dir,
                imp_name=imp_name,
                road_cost_col=road_cost_col,
                total_cost_col=total_cost_col,
                stitch_cost_factor=stitch_cost_factor,
                target_chunk_size=target_chunk_size,
                overwrite=overwrite,
                verbose=verbose,
            )

        targets, sources = dm._validate_inputs(targets, sources)
        dm._validate_node_columns(targets, sources)

        if stitch_cost_factor <= 0:
            raise ValueError("stitch_cost_factor must be positive")
        if target_chunk_size <= 0:
            raise ValueError("target_chunk_size must be positive")
        if road_cost_col == total_cost_col:
            raise ValueError("road_cost_col and total_cost_col must be different")

        output_dir = Path(output_dir)
        adopt_resume_source(output_dir)
        existing = dm._read_partitioned_distance_summary(output_dir)
        if existing is not None:
            if verbose:
                print(f"using existing partitioned sparse matrix: {output_dir}")
            return existing

        if output_dir.exists():
            resume_start, part_count, row_count, preview = _validate_existing_prefix(
                output_dir,
                target_chunk_size=target_chunk_size,
            )
            if resume_start == 0 and part_count == 0:
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            elif verbose:
                print(
                    "resuming incomplete partitioned sparse matrix "
                    f"{output_dir} from target index {resume_start:,} "
                    f"after {part_count:,} part(s)"
                )
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            resume_start = 0
            part_count = 0
            row_count = 0
            preview = None

        node_pair_cache_path = (
            None if node_pair_cache_dir is None else Path(node_pair_cache_dir)
        )

        empty_chunk_count = 0
        t_total = pc()

        for chunk_start in range(resume_start, len(targets), target_chunk_size):
            chunk_stop = min(chunk_start + target_chunk_size, len(targets))
            target_chunk = targets.iloc[chunk_start:chunk_stop]

            if verbose:
                print(
                    "computing sparse matrix target chunk "
                    f"{chunk_start:,}:{chunk_stop:,} of {len(targets):,}"
                )

            target_indices, source_indices = dm._candidate_row_pairs(
                targets=target_chunk,
                sources=sources,
                distance_threshold_largest=distance_threshold_largest,
                max_total_dist=max_total_dist,
                verbose=verbose,
            )

            distances_pl = dm._compute_unique_node_pair_distances(
                targets=target_chunk,
                sources=sources,
                target_indices=target_indices,
                source_indices=source_indices,
                network=network,
                node_pair_cache_dir=node_pair_cache_path,
                imp_name=imp_name,
                verbose=verbose,
            )

            result = dm._assemble_distance_result(
                distances_pl=distances_pl,
                targets=target_chunk,
                sources=sources,
                max_total_dist=max_total_dist,
                road_cost_col=road_cost_col,
                total_cost_col=total_cost_col,
                stitch_cost_factor=stitch_cost_factor,
                verbose=verbose,
            )

            if result.height == 0:
                empty_chunk_count += 1
                continue

            if preview is None:
                preview = result.head()

            part_path = output_dir / f"part-{part_count:06d}.parquet"
            result.write_parquet(part_path, compression="zstd")
            row_count += result.height
            part_count += 1

            if verbose:
                print(
                    f"wrote {result.height:,} sparse distance rows to "
                    f"{part_path.name}"
                )

        metadata: dict[str, object] = {
            "path": str(output_dir),
            "row_count": row_count,
            "part_count": part_count,
            "empty_chunk_count": empty_chunk_count,
            "target_chunk_size": target_chunk_size,
            "target_count": len(targets),
            "source_count": len(sources),
            "distance_threshold_km": distance_threshold_largest,
            "max_total_dist": max_total_dist,
            "elapsed_seconds": pc() - t_total,
            "resumed_from_target_index": resume_start,
        }

        success_path = output_dir / "_SUCCESS.json"
        with success_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        metadata["path"] = output_dir
        metadata["preview"] = (
            preview
            if preview is not None
            else dm._empty_distance_result(
                targets,
                sources,
                road_cost_col=road_cost_col,
                total_cost_col=total_cost_col,
            )
        )

        if verbose:
            print(
                f"wrote partitioned sparse matrix with {row_count:,} row(s) "
                f"in {part_count:,} part(s) to {output_dir} "
                f"in {metadata['elapsed_seconds']:.2f} seconds"
            )

        return metadata

    dm.write_distances_polars_partitioned = write_distances_polars_partitioned_resumable


def main() -> None:
    args, rest = _parse_args()
    pipeline_dir = Path(args.pipeline_dir).resolve()
    if not (pipeline_dir / "run_pipeline.py").exists():
        raise FileNotFoundError(f"run_pipeline.py not found in {pipeline_dir}")

    sys.path.insert(0, str(pipeline_dir))
    _install_resume_writer()

    script_path = pipeline_dir / "run_pipeline.py"
    sys.argv = [str(script_path), *rest]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
