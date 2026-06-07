from __future__ import annotations

from pathlib import Path

import pandas as pd

from scalable_distances.matrix.split import MatrixOutputMode, MatrixOutputSet, safe_layer_name, split_matrix_table


def dense_total_matrix(
    matrix: pd.DataFrame,
    *,
    sources: pd.DataFrame,
    targets: pd.DataFrame,
    value_col: str = "total_dist",
) -> pd.DataFrame:
    """Convert a sparse long matrix to target-by-source dense form."""
    dense = matrix.pivot_table(
        index="target_id",
        columns="source_id",
        values=value_col,
        aggfunc="min",
    )
    return dense.reindex(
        index=targets["target_id"].astype(str).tolist(),
        columns=sources["source_id"].astype(str).tolist(),
    )


def _write_dense(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)


def write_dense_matrix_outputs(
    matrix: pd.DataFrame,
    *,
    sources: pd.DataFrame,
    targets: pd.DataFrame,
    output_dir: Path,
    run_tag: str,
    mode: MatrixOutputMode,
) -> MatrixOutputSet:
    """Write dense total matrices, either combined, split by layer, or both."""
    if mode not in {"combined", "split", "both"}:
        raise ValueError(f"Unsupported matrix output mode: {mode}")
    paths: dict[str, Path] = {}

    if mode in {"combined", "both"}:
        path = output_dir / f"distance_matrix_dense_total_{run_tag}.parquet"
        _write_dense(dense_total_matrix(matrix, sources=sources, targets=targets), path)
        paths["dense_matrix_total"] = path

    if mode in {"split", "both"}:
        for part in split_matrix_table(matrix):
            source_subset = sources[sources["source_type"].astype(str) == part.source_type]
            target_subset = targets[targets["target_type"].astype(str) == part.target_type]
            path = output_dir / (
                "dense_matrix_total_src_"
                f"{safe_layer_name(part.source_type)}_dst_{safe_layer_name(part.target_type)}_{run_tag}.parquet"
            )
            _write_dense(
                dense_total_matrix(part.table, sources=source_subset, targets=target_subset),
                path,
            )
            paths[
                f"dense_matrix_total_src_{safe_layer_name(part.source_type)}_"
                f"dst_{safe_layer_name(part.target_type)}"
            ] = path

    return MatrixOutputSet(mode=mode, paths=paths)
