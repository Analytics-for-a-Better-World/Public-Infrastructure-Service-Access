from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

MatrixOutputMode = Literal["combined", "split", "both"]


@dataclass(frozen=True)
class MatrixPart:
    """One source-layer to target-layer matrix partition."""

    source_type: str
    target_type: str
    table: Any

    @property
    def key(self) -> str:
        return split_matrix_output_key(self.source_type, self.target_type)


@dataclass(frozen=True)
class MatrixOutputSet:
    """Manifest-ready output paths written for one matrix output request."""

    mode: MatrixOutputMode
    paths: dict[str, Path]


def safe_layer_name(value: object) -> str:
    """Return a filename-safe layer label matching the original pipeline."""
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "unknown"


def split_matrix_output_key(source_type: object, target_type: object) -> str:
    """Return the manifest key for a split matrix output."""
    return (
        f"distance_matrix_src_{safe_layer_name(source_type)}_"
        f"dst_{safe_layer_name(target_type)}"
    )


def split_matrix_path(
    output_dir: Path,
    run_tag: str,
    source_type: object,
    target_type: object,
) -> Path:
    """Return the output parquet path for one source/destination pair."""
    return output_dir / f"{split_matrix_output_key(source_type, target_type)}_{run_tag}.parquet"


def _is_polars(table: Any) -> bool:
    return table.__class__.__module__.split(".", 1)[0] == "polars"


def _require_layer_columns(table: Any) -> None:
    columns = set(table.columns)
    missing = {"source_type", "target_type"} - columns
    if missing:
        raise ValueError(
            "Split matrix output requires source_type and target_type columns; "
            f"missing {sorted(missing)}"
        )


def _write_table(table: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_polars(table):
        table.write_parquet(path)
    else:
        table.to_parquet(path, index=False)


def _concat_tables(parts: list[Any]) -> Any:
    if not parts:
        raise ValueError("Cannot concatenate an empty list of matrix parts")
    if all(_is_polars(part) for part in parts):
        import polars as pl

        return pl.concat(parts, how="vertical")

    import pandas as pd

    return pd.concat(parts, axis=0, sort=False, ignore_index=True)


def split_matrix_table(matrix: Any) -> list[MatrixPart]:
    """Split a matrix table into source_type/target_type partitions."""
    _require_layer_columns(matrix)
    if _is_polars(matrix):
        import polars as pl

        pairs = (
            matrix.select(["source_type", "target_type"])
            .unique()
            .sort(["source_type", "target_type"])
            .iter_rows()
        )
        return [
            MatrixPart(
                source_type=str(source_type),
                target_type=str(target_type),
                table=matrix.filter(
                    (pl.col("source_type") == source_type)
                    & (pl.col("target_type") == target_type)
                ),
            )
            for source_type, target_type in pairs
        ]

    return [
        MatrixPart(source_type=str(source_type), target_type=str(target_type), table=part)
        for (source_type, target_type), part in matrix.groupby(
            ["source_type", "target_type"],
            observed=True,
            sort=True,
        )
    ]


def write_matrix_outputs(
    matrix: Any | Iterable[MatrixPart],
    *,
    output_dir: Path,
    run_tag: str,
    mode: MatrixOutputMode,
    combined_name: str = "distance_matrix",
) -> MatrixOutputSet:
    """Write combined and/or split matrix outputs and return manifest-ready paths."""
    if mode not in {"combined", "split", "both"}:
        raise ValueError(f"Unsupported matrix output mode: {mode}")

    paths: dict[str, Path] = {}
    matrix_is_table = hasattr(matrix, "columns")
    parts: list[MatrixPart] | None = None

    if mode in {"split", "both"} or not matrix_is_table:
        parts = list(matrix) if not matrix_is_table else split_matrix_table(matrix)

    if mode in {"split", "both"}:
        assert parts is not None
        for part in parts:
            path = split_matrix_path(output_dir, run_tag, part.source_type, part.target_type)
            _write_table(part.table, path)
            paths[part.key] = path

    if mode in {"combined", "both"}:
        combined_path = output_dir / f"{combined_name}_{run_tag}.parquet"
        combined = matrix if matrix_is_table and mode == "combined" else _concat_tables(
            [part.table for part in parts or []]
        )
        _write_table(combined, combined_path)
        paths = {combined_name: combined_path, **paths}

    return MatrixOutputSet(mode=mode, paths=paths)
