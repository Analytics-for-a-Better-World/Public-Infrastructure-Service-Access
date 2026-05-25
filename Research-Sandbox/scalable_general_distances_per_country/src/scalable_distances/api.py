from __future__ import annotations

from pathlib import Path
from typing import Any

from scalable_distances.config import CountryDataSources
from scalable_distances.core.context import DataContext
from scalable_distances.geospatial import detect_geospatial_backend
from scalable_distances.matrix import MatrixOutputMode, MatrixOutputSet, write_matrix_outputs
from scalable_distances.storage.repository import Repository


def create_context(
    run_id: str,
    *,
    repository: Repository | None = None,
    root: str | Path | None = None,
) -> DataContext:
    """Create the default context used by API callers and notebooks."""
    if repository is None:
        if root is None:
            root = Path("data") / "runs" / run_id
        repository = Repository(root=Path(root))
    return DataContext(run_id=run_id, repository=repository)


def describe_backends() -> dict[str, Any]:
    """Return optional backend versions for reproducibility manifests."""
    return {"geospatial": detect_geospatial_backend().as_manifest()}


def describe_country_sources(
    *,
    country_slug: str,
    iso3: str,
    base_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Resolve OSM and WorldPop source naming rules without downloading data."""
    sources = CountryDataSources(
        country_slug=country_slug,
        iso3=iso3,
        base_dir=Path(base_dir),
        **kwargs,
    )
    return sources.as_manifest()


def write_distance_matrix(
    matrix: Any,
    *,
    output_dir: str | Path,
    run_tag: str,
    mode: MatrixOutputMode = "combined",
) -> MatrixOutputSet:
    """Write a distance matrix in combined, split, or both output modes."""
    return write_matrix_outputs(
        matrix,
        output_dir=Path(output_dir),
        run_tag=run_tag,
        mode=mode,
    )
