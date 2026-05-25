from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from scalable_distances.config import CountryDataSources
from scalable_distances.facilities import extract_osm_facilities
from scalable_distances.io import download_if_missing
from scalable_distances.matrix import MatrixOutputMode, MatrixOutputSet, write_matrix_outputs
from scalable_distances.network import load_driving_network
from scalable_distances.population import worldpop_to_points
from scalable_distances.routing.base import NetworkData
from scalable_distances.routing.strategies import NetworkXRouter, PandanaRouter

RouterName = Literal["networkx", "pandana"]


@dataclass(frozen=True)
class ProductionRunConfig:
    """End-to-end country run configuration."""

    sources: CountryDataSources
    output_dir: Path
    run_tag: str
    amenity_values: tuple[str, ...] = ("school",)
    router: RouterName = "networkx"
    matrix_output_mode: MatrixOutputMode = "combined"
    population_threshold: float = 1.0
    aggregate_factor: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    download: bool = True


@dataclass(frozen=True)
class ProductionRunResult:
    """Outputs and intermediates produced by a production run."""

    matrix_outputs: MatrixOutputSet
    sources: pd.DataFrame
    targets: pd.DataFrame
    network: NetworkData
    diagnostics: dict[str, object] = field(default_factory=dict)


def _router(name: RouterName):
    if name == "networkx":
        return NetworkXRouter()
    if name == "pandana":
        return PandanaRouter()
    raise ValueError(f"Unsupported router: {name}")


def run_country_pipeline(config: ProductionRunConfig) -> ProductionRunResult:
    """Run download, network, population, facilities, snapping, routing, and output writing."""
    pbf_path = config.sources.pbf_path
    worldpop_path = config.sources.resolved_worldpop_path
    if config.download:
        download_if_missing(config.sources.pbf_url, pbf_path)
        if config.sources.worldpop_path is None:
            download_if_missing(config.sources.worldpop_download_url, worldpop_path)

    network = load_driving_network(pbf_path, bbox=config.bbox)
    targets = worldpop_to_points(
        worldpop_path,
        population_threshold=config.population_threshold,
        aggregate_factor=config.aggregate_factor,
    )
    sources = extract_osm_facilities(pbf_path, amenity_values=config.amenity_values)
    if sources.empty:
        raise ValueError(f"No OSM facilities found for amenities {config.amenity_values!r}")

    router = _router(config.router)
    router.prepare(network, {})
    snapped_sources = router.snap(sources)
    snapped_targets = router.snap(targets)
    matrix = router.route_many(snapped_sources, snapped_targets)
    outputs = write_matrix_outputs(
        matrix,
        output_dir=config.output_dir,
        run_tag=config.run_tag,
        mode=config.matrix_output_mode,
    )
    return ProductionRunResult(
        matrix_outputs=outputs,
        sources=snapped_sources,
        targets=snapped_targets,
        network=network,
        diagnostics={
            "router": config.router,
            "source_count": int(len(snapped_sources)),
            "target_count": int(len(snapped_targets)),
            "matrix_rows": int(len(matrix)),
            "network_nodes": int(len(network.nodes)),
            "network_edges": int(len(network.edges)),
        },
    )
