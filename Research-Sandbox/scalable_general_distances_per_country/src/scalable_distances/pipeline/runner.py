from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import pandas as pd

from scalable_distances.candidates import build_candidate_grid
from scalable_distances.config import CountryDataSources
from scalable_distances.facilities import extract_osm_facilities
from scalable_distances.io import download_if_missing
from scalable_distances.layers import filter_bbox, load_point_table, normalize_layers
from scalable_distances.manifest import build_manifest, file_metadata, write_manifest
from scalable_distances.matrix.dense import write_dense_matrix_outputs
from scalable_distances.matrix import MatrixOutputMode, MatrixOutputSet, write_matrix_outputs
from scalable_distances.network import load_driving_network
from scalable_distances.population import worldpop_to_points
from scalable_distances.routing.base import NetworkData
from scalable_distances.routing.strategies import NetworkXRouter, PandanaRouter

RouterName = Literal["networkx", "pandana"]
MatrixShape = Literal["sparse", "dense"]


@dataclass(frozen=True)
class ProductionRunConfig:
    """End-to-end country run configuration."""

    sources: CountryDataSources
    output_dir: Path
    run_tag: str
    amenity_values: tuple[str, ...] = ("school",)
    source_layers: tuple[str, ...] = ("amenities",)
    destination_layers: tuple[str, ...] = ("population",)
    router: RouterName = "networkx"
    matrix_output_mode: MatrixOutputMode = "combined"
    matrix_shape: MatrixShape = "sparse"
    population_threshold: float = 1.0
    aggregate_factor: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    max_total_dist: float | None = None
    table_path: Path | None = None
    table_lon_col: str = "lon"
    table_lat_col: str = "lat"
    table_id_col: str | None = None
    destination_table_path: Path | None = None
    destination_table_lon_col: str = "lon"
    destination_table_lat_col: str = "lat"
    destination_table_id_col: str | None = None
    candidate_grid_spacing_m: float | None = None
    candidate_max_snap_dist_m: float | None = None
    include_osm_ways: bool = True
    network_backend: str = "osmium"
    download: bool = True


@dataclass(frozen=True)
class ProductionRunResult:
    """Outputs and intermediates produced by a production run."""

    matrix_outputs: MatrixOutputSet
    sources: pd.DataFrame
    targets: pd.DataFrame
    network: NetworkData
    diagnostics: dict[str, object] = field(default_factory=dict)


def _write_points(points: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    points.to_parquet(path, index=False)
    return path


def _router(name: RouterName):
    if name == "networkx":
        return NetworkXRouter()
    if name == "pandana":
        return PandanaRouter()
    raise ValueError(f"Unsupported router: {name}")


def _population_layer(
    worldpop_path: Path,
    *,
    role: Literal["source", "target"],
    population_threshold: float,
    aggregate_factor: int | None,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    points = worldpop_to_points(
        worldpop_path,
        population_threshold=population_threshold,
        aggregate_factor=aggregate_factor,
    )
    points = filter_bbox(points, bbox)
    if role == "target":
        return points
    points = points.rename(columns={"target_id": "source_id", "target_type": "source_type"})
    points["source_type"] = "population"
    return points


def _amenity_layer(
    pbf_path: Path,
    *,
    role: Literal["source", "target"],
    amenity_values: tuple[str, ...],
    include_osm_ways: bool,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    points = extract_osm_facilities(
        pbf_path,
        amenity_values=amenity_values,
        role=role,
        include_ways=include_osm_ways,
    )
    return filter_bbox(points, bbox)


def _table_layer(
    table_path: Path | None,
    *,
    role: Literal["source", "target"],
    lon_col: str,
    lat_col: str,
    id_col: str | None,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    if table_path is None:
        raise ValueError(f"{role} table layer requested, but no table path was provided.")
    return load_point_table(
        table_path,
        role=role,
        layer_type="table",
        lon_col=lon_col,
        lat_col=lat_col,
        id_col=id_col,
        bbox=bbox,
    )


def _candidate_layer(
    bbox: tuple[float, float, float, float] | None,
    *,
    role: Literal["source", "target"],
    spacing_m: float | None,
) -> pd.DataFrame:
    if bbox is None:
        raise ValueError("Candidate grid generation requires --bbox.")
    return build_candidate_grid(bbox, spacing_m=spacing_m or 1000.0, role=role)


def _concat(parts: list[pd.DataFrame], *, label: str) -> pd.DataFrame:
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        raise ValueError(f"No {label} points were produced.")
    return pd.concat(nonempty, ignore_index=True, sort=False)


def _build_layers(
    config: ProductionRunConfig,
    *,
    pbf_path: Path,
    worldpop_path: Path,
    role: Literal["source", "target"],
) -> pd.DataFrame:
    layer_names = config.source_layers if role == "source" else config.destination_layers
    parts: list[pd.DataFrame] = []
    for layer_name in layer_names:
        if layer_name == "population":
            parts.append(
                _population_layer(
                    worldpop_path,
                    role=role,
                    population_threshold=config.population_threshold,
                    aggregate_factor=config.aggregate_factor,
                    bbox=config.bbox,
                )
            )
        elif layer_name == "amenities":
            parts.append(
                _amenity_layer(
                    pbf_path,
                    role=role,
                    amenity_values=config.amenity_values,
                    include_osm_ways=config.include_osm_ways,
                    bbox=config.bbox,
                )
            )
        elif layer_name == "table":
            parts.append(
                _table_layer(
                    config.table_path if role == "source" else config.destination_table_path,
                    role=role,
                    lon_col=config.table_lon_col if role == "source" else config.destination_table_lon_col,
                    lat_col=config.table_lat_col if role == "source" else config.destination_table_lat_col,
                    id_col=config.table_id_col if role == "source" else config.destination_table_id_col,
                    bbox=config.bbox,
                )
            )
        elif layer_name == "candidates":
            parts.append(
                _candidate_layer(
                    config.bbox,
                    role=role,
                    spacing_m=config.candidate_grid_spacing_m,
                )
            )
        else:
            raise ValueError(f"Unsupported layer {layer_name!r}")
    return _concat(parts, label=role)


def _filter_candidate_snap_distance(
    points: pd.DataFrame,
    *,
    role: Literal["source", "target"],
    max_distance_m: float | None,
) -> pd.DataFrame:
    if max_distance_m is None or "snap_dist_m" not in points.columns:
        return points
    type_col = f"{role}_type"
    keep = (points[type_col] != "candidates") | (points["snap_dist_m"] <= max_distance_m)
    return points.loc[keep].reset_index(drop=True)


def _network_diagnostics(network: NetworkData) -> dict[str, object]:
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_edges_from(network.edges[[network.source_col, network.target_col]].itertuples(index=False, name=None))
        components = [len(component) for component in nx.connected_components(graph)]
        return {
            "weak_component_count": len(components),
            "largest_weak_component_nodes": max(components) if components else 0,
        }
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return {"weak_component_error": str(exc)}


def run_country_pipeline(config: ProductionRunConfig) -> ProductionRunResult:
    """Run download, network, population, facilities, snapping, routing, and output writing."""
    if config.network_backend not in {"auto", "osmium", "npyosmium"}:
        raise ValueError("Only the osmium/npyosmium network backend is currently implemented.")
    source_layers = normalize_layers(config.source_layers, default=("amenities",))
    destination_layers = normalize_layers(config.destination_layers, default=("population",))
    config = replace(
        config,
        source_layers=source_layers,
        destination_layers=destination_layers,
    )
    pbf_path = config.sources.pbf_path
    worldpop_path = config.sources.resolved_worldpop_path
    if config.download:
        download_if_missing(config.sources.pbf_url, pbf_path)
        if config.sources.worldpop_path is None:
            download_if_missing(config.sources.worldpop_download_url, worldpop_path)

    network = load_driving_network(pbf_path, bbox=config.bbox)
    sources = _build_layers(config, pbf_path=pbf_path, worldpop_path=worldpop_path, role="source")
    targets = _build_layers(config, pbf_path=pbf_path, worldpop_path=worldpop_path, role="target")

    router = _router(config.router)
    router.prepare(network, {})
    snapped_sources = router.snap(sources)
    snapped_targets = router.snap(targets)
    snapped_sources = _filter_candidate_snap_distance(
        snapped_sources,
        role="source",
        max_distance_m=config.candidate_max_snap_dist_m,
    )
    snapped_targets = _filter_candidate_snap_distance(
        snapped_targets,
        role="target",
        max_distance_m=config.candidate_max_snap_dist_m,
    )
    matrix = router.route_many(snapped_sources, snapped_targets)
    if config.max_total_dist is not None:
        matrix = matrix.loc[matrix["total_dist"] <= config.max_total_dist].reset_index(drop=True)
    if config.matrix_shape == "dense":
        outputs = write_dense_matrix_outputs(
            matrix,
            sources=snapped_sources,
            targets=snapped_targets,
            output_dir=config.output_dir,
            run_tag=config.run_tag,
            mode=config.matrix_output_mode,
        )
    else:
        outputs = write_matrix_outputs(
            matrix,
            output_dir=config.output_dir,
            run_tag=config.run_tag,
            mode=config.matrix_output_mode,
        )
    source_path = _write_points(snapped_sources, config.output_dir / f"sources_{config.run_tag}.parquet")
    target_path = _write_points(snapped_targets, config.output_dir / f"targets_{config.run_tag}.parquet")
    manifest_path = config.output_dir / f"run_manifest_{config.run_tag}.yaml"
    diagnostics = {
        "router": config.router,
        "source_layers": list(source_layers),
        "destination_layers": list(destination_layers),
        "source_count": int(len(snapped_sources)),
        "target_count": int(len(snapped_targets)),
        "matrix_rows": int(len(matrix)),
        "network_nodes": int(len(network.nodes)),
        "network_edges": int(len(network.edges)),
        "matrix_shape": config.matrix_shape,
        **_network_diagnostics(network),
    }
    manifest = build_manifest(
        manifest_kind="production_country_run",
        implementation={
            "package": "scalable_general_distances_per_country",
            "runner": "scalable_distances.pipeline.run_country_pipeline",
        },
        cache={
            "base_dir": config.sources.base_dir,
            "files": {
                "pbf": file_metadata(pbf_path, role="downloaded_or_reused", url=config.sources.pbf_url),
                "worldpop": file_metadata(
                    worldpop_path,
                    role="downloaded_or_reused",
                    url=None if config.sources.worldpop_path is not None else config.sources.worldpop_download_url,
                ),
            },
        },
        inputs=config.sources.as_manifest(),
        parameters={
            "amenity_values": list(config.amenity_values),
            "source_layers": list(source_layers),
            "destination_layers": list(destination_layers),
            "bbox": config.bbox,
            "population_threshold": config.population_threshold,
            "aggregate_factor": config.aggregate_factor,
            "max_total_dist": config.max_total_dist,
            "candidate_grid_spacing_m": config.candidate_grid_spacing_m,
            "candidate_max_snap_dist_m": config.candidate_max_snap_dist_m,
            "matrix_output_mode": config.matrix_output_mode,
            "matrix_shape": config.matrix_shape,
            "router": config.router,
            "network_backend": config.network_backend,
        },
        intermediate_artifacts={"sources": source_path, "targets": target_path},
        outputs={key: path for key, path in outputs.paths.items()} | {"manifest": manifest_path},
        diagnostics=diagnostics,
    )
    write_manifest(manifest, manifest_path)
    return ProductionRunResult(
        matrix_outputs=outputs,
        sources=snapped_sources,
        targets=snapped_targets,
        network=network,
        diagnostics=diagnostics | {"manifest": str(manifest_path)},
    )
