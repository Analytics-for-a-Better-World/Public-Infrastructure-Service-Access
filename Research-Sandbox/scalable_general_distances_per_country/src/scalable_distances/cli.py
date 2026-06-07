from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scalable_distances.api import describe_backends, describe_country_sources, run_production_country, write_distance_matrix
from scalable_distances.config import CountryDataSources
from scalable_distances.pipeline import ProductionRunConfig


def _json_print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _smoke_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "school_1",
                "target_id": "pop_1",
                "source_type": "amenities",
                "target_type": "population",
                "total_dist": 10.0,
            },
            {
                "source_id": "candidate_1",
                "target_id": "pop_1",
                "source_type": "candidates",
                "target_type": "population",
                "total_dist": 15.0,
            },
            {
                "source_id": "school_1",
                "target_id": "table_1",
                "source_type": "amenities",
                "target_type": "table",
                "total_dist": 20.0,
            },
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Utilities for the scalable general-distance pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("backends", help="Print optional backend versions.")

    sources = subparsers.add_parser(
        "sources",
        help="Resolve OSM and WorldPop source URLs/paths without downloading.",
    )
    sources.add_argument("--country-slug", required=True)
    sources.add_argument("--iso3", required=True)
    sources.add_argument("--base-dir", required=True)
    sources.add_argument("--worldpop-dataset", choices=["global1", "global2"], default="global1")
    sources.add_argument("--worldpop-year", type=int, default=2020)
    sources.add_argument("--worldpop-release")
    sources.add_argument("--worldpop-version", default="v1")
    sources.add_argument("--worldpop-resolution", default="100m")
    sources.add_argument("--worldpop-constrained", action="store_true")

    smoke = subparsers.add_parser(
        "split-smoke",
        help="Write a tiny combined/split matrix set for smoke testing.",
    )
    smoke.add_argument("--output-dir", default="diagnostics/split_matrix_smoke")
    smoke.add_argument("--run-tag", default="smoke")
    smoke.add_argument("--mode", choices=["combined", "split", "both"], default="both")

    run = subparsers.add_parser(
        "run",
        help="Run the full country pipeline: download, parse, rasterize, extract, snap, route, write.",
    )
    run.add_argument("--country-slug", required=True)
    run.add_argument("--iso3", required=True)
    run.add_argument("--base-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--run-tag", required=True)
    run.add_argument("--amenity", action="append", dest="amenities", default=[])
    run.add_argument(
        "--sources",
        nargs="+",
        default=["amenities"],
        help="Source layers: population, amenities, table, candidates.",
    )
    run.add_argument(
        "--destinations",
        nargs="+",
        default=["population"],
        help="Destination layers: population, amenities, table, candidates.",
    )
    run.add_argument("--router", choices=["networkx", "pandana"], default="networkx")
    run.add_argument("--matrix-output-mode", choices=["combined", "split", "both"], default="combined")
    run.add_argument("--matrix-shape", choices=["sparse", "dense"], default="sparse")
    run.add_argument("--population-threshold", type=float, default=1.0)
    run.add_argument("--aggregate-factor", type=int)
    run.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    run.add_argument("--max-total-dist", type=float)
    run.add_argument("--table")
    run.add_argument("--table-lon-col", default="lon")
    run.add_argument("--table-lat-col", default="lat")
    run.add_argument("--table-id-col")
    run.add_argument("--destination-table")
    run.add_argument("--destination-table-lon-col", default="lon")
    run.add_argument("--destination-table-lat-col", default="lat")
    run.add_argument("--destination-table-id-col")
    run.add_argument("--candidate-grid-spacing-m", type=float)
    run.add_argument("--candidate-max-snap-dist-m", type=float)
    run.add_argument("--no-osm-ways", action="store_true")
    run.add_argument("--network-backend", choices=["auto", "osmium", "npyosmium"], default="osmium")
    run.add_argument("--worldpop-dataset", choices=["global1", "global2"], default="global1")
    run.add_argument("--worldpop-year", type=int, default=2020)
    run.add_argument("--worldpop-release")
    run.add_argument("--worldpop-version", default="v1")
    run.add_argument("--worldpop-resolution", default="100m")
    run.add_argument("--worldpop-constrained", action="store_true")
    run.add_argument("--no-download", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "backends":
        _json_print(describe_backends())
        return 0

    if args.command == "sources":
        _json_print(
            describe_country_sources(
                country_slug=args.country_slug,
                iso3=args.iso3,
                base_dir=args.base_dir,
                worldpop_dataset=args.worldpop_dataset,
                worldpop_year=args.worldpop_year,
                worldpop_release=args.worldpop_release,
                worldpop_version=args.worldpop_version,
                worldpop_resolution=args.worldpop_resolution,
                worldpop_constrained=args.worldpop_constrained,
            )
        )
        return 0

    if args.command == "split-smoke":
        result = write_distance_matrix(
            _smoke_matrix(),
            output_dir=Path(args.output_dir),
            run_tag=args.run_tag,
            mode=args.mode,
        )
        _json_print({"mode": result.mode, "paths": result.paths})
        return 0

    if args.command == "run":
        sources_config = CountryDataSources(
            country_slug=args.country_slug,
            iso3=args.iso3,
            base_dir=Path(args.base_dir),
            worldpop_dataset=args.worldpop_dataset,
            worldpop_year=args.worldpop_year,
            worldpop_release=args.worldpop_release,
            worldpop_version=args.worldpop_version,
            worldpop_resolution=args.worldpop_resolution,
            worldpop_constrained=args.worldpop_constrained,
        )
        result = run_production_country(
            ProductionRunConfig(
                sources=sources_config,
                output_dir=Path(args.output_dir),
                run_tag=args.run_tag,
                amenity_values=tuple(args.amenities or ["school"]),
                source_layers=tuple(args.sources),
                destination_layers=tuple(args.destinations),
                router=args.router,
                matrix_output_mode=args.matrix_output_mode,
                matrix_shape=args.matrix_shape,
                population_threshold=args.population_threshold,
                aggregate_factor=args.aggregate_factor,
                bbox=None if args.bbox is None else tuple(args.bbox),
                max_total_dist=args.max_total_dist,
                table_path=None if args.table is None else Path(args.table),
                table_lon_col=args.table_lon_col,
                table_lat_col=args.table_lat_col,
                table_id_col=args.table_id_col,
                destination_table_path=None
                if args.destination_table is None
                else Path(args.destination_table),
                destination_table_lon_col=args.destination_table_lon_col,
                destination_table_lat_col=args.destination_table_lat_col,
                destination_table_id_col=args.destination_table_id_col,
                candidate_grid_spacing_m=args.candidate_grid_spacing_m,
                candidate_max_snap_dist_m=args.candidate_max_snap_dist_m,
                include_osm_ways=not args.no_osm_ways,
                network_backend=args.network_backend,
                download=not args.no_download,
            )
        )
        _json_print(
            {
                "diagnostics": result.diagnostics,
                "matrix_outputs": result.matrix_outputs.paths,
            }
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
