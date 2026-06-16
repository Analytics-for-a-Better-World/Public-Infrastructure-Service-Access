from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "general_distances_per_country"
FRESH_BASE_ROOT = Path(r"C:\local\Parvathy\Vietnam\fresh_downloads")
PDF_FIGURE_DIR = Path(r"C:\local\Parvathy\Vietnam\figures")

sys.path.insert(0, str(PIPELINE_DIR))

from distance_pipeline.config_loader import load_cfg  # noqa: E402
from run_pipeline import (  # noqa: E402
    build_parser,
    main,
    resolve_destination_layers_from_args,
    resolve_source_layers_from_args,
    resolve_worldpop_config,
    settings_from_args,
    setup_logging,
)


def _default_pdf_map_path(args, cfg) -> None:
    """Prefer PDF context maps for this Vietnam rerun when map output is requested."""
    if not (args.save_map or args.build_map or args.map_only):
        return
    if args.map_path:
        return
    PDF_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    args.map_path = str(PDF_FIGURE_DIR / f"{cfg.country_slug}_stroke_access_context_map.pdf")


def run() -> None:
    parser = build_parser()
    parser.description = (
        "Run the Vietnam PISA pipeline for Parvathy's PhD reproducibility work. "
        "Pipeline code is imported unchanged from Research-Sandbox/general_distances_per_country. "
        "Only the country base_root is redirected to the fresh Parvathy Vietnam data root."
    )
    parser.add_argument(
        "--distance-threshold-km",
        type=float,
        default=None,
        help=(
            "Vietnam-local override for the spatial prefilter distance used by the "
            "PISA matrix builder. Use this for dense grids where the default 100 km "
            "country threshold would create an infeasible number of source-target pairs."
        ),
    )
    args = parser.parse_args()

    setup_logging(args.log_file, verbose=not args.quiet)
    cfg = resolve_worldpop_config(load_cfg(args.country_code), args)
    if args.distance_threshold_km is not None:
        if args.distance_threshold_km <= 0:
            raise ValueError("--distance-threshold-km must be positive.")
        cfg = replace(cfg, distance_threshold_km=float(args.distance_threshold_km))
    cfg = replace(cfg, base_root=FRESH_BASE_ROOT)
    _default_pdf_map_path(args, cfg)
    settings = settings_from_args(args)

    source_layers = resolve_source_layers_from_args(args)
    destination_layers = resolve_destination_layers_from_args(args)

    main(
        cfg,
        settings,
        args.aggregate_factor,
        args.no_aggregate,
        args.build_map,
        args.map_only,
        args.amenity,
        source_layers,
        destination_layers,
        args.source_table,
        args.source_lon_column,
        args.source_lat_column,
        args.source_id_column,
        args.destination_table,
        args.destination_lon_column,
        args.destination_lat_column,
        args.destination_id_column,
    )


if __name__ == "__main__":
    run()
