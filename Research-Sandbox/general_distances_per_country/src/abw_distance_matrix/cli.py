"""Console entry point for the distance-matrix pipeline."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    """Run the historical ``run_pipeline.py`` CLI through the package entry point."""
    from run_pipeline import (
        build_parser,
        load_cfg,
        main as run_pipeline_main,
        resolve_input_config,
        resolve_source_layers_from_args,
        resolve_destination_layers_from_args,
        settings_from_args,
        setup_logging,
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.log_file, verbose=not args.quiet)
    settings = settings_from_args(args)
    base_cfg = load_cfg(args.country_code)
    cfg = resolve_input_config(base_cfg, args)
    source_layers = resolve_source_layers_from_args(args)
    destination_layers = resolve_destination_layers_from_args(args)

    run_pipeline_main(
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
        base_cfg=base_cfg,
    )


if __name__ == "__main__":
    main()
