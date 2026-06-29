from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


def main() -> None:
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--pipeline-dir", type=Path, required=True)
    wrapper_parser.add_argument("--fresh-base-root", type=Path, required=True)
    wrapper_args, remaining = wrapper_parser.parse_known_args()

    pipeline_dir = wrapper_args.pipeline_dir.resolve()
    sys.path.insert(0, str(pipeline_dir))

    from distance_pipeline.config_loader import load_cfg
    from run_pipeline import (
        build_parser,
        main as pipeline_main,
        resolve_destination_layers_from_args,
        resolve_input_config,
        resolve_source_layers_from_args,
        settings_from_args,
        setup_logging,
    )

    parser = build_parser()
    args = parser.parse_args(remaining)

    setup_logging(args.log_file, verbose=not args.quiet)
    settings = settings_from_args(args)
    base_cfg = load_cfg(args.country_code)
    cfg = resolve_input_config(base_cfg, args)
    cfg = replace(cfg, base_root=wrapper_args.fresh_base_root)
    source_layers = resolve_source_layers_from_args(args)
    destination_layers = resolve_destination_layers_from_args(args)

    pipeline_main(
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
