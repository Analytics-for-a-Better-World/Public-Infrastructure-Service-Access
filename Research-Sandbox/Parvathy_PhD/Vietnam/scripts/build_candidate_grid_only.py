from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter as pc

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "general_distances_per_country"
FRESH_BASE_ROOT = Path(r"C:\local\Parvathy\Vietnam\fresh_downloads")

sys.path.insert(0, str(PIPELINE_DIR))

from distance_pipeline.cache import CacheManager  # noqa: E402
from distance_pipeline.candidate_builder import build_candidate_grid  # noqa: E402
from distance_pipeline.config_loader import load_cfg  # noqa: E402
from distance_pipeline.settings import PipelineSettings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build only the unsnapped Vietnam candidate grid using the PISA "
            "candidate-grid cache path, without loading the road network."
        )
    )
    parser.add_argument("--country-code", default="vietnam")
    parser.add_argument("--base-root", type=Path, default=FRESH_BASE_ROOT)
    parser.add_argument("--candidate-grid-spacing-m", type=float, required=True)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = pc()
    cfg = replace(load_cfg(args.country_code), base_root=args.base_root)
    settings = PipelineSettings(
        candidate_grid_spacing_m=float(args.candidate_grid_spacing_m),
        force_recompute=bool(args.force_recompute),
        verbose=not args.quiet,
        show_context_map=False,
        save_context_map=False,
    )
    cache = CacheManager(
        cfg=cfg,
        force_recompute=settings.force_recompute,
        verbose=settings.verbose,
    )
    grid = build_candidate_grid(cfg=cfg, settings=settings, cache=cache)
    elapsed = pc() - t0
    grid_path = cache.candidate_sites_path(
        grid_spacing_m=float(args.candidate_grid_spacing_m),
        exclude_water=cfg.candidate_exclude_water,
        include_boundary=cfg.candidate_include_boundary,
    )
    summary = {
        "country_code": args.country_code,
        "base_root": str(args.base_root),
        "candidate_grid_spacing_m": float(args.candidate_grid_spacing_m),
        "candidate_grid_path": str(grid_path),
        "n_candidates": int(0 if grid is None else len(grid)),
        "seconds": float(elapsed),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
