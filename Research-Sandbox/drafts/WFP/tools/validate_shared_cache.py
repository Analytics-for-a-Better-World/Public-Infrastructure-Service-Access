from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


WFP_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_SANDBOX = WFP_ROOT.parents[1]
ORIGINAL_ROOT = RESEARCH_SANDBOX / "general_distances_per_country"
OVERLEAF_ROOT = Path(
    r"C:\Users\joaqu\Dropbox\Apps\Overleaf\Real Life Distance Generator"
)

if str(ORIGINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_ROOT))
if str(WFP_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(WFP_ROOT / "src"))

from countries.base import CountryConfig  # noqa: E402
from distance_pipeline.boundaries import (  # noqa: E402
    NATURAL_EARTH_COUNTRIES_URL,
    download_natural_earth_boundaries,
)
from distance_pipeline.cache import CacheManager  # noqa: E402
from distance_pipeline.config_loader import load_cfg  # noqa: E402
from distance_pipeline.io import download_file  # noqa: E402
from wfp_access.manifest import build_manifest, write_manifest  # noqa: E402


@dataclass(frozen=True)
class ValidationCase:
    name: str
    country_code: str | None
    description: str
    original_command: list[str] | None
    reengineered_status: str
    expected_summary: dict[str, Any]
    needs_natural_earth: bool = True


CASES: tuple[ValidationCase, ...] = (
    ValidationCase(
        name="luxembourg_schools_full",
        country_code="luxembourg",
        description="Luxembourg school accessibility, unaggregated population, 1 km total-distance filter.",
        original_command=[
            "py",
            "run_pipeline.py",
            "luxembourg",
            "--amenity",
            "school",
            "--population-threshold",
            "1",
            "--sample-fraction",
            "1",
            "--max-total-dist",
            "1000",
            "--save-map",
        ],
        reengineered_status="Implemented in notebooks/luxembourg_school_case_architecture.ipynb; full matrix cache may need explicit rebuild.",
        expected_summary={
            "population_points": 102061,
            "sources": 490,
            "matrix_rows_paper": 65583,
        },
    ),
    ValidationCase(
        name="luxembourg_schools_agg10",
        country_code="luxembourg",
        description="Luxembourg school accessibility, aggregate factor 10, 1 km total-distance filter.",
        original_command=[
            "py",
            "run_pipeline.py",
            "luxembourg",
            "--aggregate-factor",
            "10",
            "--amenity",
            "school",
            "--population-threshold",
            "1",
            "--sample-fraction",
            "1",
            "--max-total-dist",
            "1000",
            "--save-map",
        ],
        reengineered_status="Implemented and executed in notebooks/luxembourg_school_case_architecture.ipynb.",
        expected_summary={
            "population_points": 4800,
            "current_architecture_matrix_rows": 879,
        },
    ),
    ValidationCase(
        name="timor_leste_health_agg8_10km",
        country_code="timor_leste",
        description="Timor-Leste health-service maximum-covering input matrix, aggregate factor 8, 10 km total-distance filter.",
        original_command=[
            "py",
            "run_pipeline.py",
            "timor_leste",
            "--aggregate-factor",
            "8",
            "--population-threshold",
            "1",
            "--sample-fraction",
            "1",
            "--max-total-dist",
            "10000",
        ],
        reengineered_status="Not yet implemented as a WFP architecture case.",
        expected_summary={
            "population_points_paper": 28162,
            "sources_paper": 860,
            "retained_paths_paper": 15688422,
            "distance_component_rows_paper": 100997,
        },
    ),
    ValidationCase(
        name="vietnam_health_agg10_150km",
        country_code="vietnam",
        description="Vietnam health-service maximum-covering input matrix, aggregate factor 10, 150 km total-distance filter.",
        original_command=[
            "py",
            "run_pipeline.py",
            "vietnam",
            "--aggregate-factor",
            "10",
            "--population-threshold",
            "1",
            "--sample-fraction",
            "1",
            "--max-total-dist",
            "150000",
        ],
        reengineered_status="Not yet implemented as a WFP architecture case.",
        expected_summary={
            "population_points_paper": 408838,
            "sources_paper": 8795,
            "retained_paths_paper": 192654000,
            "distance_component_rows_paper": 208568143,
        },
    ),
    ValidationCase(
        name="nusa_tenggara_schools_candidates_2km",
        country_code="nusa_tenggara",
        description="Nusa Tenggara candidate-to-school matrix using 2 km candidate grid and school table destinations.",
        original_command=[
            "py",
            "run_pipeline.py",
            "nusa_tenggara",
            "--sources",
            "candidates",
            "--destinations",
            "table",
            "--destination-table",
            r"C:\local\GIT\route-the-meals\geocoding\draft\17_routing_targets_enhanced.csv",
            "--destination-lon-column",
            "routing_lon",
            "--destination-lat-column",
            "routing_lat",
            "--destination-id-column",
            "source_id",
            "--bbox",
            "118.8",
            "-11.1",
            "125.4",
            "-7.1",
            "--candidate-grid-spacing-m",
            "2000",
            "--candidate-max-snap-dist-m",
            "1000",
            "--max-total-dist",
            "150000",
        ],
        reengineered_status="Not yet implemented as a WFP architecture case.",
        expected_summary={
            "targets_paper": 14099,
            "sources_paper": 7755,
            "retained_paths_paper": 24952321,
            "distance_component_rows_paper": 15947644,
        },
    ),
    ValidationCase(
        name="luxembourg_aed_network_nodes",
        country_code="luxembourg",
        description="Luxembourg AED network-node set-covering experiment from defibrillator_experiments.py.",
        original_command=["py", "defibrillator_experiments.py"],
        reengineered_status="Not yet implemented as a WFP architecture case.",
        expected_summary={},
        needs_natural_earth=False,
    ),
)


def selected_cases(names: list[str] | None) -> list[ValidationCase]:
    if not names:
        return list(CASES)
    by_name = {case.name: case for case in CASES}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise SystemExit(f"Unknown validation case(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def cfg_for(case: ValidationCase) -> CountryConfig | None:
    if case.country_code is None:
        return None
    return load_cfg(case.country_code)


def file_record(path: Path, url: str | None = None, role: str = "download", do_hash: bool = False) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "url": url,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "mtime_utc": path.stat().st_mtime if exists else None,
    }
    if exists and do_hash:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        record["sha256"] = digest.hexdigest()
    return record


def expected_downloads(case: ValidationCase) -> list[dict[str, Any]]:
    cfg = cfg_for(case)
    if cfg is None:
        return []
    cache = CacheManager(cfg, force_recompute=False, verbose=False)
    records = [
        file_record(cfg.PBF_PATH, cfg.PBF_URL, role="download:geofabrik_pbf"),
        file_record(cfg.WORLDPOP_PATH, cfg.WORLDPOP_URL, role="download:worldpop_raster"),
    ]
    if case.needs_natural_earth:
        records.append(
            file_record(
                cache.boundary_archive_path(),
                NATURAL_EARTH_COUNTRIES_URL,
                role="download:natural_earth_boundaries",
            )
        )
    return records


def inventory(cases: list[ValidationCase], do_hash: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "original_root": str(ORIGINAL_ROOT),
        "wfp_root": str(WFP_ROOT),
        "overleaf_root": str(OVERLEAF_ROOT),
        "cases": [],
    }
    for case in cases:
        cfg = cfg_for(case)
        downloads = []
        for record in expected_downloads(case):
            path = Path(record["path"])
            downloads.append(
                file_record(path, record["url"], role=record["role"], do_hash=do_hash)
            )
        payload["cases"].append(
            {
                **asdict(case),
                "shared_base_dir": str(cfg.BASE_DIR) if cfg else None,
                "downloads": downloads,
            }
        )
    return payload


def inventory_manifest(cases: list[ValidationCase], do_hash: bool = False) -> dict[str, Any]:
    payload = inventory(cases, do_hash=do_hash)
    inputs: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {
        "case_count": len(payload["cases"]),
        "download_count": 0,
        "missing_download_count": 0,
    }

    for case in payload["cases"]:
        case_inputs: dict[str, Any] = {}
        for download in case["downloads"]:
            case_inputs[download["role"]] = download
            diagnostics["download_count"] += 1
            if not download["exists"]:
                diagnostics["missing_download_count"] += 1
        inputs[case["name"]] = case_inputs

    return build_manifest(
        manifest_kind="cache_inventory",
        implementation={
            "name": "wfp_access",
            "role": "reengineered",
            "root": str(WFP_ROOT),
        },
        code={},
        case={
            "names": [case["name"] for case in payload["cases"]],
        },
        cache={
            "root": r"C:\local\Download_Depot",
            "policy": "downloaded inputs are restored with overwrite=False and reused when present",
        },
        inputs=inputs,
        parameters={
            "hash_files": do_hash,
            "overleaf_root": str(OVERLEAF_ROOT),
            "original_root": str(ORIGINAL_ROOT),
        },
        diagnostics=diagnostics,
    )


def restore_downloads(cases: list[ValidationCase]) -> None:
    seen: set[Path] = set()
    for case in cases:
        cfg = cfg_for(case)
        if cfg is None:
            continue
        for path, url in [(cfg.PBF_PATH, cfg.PBF_URL), (cfg.WORLDPOP_PATH, cfg.WORLDPOP_URL)]:
            if path in seen:
                continue
            seen.add(path)
            download_file(url, path, overwrite=False, verbose=True)
        if case.needs_natural_earth:
            cache = CacheManager(cfg, force_recompute=False, verbose=True)
            if cache.boundary_archive_path() not in seen:
                seen.add(cache.boundary_archive_path())
                download_natural_earth_boundaries(
                    cache.boundary_archive_path(),
                    overwrite=False,
                    verbose=True,
                )


def run_original(case: ValidationCase, dry_run: bool = True) -> int:
    if not case.original_command:
        print(f"{case.name}: no original command defined")
        return 2
    print(" ".join(case.original_command))
    if dry_run:
        return 0
    return subprocess.call(case.original_command, cwd=ORIGINAL_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Shared-cache validation harness for original and WFP pipelines.")
    parser.add_argument(
        "action",
        choices=("inventory", "manifest", "restore-downloads", "run-original"),
    )
    parser.add_argument("--case", action="append", dest="cases", help="Case name; repeat to select multiple.")
    parser.add_argument("--hash", action="store_true", help="Hash existing files during inventory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--execute", action="store_true", help="Actually execute original commands for run-original.")
    args = parser.parse_args()

    cases = selected_cases(args.cases)

    if args.action == "inventory":
        payload = inventory(cases, do_hash=args.hash)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text)
        return 0

    if args.action == "manifest":
        payload = inventory_manifest(cases, do_hash=args.hash)
        if args.output is None:
            args.output = WFP_ROOT / "diagnostics" / "shared_cache_manifest.yaml"
        write_manifest(payload, args.output)
        print(args.output)
        return 0

    if args.action == "restore-downloads":
        restore_downloads(cases)
        return 0

    if args.action == "run-original":
        status = 0
        for case in cases:
            status = max(status, run_original(case, dry_run=not args.execute))
        return status

    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
