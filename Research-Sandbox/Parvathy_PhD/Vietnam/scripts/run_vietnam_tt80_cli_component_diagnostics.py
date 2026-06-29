from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(r"C:\github\Public-Infrastructure-Service-Access\Research-Sandbox\general_distances_per_country")
PYTHON = PIPELINE_DIR / ".venv" / "Scripts" / "python.exe"
RUN_ROOT = ROOT / "runs" / "vietnam_tt80_component_pipeline"
CONFIG_ROOT = RUN_ROOT / "cli_configs"
COUNTRIES_DIR = CONFIG_ROOT / "countries"
WORK_DIR = RUN_ROOT / "cli_inputs"
OUT_DIR = ROOT / "outputs" / "article_components"

SOURCE_CSV = WORK_DIR / "tt80_gia_lai_source.csv"
DUMMY_WORLDPOP = WORK_DIR / "tt80_dummy_worldpop.tif"
CLI_MANIFEST = OUT_DIR / "vietnam_tt80_pipeline_cli_runs.json"
BBOX = (108.0198, 13.9756, 108.0364, 13.9894)


def write_cli_country_config() -> None:
    COUNTRIES_DIR.mkdir(parents=True, exist_ok=True)
    (COUNTRIES_DIR / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(PIPELINE_DIR / "countries" / "base.py", COUNTRIES_DIR / "base.py")
    config = f"""from pathlib import Path

from countries.base import build_config


CFG = build_config(
    {{
        'iso3': 'VNM',
        'iso2': 'VN',
        'country_name': 'Vietnam',
        'country_slug': 'vietnam',
        'projected_epsg': 3405,
        'distance_threshold_km': 100.0,
        'geofabrik_region': 'asia',
        'worldpop_filename': 'tt80_dummy_worldpop.tif',
        'worldpop_path': Path(r'{DUMMY_WORLDPOP}'),
        'pbf_filename': 'vietnam-latest.osm.pbf',
        'base_root': Path(r'{RUN_ROOT}'),
        'plot_title_suffix': 'TT80 component diagnostic from pipeline CLI',
        'candidate_grid_spacing_m': 10000.0,
        'candidate_max_snap_dist_m': 5000.0,
        'candidate_exclude_water': False,
        'aggregate_factor': None,
    }}
)
"""
    (COUNTRIES_DIR / "vietnam_tt80_cli.py").write_text(config, encoding="utf-8")


def write_inputs() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_CSV.write_text(
        "TT,Name_English,longitude,latitude\n"
        "80,Gia Lai Provincial General Hospital,108.027844,13.982478\n",
        encoding="utf-8",
    )
    if not DUMMY_WORLDPOP.exists():
        # The current CLI constructs population points before it reaches table
        # snapping, even for table-only runs. This tiny raster satisfies that
        # unconditional step; the diagnostic source/target layers below are
        # both table layers, so the raster is not used for the snap comparison.
        transform = from_origin(108.0275, 13.9829, 0.0002, 0.0002)
        with rasterio.open(
            DUMMY_WORLDPOP,
            "w",
            driver="GTiff",
            height=1,
            width=1,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(np.array([[1.0]], dtype="float32"), 1)


def base_command(log_file: Path) -> list[str]:
    min_lon, min_lat, max_lon, max_lat = BBOX
    return [
        str(PYTHON),
        "-m",
        "run_pipeline",
        "vietnam_tt80_cli",
        "--sources",
        "table",
        "--destinations",
        "table",
        "--source-table",
        str(SOURCE_CSV),
        "--destination-table",
        str(SOURCE_CSV),
        "--source-lon-column",
        "longitude",
        "--source-lat-column",
        "latitude",
        "--source-id-column",
        "TT",
        "--destination-lon-column",
        "longitude",
        "--destination-lat-column",
        "latitude",
        "--destination-id-column",
        "TT",
        "--bbox",
        str(min_lon),
        str(min_lat),
        str(max_lon),
        str(max_lat),
        "--network-backend",
        "osmium",
        "--no-aggregate",
        "--max-total-dist",
        "150000",
        "--diagnose-connectivity",
        "true",
        "--log-file",
        str(log_file),
    ]


def run_cli(label: str, extra_args: list[str], *, force_recompute: bool = False) -> dict[str, object]:
    log_file = OUT_DIR / f"vietnam_tt80_cli_{label}.log"
    cmd = base_command(log_file)
    if force_recompute:
        cmd.append("--force-recompute")
    cmd.extend(extra_args)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CONFIG_ROOT}{os.pathsep}{PIPELINE_DIR}"
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "label": label,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "log_file": str(log_file),
    }


def main() -> None:
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_inputs()
    write_cli_country_config()

    runs = [
        run_cli("unrestricted", [], force_recompute=True),
        run_cli("snap_component_0", ["--snap-components", "0"], force_recompute=False),
    ]
    for run in runs:
        if run["returncode"] != 0:
            print(json.dumps(run, indent=2))
            raise SystemExit(int(run["returncode"]))

    payload = {
        "pipeline_dir": str(PIPELINE_DIR),
        "python": str(PYTHON),
        "config_root": str(CONFIG_ROOT),
        "run_root": str(RUN_ROOT),
        "source_csv": str(SOURCE_CSV),
        "dummy_worldpop": str(DUMMY_WORLDPOP),
        "bbox": {
            "min_lon": BBOX[0],
            "min_lat": BBOX[1],
            "max_lon": BBOX[2],
            "max_lat": BBOX[3],
        },
        "runs": runs,
    }
    CLI_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
