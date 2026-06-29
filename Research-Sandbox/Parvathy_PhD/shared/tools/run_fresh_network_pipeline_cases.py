from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_DIR = Path(
    r"C:\github\Public-Infrastructure-Service-Access\Research-Sandbox\general_distances_per_country"
)
DEFAULT_PYTHON = DEFAULT_PIPELINE_DIR / ".venv" / "Scripts" / "python.exe"
VIETNAM_STROKE_TABLE = ROOT / "reference_cache" / "data" / "vietnam" / "vietnam_stroke_centers_130_en_source.xlsx"


@dataclass(slots=True)
class PipelineCase:
    case_id: str
    country_code: str
    country_dir_name: str
    mobility_profile: str
    grid_spacing_m: int
    service_threshold_m: int
    sources: tuple[str, ...]
    destinations: tuple[str, ...] = ("population",)
    candidate_max_snap_dist_m: int = 5000
    network_backend: str = "osmium"
    simplify_network: bool = True
    source_table: Path | None = None
    source_lon_column: str | None = None
    source_lat_column: str | None = None
    source_id_column: str | None = None

    @property
    def run_tag_marker(self) -> str:
        grid = f"candidates_spacing_{self.grid_spacing_m}"
        maxdist = f"maxdist_{self.service_threshold_m:g}"
        profile = (
            "" if self.mobility_profile == "driving"
            else "_network_osmium-walking-trails-simplified"
        )
        return f"{maxdist}_{grid}{profile}"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pipeline_cases(include_timor_walking: bool, include_vietnam: bool) -> list[PipelineCase]:
    cases: list[PipelineCase] = []
    for profile in ("driving", "walking_trails") if include_timor_walking else ("driving",):
        for spacing in (10000, 5000, 1000):
            cases.append(
                PipelineCase(
                    case_id=f"timor_leste_{profile}_{spacing // 1000}km",
                    country_code="timor_leste",
                    country_dir_name="east-timor_data",
                    mobility_profile=profile,
                    grid_spacing_m=spacing,
                    service_threshold_m=5000,
                    sources=("amenities", "candidates"),
                )
            )

    if include_vietnam:
        for spacing in (10000, 5000, 1000):
            cases.append(
                PipelineCase(
                    case_id=f"vietnam_driving_{spacing // 1000}km",
                    country_code="vietnam",
                    country_dir_name="vietnam_data",
                    mobility_profile="driving",
                    grid_spacing_m=spacing,
                    service_threshold_m=20000,
                    sources=("table", "candidates"),
                    source_table=VIETNAM_STROKE_TABLE,
                    source_lon_column="longitude",
                    source_lat_column="latitude",
                    source_id_column="TT",
                )
            )
    return cases


def build_command(
    *,
    python: Path,
    pipeline_dir: Path,
    fresh_root: Path,
    wrapper: Path,
    case: PipelineCase,
    log_file: Path,
    force_recompute: bool,
) -> list[str]:
    cmd = [
        str(python),
        str(wrapper),
        "--pipeline-dir",
        str(pipeline_dir),
        "--fresh-base-root",
        str(fresh_root),
        "--mobility-profile",
        case.mobility_profile,
        case.country_code,
        "--log-file",
        str(log_file),
        "--network-backend",
        case.network_backend,
        "--simplify-network",
        "true" if case.simplify_network else "false",
        "--max-total-dist",
        str(case.service_threshold_m),
        "--candidate-grid-spacing-m",
        str(case.grid_spacing_m),
        "--candidate-max-snap-dist-m",
        str(case.candidate_max_snap_dist_m),
        "--sources",
        *case.sources,
        "--destinations",
        *case.destinations,
        "--matrix-output-mode",
        "split",
    ]
    if force_recompute:
        cmd.append("--force-recompute")
    if case.source_table is not None:
        cmd.extend(
            [
                "--source-table",
                str(case.source_table),
                "--source-lon-column",
                str(case.source_lon_column),
                "--source-lat-column",
                str(case.source_lat_column),
                "--source-id-column",
                str(case.source_id_column),
            ]
        )
    return cmd


def run_command(cmd: list[str], *, cwd: Path, console_log: Path) -> int:
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with console_log.open("w", encoding="utf-8", errors="replace") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return int(proc.wait())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--only-country", choices=("all", "timor_leste", "vietnam"), default="all")
    parser.add_argument("--timor-walking", choices=("true", "false"), default="true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    wrapper = ROOT / "tools" / "run_pipeline_fresh_root_profile.py"
    include_timor = args.only_country in ("all", "timor_leste")
    include_vietnam = args.only_country in ("all", "vietnam")
    cases = pipeline_cases(
        include_timor_walking=args.timor_walking == "true",
        include_vietnam=include_vietnam,
    )
    if not include_timor:
        cases = [case for case in cases if case.country_code != "timor_leste"]
    if not include_vietnam:
        cases = [case for case in cases if case.country_code != "vietnam"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": now_iso(),
        "fresh_root": str(args.fresh_root),
        "output_dir": str(args.output_dir),
        "pipeline_dir": str(args.pipeline_dir),
        "python": str(args.python),
        "wrapper": str(wrapper),
        "cases": [asdict(case) for case in cases],
    }
    (args.output_dir / "pipeline_case_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for case in cases:
        log_dir = args.output_dir / "pipeline_logs" / case.case_id
        pipeline_log = log_dir / f"{case.case_id}.pipeline.log"
        console_log = log_dir / f"{case.case_id}.console.log"
        cmd = build_command(
            python=args.python,
            pipeline_dir=args.pipeline_dir,
            fresh_root=args.fresh_root,
            wrapper=wrapper,
            case=case,
            log_file=pipeline_log,
            force_recompute=bool(args.force_recompute),
        )
        print(f"\n=== {case.case_id} ===", flush=True)
        print(" ".join(cmd), flush=True)
        start_iso = now_iso()
        start = perf_counter()
        return_code = run_command(cmd, cwd=ROOT, console_log=console_log)
        elapsed = perf_counter() - start
        row = {
            **asdict(case),
            "source_table": None if case.source_table is None else str(case.source_table),
            "fresh_root": str(args.fresh_root),
            "country_output_dir": str(args.fresh_root / case.country_dir_name / "outputs"),
            "pipeline_log": str(pipeline_log),
            "console_log": str(console_log),
            "started_at": start_iso,
            "finished_at": now_iso(),
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "command": " ".join(cmd),
            "run_tag_marker": case.run_tag_marker,
        }
        rows.append(row)
        write_csv(args.output_dir / "pipeline_case_timings.csv", rows)
        (args.output_dir / "pipeline_case_timings.json").write_text(
            json.dumps(rows, indent=2, default=str),
            encoding="utf-8",
        )
        if return_code != 0 and args.stop_on_error:
            raise subprocess.CalledProcessError(return_code, cmd)

    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
