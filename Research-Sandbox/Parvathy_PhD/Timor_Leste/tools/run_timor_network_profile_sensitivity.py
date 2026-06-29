from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_DIR = Path(
    r"C:\github\Public-Infrastructure-Service-Access\Research-Sandbox\general_distances_per_country"
)
DEFAULT_PYTHON = DEFAULT_PIPELINE_DIR / ".venv" / "Scripts" / "python.exe"
DEFAULT_SEED_ROOT = ROOT / "runs" / "network_only_20260622_1645" / "east-timor_data"


@dataclass(slots=True)
class TimorCase:
    case_id: str
    network_profile: str
    simplify_network: bool
    grid_spacing_m: int
    snap_components: str | None = None
    service_threshold_m: int = 5000
    candidate_max_snap_dist_m: int = 5000


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clock_ms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total_ms = int(round(float(seconds) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def seed_inputs(seed_root: Path, fresh_root: Path) -> dict[str, str]:
    target = fresh_root / "east-timor_data"
    target.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in ("east-timor-latest.osm.pbf", "tls_ppp_2020.tif"):
        source = seed_root / name
        destination = target / name
        if not source.exists():
            raise FileNotFoundError(source)
        if not destination.exists() or source.stat().st_size != destination.stat().st_size:
            shutil.copy2(source, destination)
        copied[name] = str(destination)
    return copied


def build_cases(spacings: list[int], profiles: list[str], simplify_values: list[bool]) -> list[TimorCase]:
    cases: list[TimorCase] = []
    for profile in profiles:
        for simplify in simplify_values:
            simplify_label = "simplified" if simplify else "unsimplified"
            for spacing in spacings:
                cases.append(
                    TimorCase(
                        case_id=f"timor_{profile}_{simplify_label}_{spacing // 1000}km",
                        network_profile=profile,
                        simplify_network=bool(simplify),
                        grid_spacing_m=int(spacing),
                    )
                )
    return cases


def build_command(
    *,
    python: Path,
    pipeline_dir: Path,
    fresh_root: Path,
    wrapper: Path,
    case: TimorCase,
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
        case.network_profile,
        "timor_leste",
        "--log-file",
        str(log_file),
        "--network-backend",
        "osmium",
        "--simplify-network",
        "true" if case.simplify_network else "false",
        "--diagnose-connectivity",
        "true",
        "--max-total-dist",
        str(case.service_threshold_m),
        "--candidate-grid-spacing-m",
        str(case.grid_spacing_m),
        "--candidate-max-snap-dist-m",
        str(case.candidate_max_snap_dist_m),
        "--sources",
        "amenities",
        "candidates",
        "--destinations",
        "population",
        "--matrix-output-mode",
        "split",
        "--no-aggregate",
    ]
    if case.snap_components:
        cmd.extend(["--snap-components", str(case.snap_components)])
    if force_recompute:
        cmd.append("--force-recompute")
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
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--spacings", type=int, nargs="+", default=[10000, 5000, 1000])
    parser.add_argument("--profiles", nargs="+", choices=("driving", "driving_walk"), default=["driving", "driving_walk"])
    parser.add_argument("--simplify", nargs="+", choices=("true", "false"), default=["true", "false"])
    parser.add_argument(
        "--snap-components",
        default=None,
        help="Comma-separated component ids passed to the pipeline for component-aware snapping.",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    wrapper = ROOT / "tools" / "run_pipeline_fresh_root_profile.py"
    simplify_values = [value == "true" for value in args.simplify]
    cases = build_cases(
        [int(value) for value in args.spacings],
        [str(value) for value in args.profiles],
        simplify_values,
    )
    if args.snap_components:
        for case in cases:
            case.snap_components = str(args.snap_components)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeded_inputs = seed_inputs(args.seed_root, args.fresh_root)

    manifest = {
        "created_at": now_iso(),
        "fresh_root": str(args.fresh_root),
        "output_dir": str(args.output_dir),
        "pipeline_dir": str(args.pipeline_dir),
        "python": str(args.python),
        "wrapper": str(wrapper),
        "seed_root": str(args.seed_root),
        "seeded_inputs": seeded_inputs,
        "cases": [asdict(case) for case in cases],
    }
    (args.output_dir / "timor_network_profile_case_manifest.json").write_text(
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
            "fresh_root": str(args.fresh_root),
            "country_output_dir": str(args.fresh_root / "east-timor_data" / "outputs"),
            "pipeline_log": str(pipeline_log),
            "console_log": str(console_log),
            "started_at": start_iso,
            "finished_at": now_iso(),
            "elapsed_seconds": float(elapsed),
            "elapsed_clock_ms": clock_ms(elapsed),
            "return_code": int(return_code),
            "command": " ".join(cmd),
        }
        rows.append(row)
        write_csv(args.output_dir / "timor_network_profile_case_timings.csv", rows)
        (args.output_dir / "timor_network_profile_case_timings.json").write_text(
            json.dumps(rows, indent=2, default=str),
            encoding="utf-8",
        )
        if return_code != 0 and args.stop_on_error:
            raise subprocess.CalledProcessError(return_code, cmd)

    print(json.dumps(rows, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
