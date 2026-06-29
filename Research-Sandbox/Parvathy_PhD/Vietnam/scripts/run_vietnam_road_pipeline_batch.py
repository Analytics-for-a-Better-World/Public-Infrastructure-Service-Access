from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def clock_ms(seconds: float) -> str:
    millis = int(round(seconds * 1000.0))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def run_case(
    *,
    python: Path,
    wrapper: Path,
    pipeline_dir: Path,
    fresh_root: Path,
    output_root: Path,
    source_table: Path,
    network_profile: str,
    spacing_m: int,
    snap_m: int,
    max_total_dist_m: int,
    snap_components: str,
    aggregate_factor: int | None,
    matrix_target_chunk_size: int,
    force_recompute: bool,
) -> dict[str, object]:
    profile_tag = network_profile.replace("_", "-")
    case_id = f"vietnam_{profile_tag}_unsimplified_{spacing_m // 1000}km"
    log_dir = output_root / "pipeline_logs" / case_id
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = log_dir / f"{case_id}.pipeline.log"
    console_log = log_dir / f"{case_id}.console.log"

    cmd = [
        str(python),
        str(wrapper),
        "--pipeline-dir",
        str(pipeline_dir),
        "--fresh-base-root",
        str(fresh_root),
        "vietnam",
        "--log-file",
        str(pipeline_log),
        "--network-backend",
        "osmium",
        "--simplify-network",
        "false",
        "--network-profile",
        network_profile,
        "--diagnose-connectivity",
        "true",
        "--snap-components",
        snap_components,
        "--sources",
        "table",
        "candidates",
        "--destinations",
        "population",
        "--source-table",
        str(source_table),
        "--source-lon-column",
        "longitude",
        "--source-lat-column",
        "latitude",
        "--source-id-column",
        "TT",
        "--candidate-grid-spacing-m",
        str(spacing_m),
        "--candidate-max-snap-dist-m",
        str(snap_m),
        "--candidate-exclude-water",
        "false",
        "--max-total-dist",
        str(max_total_dist_m),
        "--matrix-output-mode",
        "split",
        "--matrix-shape",
        "sparse",
    ]
    if aggregate_factor is not None:
        cmd.extend(["--aggregate-factor", str(int(aggregate_factor))])
    if matrix_target_chunk_size:
        cmd.extend(["--sparse-target-chunk-size", str(int(matrix_target_chunk_size))])
    if force_recompute:
        cmd.append("--force-recompute")

    start = perf_counter()
    print(f"\n=== {case_id} ===", flush=True)
    print(" ".join(cmd), flush=True)
    with console_log.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(pipeline_dir),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = proc.wait()
    elapsed = perf_counter() - start
    row: dict[str, object] = {
        "case_id": case_id,
        "country": "Vietnam",
        "network_profile": network_profile,
        "simplify_network": False,
        "candidate_grid_spacing_m": spacing_m,
        "candidate_max_snap_dist_m": snap_m,
        "max_total_dist_m": max_total_dist_m,
        "snap_components": snap_components,
        "aggregate_factor": aggregate_factor,
        "matrix_target_chunk_size": matrix_target_chunk_size,
        "elapsed_seconds": elapsed,
        "elapsed_clock_ms": clock_ms(elapsed),
        "return_code": int(return_code),
        "pipeline_log": str(pipeline_log),
        "console_log": str(console_log),
    }
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "country",
        "network_profile",
        "simplify_network",
        "candidate_grid_spacing_m",
        "candidate_max_snap_dist_m",
        "max_total_dist_m",
        "snap_components",
        "aggregate_factor",
        "matrix_target_chunk_size",
        "elapsed_seconds",
        "elapsed_clock_ms",
        "return_code",
        "pipeline_log",
        "console_log",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path(r"C:\github\Public-Infrastructure-Service-Access\Research-Sandbox\general_distances_per_country"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            r"C:\github\Public-Infrastructure-Service-Access\Research-Sandbox\general_distances_per_country\.venv\Scripts\python.exe"
        ),
    )
    parser.add_argument(
        "--fresh-root",
        type=Path,
        default=Path("runs/vietnam_road_20260623"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/vietnam_road_20260623"),
    )
    parser.add_argument(
        "--source-table",
        type=Path,
        default=Path("reference_cache/data/vietnam/vietnam_stroke_centers_130_en_source.xlsx"),
    )
    parser.add_argument("--network-profile", choices=["driving", "driving_walk"], default="driving")
    parser.add_argument("--spacings", type=int, nargs="+", default=[10000, 5000])
    parser.add_argument(
        "--snap-distance-ratio",
        type=float,
        default=0.5,
        help="Candidate max snap distance as a fraction of grid spacing.",
    )
    parser.add_argument("--max-total-dist-m", type=int, default=150000)
    parser.add_argument("--snap-components", default="0,1")
    parser.add_argument("--aggregate-factor", type=int, default=None)
    parser.add_argument("--matrix-target-chunk-size", type=int, default=0)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    wrapper = root / "tools" / "run_pipeline_fresh_root.py"
    output_root = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    fresh_root = (root / args.fresh_root).resolve() if not args.fresh_root.is_absolute() else args.fresh_root
    source_table = (root / args.source_table).resolve() if not args.source_table.is_absolute() else args.source_table
    output_root.mkdir(parents=True, exist_ok=True)
    fresh_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for spacing_m in args.spacings:
        snap_m = max(1, int(round(float(spacing_m) * float(args.snap_distance_ratio))))
        row = run_case(
            python=args.python,
            wrapper=wrapper,
            pipeline_dir=args.pipeline_dir,
            fresh_root=fresh_root,
            output_root=output_root,
            source_table=source_table,
            network_profile=str(args.network_profile),
            spacing_m=int(spacing_m),
            snap_m=snap_m,
            max_total_dist_m=int(args.max_total_dist_m),
            snap_components=str(args.snap_components),
            aggregate_factor=args.aggregate_factor,
            matrix_target_chunk_size=int(args.matrix_target_chunk_size),
            force_recompute=bool(args.force_recompute),
        )
        rows.append(row)
        write_csv(output_root / "vietnam_road_pipeline_case_timings.csv", rows)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "pipeline_dir": str(args.pipeline_dir),
        "fresh_root": str(fresh_root),
        "output_root": str(output_root),
        "source_table": str(source_table),
        "network_profile": str(args.network_profile),
        "spacings": [int(value) for value in args.spacings],
        "snap_distance_ratio": float(args.snap_distance_ratio),
        "max_total_dist_m": int(args.max_total_dist_m),
        "snap_components": str(args.snap_components),
        "aggregate_factor": args.aggregate_factor,
        "matrix_target_chunk_size": int(args.matrix_target_chunk_size),
        "force_recompute": bool(args.force_recompute),
        "timings_csv": str(output_root / "vietnam_road_pipeline_case_timings.csv"),
    }
    (output_root / "vietnam_road_pipeline_batch_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
