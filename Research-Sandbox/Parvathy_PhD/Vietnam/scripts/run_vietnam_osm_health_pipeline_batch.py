from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from time import perf_counter


def default_wrapper_path(script_path: Path) -> Path:
    resolved = script_path.resolve()
    candidates = [
        # Sandbox layout: <workspace>/tools/<script>.py
        resolved.parents[1] / "tools" / "run_pipeline_fresh_root.py",
        # Repository layout: Research-Sandbox/Parvathy_PhD/Vietnam/scripts/<script>.py
        resolved.parents[2] / "shared" / "tools" / "run_pipeline_fresh_root.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
    network_profile: str,
    spacing_m: int,
    snap_m: int,
    max_total_dist_m: int,
    snap_components: str,
    aggregate_factor: int | None,
    matrix_target_chunk_size: int,
    include_candidates: bool,
    force_recompute: bool,
) -> dict[str, object]:
    profile_tag = network_profile.replace("_", "-")
    source_tag = "osm_health_candidates" if include_candidates else "osm_health"
    case_id = f"vietnam_{source_tag}_{profile_tag}_unsimplified_{spacing_m // 1000}km"
    log_dir = output_root / "pipeline_logs" / case_id
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = log_dir / f"{case_id}.pipeline.log"
    console_log = log_dir / f"{case_id}.console.log"

    sources = ["amenities", "candidates"] if include_candidates else ["amenities"]
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
        *sources,
        "--destinations",
        "population",
        "--amenity",
        "hospital",
        "clinic",
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
        "source_mode": source_tag,
        "amenity_values": "hospital;clinic",
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
        "source_mode",
        "amenity_values",
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
        "--wrapper",
        type=Path,
        default=None,
        help="Path to run_pipeline_fresh_root.py. If omitted, inferred from the sandbox or Parvathy_PhD repo layout.",
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
        default=Path("runs/vietnam_170_agg5_20260624_s20"),
        help="Defaults to the existing Vietnam 170 run root to reuse the same OSM PBF and WorldPop inputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/vietnam_osm_health_agg5_20260630_s20"),
    )
    parser.add_argument("--network-profile", choices=["driving", "driving_walk"], default="driving")
    parser.add_argument("--spacings", type=int, nargs="+", default=[10000, 5000, 1000])
    parser.add_argument("--snap-distance-ratio", type=float, default=0.5)
    parser.add_argument("--max-total-dist-m", type=int, default=20000)
    parser.add_argument("--snap-components", default="0,1")
    parser.add_argument("--aggregate-factor", type=int, default=5)
    parser.add_argument("--matrix-target-chunk-size", type=int, default=0)
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    wrapper = args.wrapper if args.wrapper is not None else default_wrapper_path(Path(__file__))
    output_root = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    fresh_root = (root / args.fresh_root).resolve() if not args.fresh_root.is_absolute() else args.fresh_root
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
            network_profile=str(args.network_profile),
            spacing_m=int(spacing_m),
            snap_m=snap_m,
            max_total_dist_m=int(args.max_total_dist_m),
            snap_components=str(args.snap_components),
            aggregate_factor=args.aggregate_factor,
            matrix_target_chunk_size=int(args.matrix_target_chunk_size),
            include_candidates=bool(args.include_candidates),
            force_recompute=bool(args.force_recompute),
        )
        rows.append(row)
        write_csv(output_root / "vietnam_osm_health_pipeline_case_timings.csv", rows)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "pipeline_dir": str(args.pipeline_dir),
        "fresh_root": str(fresh_root),
        "output_root": str(output_root),
        "network_profile": str(args.network_profile),
        "spacings": [int(value) for value in args.spacings],
        "snap_distance_ratio": float(args.snap_distance_ratio),
        "max_total_dist_m": int(args.max_total_dist_m),
        "snap_components": str(args.snap_components),
        "aggregate_factor": args.aggregate_factor,
        "matrix_target_chunk_size": int(args.matrix_target_chunk_size),
        "include_candidates": bool(args.include_candidates),
        "force_recompute": bool(args.force_recompute),
        "timings_csv": str(output_root / "vietnam_osm_health_pipeline_case_timings.csv"),
    }
    (output_root / "vietnam_osm_health_pipeline_batch_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
