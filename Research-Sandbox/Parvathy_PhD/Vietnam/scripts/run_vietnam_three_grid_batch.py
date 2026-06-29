from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n>>> " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
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
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--fresh-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, type=Path)
    parser.add_argument("--run-tag-marker", default="maxdist_150000")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    vietnam_dir = args.repo_root / "Research-Sandbox" / "Parvathy_PhD" / "Vietnam"
    scripts_dir = vietnam_dir / "scripts"
    outputs_dir = args.fresh_root / "vietnam_data" / "outputs"
    logs_dir = args.output_root / "logs"
    opt_dir = args.output_root / "optimization"
    analysis_10_dir = args.output_root / "fleur_style_10km_network"
    dense_root = args.output_root / "dense_grid_straightline"
    opt_dir.mkdir(parents=True, exist_ok=True)

    instances = []
    for threshold_km in (20, 50, 100):
        out = opt_dir / f"vietnam_10kmgrid_{threshold_km}km_threshold.npz"
        instances.append(out)
        run(
            [
                str(args.python),
                str(scripts_dir / "build_pisa_instance.py"),
                "--outputs-dir",
                str(outputs_dir),
                "--run-tag-marker",
                args.run_tag_marker,
                "--threshold-m",
                str(threshold_km * 1000),
                "--output-npz",
                str(out),
            ],
            cwd=vietnam_dir,
            log_path=logs_dir / f"build_instance_{threshold_km}km.log",
        )

    run(
        [
            str(args.python),
            str(scripts_dir / "run_vietnam_fleur_style_analysis.py"),
            "--instances",
            *(str(path) for path in instances),
            "--budgets",
            "20",
            "40",
            "60",
            "80",
            "100",
            "200",
            "--local-search-budgets",
            "20",
            "40",
            "60",
            "80",
            "100",
            "200",
            "--randomized-budgets",
            "20",
            "40",
            "60",
            "80",
            "--randomized-repeats",
            "3",
            "--grasp-max-iterations",
            "5",
            "--grasp-time-limit-seconds",
            "120",
            "--local-search",
            "first_sparse",
            "--path-relinking-method",
            "fast",
            "--output-dir",
            str(analysis_10_dir),
        ],
        cwd=vietnam_dir,
        log_path=logs_dir / "fleur_style_10km_network.log",
    )

    if not args.skip_dense:
        grid_paths = {}
        for spacing in (5000, 1000):
            summary_json = dense_root / f"candidate_grid_{spacing}m_summary.json"
            run(
                [
                    str(args.python),
                    str(scripts_dir / "build_candidate_grid_only.py"),
                    "--base-root",
                    str(args.fresh_root),
                    "--candidate-grid-spacing-m",
                    str(spacing),
                    "--summary-json",
                    str(summary_json),
                ],
                cwd=vietnam_dir,
                log_path=logs_dir / f"candidate_grid_{spacing}m.log",
            )
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            grid_paths[spacing] = Path(summary["candidate_grid_path"])

        for spacing in (5000, 1000):
            run(
                [
                    str(args.python),
                    str(scripts_dir / "run_dense_grid_straightline_analysis.py"),
                    "--outputs-dir",
                    str(outputs_dir),
                    "--run-tag-marker",
                    args.run_tag_marker,
                    "--candidate-grid",
                    str(grid_paths[spacing]),
                    "--grid-spacing-m",
                    str(spacing),
                    "--thresholds-km",
                    "20",
                    "50",
                    "100",
                    "--budgets",
                    "20",
                    "80",
                    "200",
                    "--local-search-budgets",
                    "20",
                    "80",
                    "200",
                    "--randomized-budgets",
                    "20",
                    "--randomized-repeats",
                    "2",
                    "--output-dir",
                    str(dense_root / f"grid_{spacing}m"),
                    "--population-sample",
                    "30000",
                ],
                cwd=vietnam_dir,
                log_path=logs_dir / f"dense_grid_{spacing}m.log",
            )

    manifest = {
        "repo_root": str(args.repo_root),
        "fresh_root": str(args.fresh_root),
        "outputs_dir": str(outputs_dir),
        "output_root": str(args.output_root),
        "run_tag_marker": args.run_tag_marker,
        "network_10km_instances": [str(path) for path in instances],
        "analysis_10km_dir": str(analysis_10_dir),
        "dense_root": str(dense_root),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "vietnam_three_grid_batch_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
