from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_timor_network_profiles_abw import (  # noqa: E402
    build_instance_from_manifest,
    clock_ms,
    load_manifests,
)

from abw_maxcover import (  # noqa: E402
    GurobiConfig,
    HeuristicConfig,
    approximate_pareto_curve,
    exact_pareto_curve,
)


def read_saturation(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["case_id"]: row for row in rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def curve_rows(
    *,
    case_id: str,
    curve_kind: str,
    stats: dict[str, Any],
    saturation: dict[str, Any],
    curve: Any,
    weight_scale: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in curve.results:
        objective = None if result.objective is None else int(result.objective)
        total_covered = None if objective is None else int(stats["baseline_weight"]) + objective
        total_seconds = None if result.total_seconds is None else float(result.total_seconds)
        rows.append(
            {
                "case_id": case_id,
                "curve_kind": curve_kind,
                "network_profile": stats["network_profile"],
                "simplify_network": stats["simplify_network"],
                "candidate_grid_spacing_m": stats["candidate_grid_spacing_m"],
                "budget": int(result.budget),
                "method": result.method,
                "status": result.status,
                "objective_population": None if objective is None else objective / weight_scale,
                "total_covered_population": None if total_covered is None else total_covered / weight_scale,
                "coverage_percent_total_population": None
                if total_covered is None
                else 100.0 * total_covered / int(stats["total_weight"]),
                "selected_count": len(result.solution),
                "solve_seconds": result.solve_seconds,
                "solve_clock_ms": clock_ms(result.solve_seconds),
                "total_seconds": total_seconds,
                "total_clock_ms": clock_ms(total_seconds),
                "construction_seconds": result.construction_seconds,
                "construction_clock_ms": clock_ms(result.construction_seconds),
                "local_search_moves": result.local_search_moves,
                "exact_saturation_budget": number(saturation.get("exact_saturation_budget")),
                "greedy_saturation_budget": number(saturation.get("greedy_saturation_budget")),
                "all_candidates_coverage_percent": stats["all_candidates_coverage_percent"],
                "n_candidates": stats["n_candidate_sources_in_matrix"],
                "n_arcs": stats["n_candidate_arcs_after_existing_coverage_removed"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("runs/timor_network_profile_20260623/east-timor_data/outputs"),
    )
    parser.add_argument(
        "--saturation-csv",
        type=Path,
        default=Path("outputs/timor_saturation_20260625/timor_primary_saturation_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/timor_saturation_20260625"))
    parser.add_argument("--weight-scale", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--skip-approx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "timor_primary_curves_to_saturation.csv"
    saturation_by_case = read_saturation(args.saturation_csv)
    manifests = load_manifests(args.outputs_dir)
    rows: list[dict[str, Any]] = []
    if args.skip_exact and output_csv.exists():
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if str(row.get("curve_kind", "")) != "approx_regreedy"
            ]
    try:
        import gurobipy as gb

        gb.setParam("Threads", int(args.threads))
        gb.setParam("OutputFlag", 0)
    except Exception:
        pass

    for case_id, saturation in saturation_by_case.items():
        exact_budget = number(saturation.get("exact_saturation_budget"))
        if exact_budget is None:
            continue
        greedy_budget = number(saturation.get("greedy_saturation_budget")) or exact_budget
        manifest_path, manifest = manifests[case_id]
        print(f"Building {case_id}", flush=True)
        instance, stats, _, _ = build_instance_from_manifest(
            case_id,
            manifest_path,
            manifest,
            weight_scale=int(args.weight_scale),
        )
        exact_budgets = list(range(0, int(exact_budget) + 1))
        approx_budgets = list(range(0, int(greedy_budget) + 1))
        if not args.skip_exact:
            print(f"  exact budgets 0..{int(exact_budget)}", flush=True)
            exact_curve = exact_pareto_curve(
                instance,
                exact_budgets,
                solver="gurobi",
                gurobi_config=GurobiConfig(
                    time_limit_seconds=float(args.time_limit),
                    mip_gap=1e-8,
                    trace=False,
                    warm_start=True,
                ),
            )
            rows.extend(
                curve_rows(
                    case_id=case_id,
                    curve_kind="exact",
                    stats=stats,
                    saturation=saturation,
                    curve=exact_curve,
                    weight_scale=int(args.weight_scale),
                )
            )
            write_csv(output_csv, rows)
        if not args.skip_approx:
            print(f"  approximate regreedy budgets 0..{int(greedy_budget)}", flush=True)
            approx_curve = approximate_pareto_curve(
                instance,
                approx_budgets,
                config=HeuristicConfig(
                    constructors=("regreedy",),
                    randomized_repeats=0,
                    use_path_relinking=False,
                    seed=42,
                ),
                select_best=True,
            )
            rows.extend(
                curve_rows(
                    case_id=case_id,
                    curve_kind="approx_regreedy",
                    stats=stats,
                    saturation=saturation,
                    curve=approx_curve,
                    weight_scale=int(args.weight_scale),
                )
            )
            write_csv(output_csv, rows)
    write_csv(output_csv, rows)


if __name__ == "__main__":
    main()
