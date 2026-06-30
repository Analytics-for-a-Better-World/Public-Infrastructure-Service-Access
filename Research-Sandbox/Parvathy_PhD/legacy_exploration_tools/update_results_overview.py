from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "grid_experiment_report"
TIMOR_DIR = ROOT / "outputs" / "timor_abw_maxcover_rerun_gdc_venv"
REPORT_PATH = REPORT_DIR / "combined_timor_vietnam_grid_report.md"
MANIFEST_PATH = REPORT_DIR / "combined_grid_report_manifest.json"


def fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], *, digits: int = 3) -> str:
    if not rows:
        return "_No rows._"
    headers = [header for header, _ in columns]
    keys = [key for _, key in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        rendered = []
        for key in keys:
            value = row.get(key)
            if key.endswith("_percent"):
                rendered.append(fmt(value, digits, "%"))
            elif key.endswith("_pp") or "gap" in key or "gain" in key or "minus" in key:
                rendered.append(fmt(value, digits))
            elif "seconds" in key:
                rendered.append(fmt(value, digits))
            else:
                rendered.append(fmt(value, digits))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def extract_vietnam_sections(existing_report: str) -> str:
    marker = "## Vietnam 10 km Selected p Metrics"
    index = existing_report.find(marker)
    if index < 0:
        return ""
    return existing_report[index:].strip()


def row_by_grid_budget(frame: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
    return {
        (str(row["grid"]), int(row["budget"])): row
        for _, row in frame.iterrows()
    }


def build_timor_tables() -> dict[str, list[dict[str, Any]]]:
    stats = pd.read_csv(TIMOR_DIR / "timor_abw_exact_stats.csv")
    exact = pd.read_csv(TIMOR_DIR / "timor_abw_exact_curves.csv")
    heuristic = pd.read_csv(TIMOR_DIR / "timor_abw_heuristics_selected_best.csv")
    selected = pd.read_csv(TIMOR_DIR / "timor_abw_selected_budget_comparison.csv")

    exact_lookup = row_by_grid_budget(exact)
    heuristic_lookup = row_by_grid_budget(heuristic)
    selected_budgets = sorted(int(value) for value in selected["budget"].unique())
    grids = ["10 km", "5 km", "1 km"]

    summary_rows = []
    for _, row in stats.iterrows():
        summary_rows.append(
            {
                "grid": row["grid"],
                "n_candidates": int(row["n_candidates"]),
                "candidate_edge_rows": int(row["candidate_edge_rows"]),
                "all_candidate_coverage_percent": float(row["all_candidate_coverage_percent"]),
                "exact_saturation_budget": int(row["exact_saturation_budget"]),
                "exact_total_seconds": float(row["exact_total_seconds"]),
            }
        )

    selected_rows = []
    for _, row in selected.iterrows():
        selected_rows.append(
            {
                "p": int(row["budget"]),
                "exact_10km_percent": float(row["timor_10km_exact_coverage_percent"]),
                "exact_5km_percent": float(row["timor_5km_exact_coverage_percent"]),
                "exact_1km_percent": float(row["timor_1km_exact_coverage_percent"]),
                "gain_10_to_5_pp": float(row["gain_10km_to_5km_percentage_points"]),
                "gain_5_to_1_pp": float(row["gain_5km_to_1km_percentage_points"]),
                "heuristic_1km_percent": float(row["timor_1km_best_heuristic_coverage_percent"]),
                "exact_minus_heuristic_1km_pp": float(row["timor_1km_exact_minus_heuristic_percentage_points"]),
                "heuristic_1km_method": row["timor_1km_best_heuristic_method"],
                "exact_1km_seconds": float(row["timor_1km_exact_seconds"]),
                "heuristic_1km_seconds": float(row["timor_1km_best_heuristic_seconds"]),
            }
        )

    quality_rows = []
    for grid in grids:
        for budget in selected_budgets:
            exact_row = exact_lookup[(grid, budget)]
            heuristic_row = heuristic_lookup[(grid, budget)]
            quality_rows.append(
                {
                    "grid": grid,
                    "p": budget,
                    "exact_percent": float(exact_row["coverage_percent"]),
                    "heuristic_percent": float(heuristic_row["coverage_percent"]),
                    "exact_minus_heuristic_pp": float(exact_row["coverage_percent"])
                    - float(heuristic_row["coverage_percent"]),
                    "heuristic_method": heuristic_row["method"],
                    "exact_seconds": float(exact_row["solve_seconds"]),
                    "heuristic_seconds": float(heuristic_row["total_seconds"]),
                }
            )

    fleur_rows = []
    for budget in selected_budgets:
        exact_10 = exact_lookup[("10 km", budget)]
        exact_5 = exact_lookup[("5 km", budget)]
        exact_1 = exact_lookup[("1 km", budget)]
        heuristic_5 = heuristic_lookup[("5 km", budget)]
        heuristic_1 = heuristic_lookup[("1 km", budget)]
        fleur_rows.append(
            {
                "p": budget,
                "exact_10km_percent": float(exact_10["coverage_percent"]),
                "heuristic_5km_percent": float(heuristic_5["coverage_percent"]),
                "heuristic_5km_minus_exact_10km_pp": float(heuristic_5["coverage_percent"])
                - float(exact_10["coverage_percent"]),
                "exact_5km_minus_heuristic_5km_pp": float(exact_5["coverage_percent"])
                - float(heuristic_5["coverage_percent"]),
                "heuristic_5km_method": heuristic_5["method"],
                "exact_5km_percent": float(exact_5["coverage_percent"]),
                "heuristic_1km_percent": float(heuristic_1["coverage_percent"]),
                "heuristic_1km_minus_exact_5km_pp": float(heuristic_1["coverage_percent"])
                - float(exact_5["coverage_percent"]),
                "exact_1km_minus_heuristic_1km_pp": float(exact_1["coverage_percent"])
                - float(heuristic_1["coverage_percent"]),
                "heuristic_1km_method": heuristic_1["method"],
            }
        )

    return {
        "summary": summary_rows,
        "selected": selected_rows,
        "quality": quality_rows,
        "fleur": fleur_rows,
    }


def write_csv_outputs(tables: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "timor_abw_exact_summary": REPORT_DIR / "timor_abw_exact_summary.csv",
        "timor_abw_selected_p_10_5_1_gain": REPORT_DIR / "timor_abw_selected_p_10_5_1_gain.csv",
        "timor_abw_heuristic_quality_selected_p": REPORT_DIR / "timor_abw_heuristic_quality_selected_p.csv",
        "timor_abw_fleur_style_richer_grid_comparison": REPORT_DIR
        / "timor_abw_fleur_style_richer_grid_comparison.csv",
    }
    pd.DataFrame(tables["summary"]).to_csv(outputs["timor_abw_exact_summary"], index=False)
    pd.DataFrame(tables["selected"]).to_csv(outputs["timor_abw_selected_p_10_5_1_gain"], index=False)
    pd.DataFrame(tables["quality"]).to_csv(outputs["timor_abw_heuristic_quality_selected_p"], index=False)
    pd.DataFrame(tables["fleur"]).to_csv(outputs["timor_abw_fleur_style_richer_grid_comparison"], index=False)
    return {key: str(path) for key, path in outputs.items()}


def build_report(tables: dict[str, list[dict[str, Any]]], vietnam_sections: str) -> str:
    fleur = pd.DataFrame(tables["fleur"])
    quality = pd.DataFrame(tables["quality"])
    max_1km_gap = quality.loc[quality["grid"] == "1 km", "exact_minus_heuristic_pp"].max()
    max_5km_gap = quality.loc[quality["grid"] == "5 km", "exact_minus_heuristic_pp"].max()
    min_5_over_10 = fleur["heuristic_5km_minus_exact_10km_pp"].min()
    min_1_over_5 = fleur["heuristic_1km_minus_exact_5km_pp"].min()

    timor_text = "\n".join(
        [
            "# Timor/Vietnam Grid Experiment Report",
            "",
            "Timor-Leste is reported as projected straight-line 5 km screening. "
            "The latest Timor rerun used the `general_distances_per_country` Python 3.14 venv "
            "and the new local `abw_maxcover` package. Exact optima are certified for the "
            "straight-line 10 km, 5 km, and 1 km Timor instances; heuristic rows are therefore "
            "used not as a substitute for optimality, but to assess heuristic quality and to "
            "recreate the Fleur-style question of whether a richer-grid heuristic can beat a "
            "coarser-grid optimum.",
            "",
            "Vietnam sections are retained from the prior grid overview: 10 km network-distance "
            "tradeoff comparisons, and dense projected straight-line 1 km heuristic versus 5 km "
            "exact comparisons.",
            "",
            "## Timor Exact Instance Summary",
            "",
            markdown_table(
                tables["summary"],
                [
                    ("Grid", "grid"),
                    ("Candidates", "n_candidates"),
                    ("Edges", "candidate_edge_rows"),
                    ("All-Candidate Coverage", "all_candidate_coverage_percent"),
                    ("Saturation p", "exact_saturation_budget"),
                    ("Exact Total Seconds", "exact_total_seconds"),
                ],
            ),
            "",
            "## Timor Selected p Exact Gains",
            "",
            markdown_table(
                tables["selected"],
                [
                    ("p", "p"),
                    ("10 km Exact", "exact_10km_percent"),
                    ("5 km Exact", "exact_5km_percent"),
                    ("1 km Exact", "exact_1km_percent"),
                    ("10 to 5 Gain pp", "gain_10_to_5_pp"),
                    ("5 to 1 Gain pp", "gain_5_to_1_pp"),
                    ("1 km Heuristic Best", "heuristic_1km_percent"),
                    ("Exact-Heuristic pp", "exact_minus_heuristic_1km_pp"),
                    ("1 km Method", "heuristic_1km_method"),
                ],
            ),
            "",
            "## Timor Fleur-Style Richer-Grid Heuristic Comparison",
            "",
            "This table mirrors the Fleur-style comparison: a heuristic on a richer grid is "
            "placed next to the exact optimum on a coarser grid. Because Timor 10/5/1 km are "
            "all solved exactly, the table also reports the same-grid optimality gap for each "
            "heuristic. Positive `Heur 5 - Opt 10` and `Heur 1 - Opt 5` values mean the richer "
            "heuristic beats the coarser optimum at the same p.",
            "",
            markdown_table(
                tables["fleur"],
                [
                    ("p", "p"),
                    ("Opt 10 km", "exact_10km_percent"),
                    ("Heur 5 km", "heuristic_5km_percent"),
                    ("Heur 5 - Opt 10 pp", "heuristic_5km_minus_exact_10km_pp"),
                    ("Opt 5 - Heur 5 pp", "exact_5km_minus_heuristic_5km_pp"),
                    ("5 km Method", "heuristic_5km_method"),
                    ("Opt 5 km", "exact_5km_percent"),
                    ("Heur 1 km", "heuristic_1km_percent"),
                    ("Heur 1 - Opt 5 pp", "heuristic_1km_minus_exact_5km_pp"),
                    ("Opt 1 - Heur 1 pp", "exact_1km_minus_heuristic_1km_pp"),
                    ("1 km Method", "heuristic_1km_method"),
                ],
            ),
            "",
            "Takeaway: the richer-grid heuristic dominates the coarser optimum in these Timor "
            f"selected p values. The smallest 5 km heuristic gain over the 10 km optimum is "
            f"{min_5_over_10:.3f} percentage points; the smallest 1 km heuristic gain over the "
            f"5 km optimum is {min_1_over_5:.3f} percentage points. Same-grid heuristic quality "
            f"is tight: max selected-p gap is {max_5km_gap:.3f} pp on 5 km and "
            f"{max_1km_gap:.3f} pp on 1 km.",
            "",
            "## Timor Heuristic Quality Against Certified Optima",
            "",
            markdown_table(
                tables["quality"],
                [
                    ("Grid", "grid"),
                    ("p", "p"),
                    ("Exact", "exact_percent"),
                    ("Heuristic", "heuristic_percent"),
                    ("Exact-Heuristic pp", "exact_minus_heuristic_pp"),
                    ("Method", "heuristic_method"),
                    ("Exact Seconds", "exact_seconds"),
                    ("Heuristic Seconds", "heuristic_seconds"),
                ],
            ),
            "",
        ]
    )
    return timor_text + "\n" + vietnam_sections.strip() + "\n"


def main() -> None:
    existing_report = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    vietnam_sections = extract_vietnam_sections(existing_report)
    tables = build_timor_tables()
    output_paths = write_csv_outputs(tables)
    REPORT_PATH.write_text(build_report(tables, vietnam_sections), encoding="utf-8")

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.setdefault("inputs", {})
    manifest.setdefault("outputs", {})
    manifest["inputs"]["timor_abw_latest_dir"] = str(TIMOR_DIR)
    manifest["inputs"]["timor_abw_python"] = (
        r"C:\github\Public-Infrastructure-Service-Access\Research-Sandbox"
        r"\general_distances_per_country\.venv\Scripts\python.exe"
    )
    manifest["outputs"]["report"] = str(REPORT_PATH)
    manifest["outputs"].update(output_paths)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
