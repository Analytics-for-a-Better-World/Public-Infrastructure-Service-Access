from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create lightweight solution maps from Vietnam dense-grid analysis outputs."
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--run-tag-marker", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 80, 200])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population-sample", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match for {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("longitude", "latitude"),
        ("Longitude", "Latitude"),
        ("xcoord", "ycoord"),
        ("lon", "lat"),
    ]
    for lon, lat in candidates:
        if lon in df.columns and lat in df.columns:
            return lon, lat
    raise KeyError(f"No supported coordinate columns in {list(df.columns)}")


def method_slug(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or "method"


def load_background(outputs_dir: Path, marker: str, sample_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    population_path = find_one(outputs_dir, f"population_*{marker}*.parquet")
    sources_path = find_one(outputs_dir, f"sources_*{marker}*.parquet")
    population = pd.read_parquet(population_path)
    if len(population) > sample_size:
        population = population.sample(sample_size, random_state=seed)
    sources = pd.read_parquet(sources_path)
    source_type = sources.get("source_type", pd.Series("", index=sources.index)).astype(str)
    existing = sources[source_type.isin(["table", "existing"])].copy()
    return population, existing


def choose_best_rows(summary: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    keep = summary[summary["budget"].isin(budgets)].copy()
    if keep.empty:
        return keep
    keep = keep.sort_values(
        ["threshold_km", "budget", "total_covered_population", "seconds"],
        ascending=[True, True, False, True],
    )
    return keep.groupby(["threshold_km", "budget"], as_index=False).head(1)


def selected_for_best(selected: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    mask = (
        np.isclose(selected["threshold_km"].astype(float), float(best["threshold_km"]))
        & (selected["budget"].astype(int) == int(best["budget"]))
        & (selected["method"].astype(str) == str(best["method"]))
    )
    if "instance" in selected.columns and "instance" in best.index:
        mask &= selected["instance"].astype(str) == str(best["instance"])
    if pd.notna(best.get("seed")) and "seed" in selected.columns:
        mask &= selected["seed"].fillna(-1).astype(float) == float(best["seed"])
    if pd.notna(best.get("repeat")) and "repeat" in selected.columns:
        mask &= selected["repeat"].fillna(-1).astype(float) == float(best["repeat"])
    return selected[mask].sort_values("rank")


def write_map(
    *,
    population: pd.DataFrame,
    existing: pd.DataFrame,
    chosen: pd.DataFrame,
    best: pd.Series,
    label: str,
    outpath: Path,
) -> None:
    import matplotlib.pyplot as plt

    pop_lon, pop_lat = coordinate_columns(population)
    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    pop_size = None
    if "population" in population.columns:
        values = population["population"].to_numpy(dtype=float)
        pop_size = 1.5 + 12.0 * np.sqrt(values / max(values.max(), 1.0))
    ax.scatter(
        population[pop_lon],
        population[pop_lat],
        s=pop_size if pop_size is not None else 3,
        c="#d4d4d4",
        alpha=0.35,
        linewidths=0,
        label="population sample",
    )

    if not existing.empty:
        ex_lon, ex_lat = coordinate_columns(existing)
        ax.scatter(
            existing[ex_lon],
            existing[ex_lat],
            s=18,
            c="#111111",
            alpha=0.75,
            marker="x",
            linewidths=0.8,
            label="existing stroke facilities",
        )

    ax.scatter(
        chosen["longitude"],
        chosen["latitude"],
        s=28,
        c="#d62728",
        edgecolors="white",
        linewidths=0.5,
        label="selected candidates",
        zorder=5,
    )

    threshold = float(best["threshold_km"])
    budget = int(best["budget"])
    covered_m = float(best["total_covered_population"]) / 1_000_000.0
    seconds = float(best["seconds"])
    method = str(best["method"])
    ax.set_title(
        f"{label}: {threshold:g} km, p={budget}\n"
        f"{method} | {covered_m:.2f}M covered | {seconds:.1f}s",
        fontsize=11,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.analysis_dir / "coverage_summary_by_budget.csv")
    selected = pd.read_csv(args.analysis_dir / "selected_candidates.csv")
    population, existing = load_background(
        args.outputs_dir,
        args.run_tag_marker,
        args.population_sample,
        args.seed,
    )

    rows = []
    for _, best in choose_best_rows(summary, args.budgets).iterrows():
        chosen = selected_for_best(selected, best)
        if chosen.empty:
            rows.append({**best.to_dict(), "figure": "", "status": "missing selected candidates"})
            continue
        threshold = float(best["threshold_km"])
        budget = int(best["budget"])
        filename = (
            f"solution_{method_slug(args.label)}_"
            f"{threshold:g}km_p{budget}_{method_slug(best['method'])}.png"
        )
        outpath = args.output_dir / filename
        write_map(
            population=population,
            existing=existing,
            chosen=chosen,
            best=best,
            label=args.label,
            outpath=outpath,
        )
        rows.append({**best.to_dict(), "figure": str(outpath), "status": "ok"})

    pd.DataFrame(rows).to_csv(args.output_dir / f"{method_slug(args.label)}_solution_figure_index.csv", index=False)
    print({"figures": len(rows), "output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
