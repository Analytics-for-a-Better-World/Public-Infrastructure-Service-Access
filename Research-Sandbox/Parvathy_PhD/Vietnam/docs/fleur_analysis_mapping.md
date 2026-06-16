# Fleur-Style Analysis Mapping

This document records how the fresh Vietnam replication mimics Fleur Theulen's thesis analysis without using the old `.npy`, pickle, CSV, or notebook-local data files.

## What Fleur's Analysis Did

The recovered notebooks and helper scripts point to this workflow:

1. Set a country, grid size, and distance threshold.
2. Load existing facilities, population demand, and potential locations.
3. Build sparse row/column coverage dictionaries from thresholded distances.
4. Compute the population already covered by existing facilities.
5. Compare constructive solutions:
   - k-means-inspired construction;
   - greedy construction;
   - randomized greedy / restricted candidate list construction.
6. Improve solutions with 1-for-1 swap local search.
7. Repeat randomized constructions and compare best/average outcomes.
8. Where possible, compare heuristic starts with exact Gurobi runs and solver gaps.
9. Save tables, traces, and plots for objective values, runtime, marginal gains, and local-search improvement.

## Fresh Replication Equivalent

The local Vietnam replication keeps the same analytical structure but replaces the data layer:

| Fleur-era element | Fresh replication element |
|---|---|
| `stroke-facs.csv` | current stroke table read by the PISA pipeline |
| `pop_2020_1km.csv` / `.npy` population arrays | fresh PISA population parquet |
| `potential_location_grid_*.geojson` | PISA candidate-grid source layer |
| road preprocessing notebooks | PISA OSM/Pandana distance pipeline |
| row/column pickle dictionaries | CSR `.npz` max-cover instances |
| existing-hospital coverage subtraction | `baseline_covered_mask` and zeroed incremental weights |
| greedy notebook cells | `budgeted_construct(..., constructor="greedy")` |
| 1-swap local search notebooks | Vietnam-local `first_sparse` first-improvement swap local search, with the original `approximated_tradeoff` path kept for validation |
| randomized greedy / article experiments | randomized GRASP with RCL size and repeated seeds |
| path relinking sketches | implemented forward/backward path relinking with an elite pool |

The script that produces these artifacts is:

```text
scripts\run_vietnam_fleur_style_analysis.py
```

## Output Families

The analysis runner writes:

- `instance_statistics.csv`: baseline coverage, candidate count, retained matrix rows, and available incremental population;
- `coverage_summary_by_budget.csv`: greedy, greedy+local-search, and randomized GRASP results by threshold and budget;
- `*_greedy_marginal_trace.csv`: Fleur-style marginal gains of the greedy construction;
- `*_first_sparse_trace.csv`: local-search improvement trace by accepted swap;
- `*_randomized_first_sparse_fast_repeat*_trace.csv`: GRASP iteration traces;
- `selected_candidates.csv`: selected fresh candidate IDs and coordinates;
- `coverage_by_budget.png`: objective curves over budgets;
- `greedy_marginal_gains.png`: marginal gain decay curves;
- `analysis_manifest.json`: provenance and parameter settings.

## Exact Solver Comparison

Fleur's notebooks also compared heuristic starts and Gurobi progress logs. This replication keeps that comparison as a planned extension rather than a default run because the fresh national instances are much larger than the old notebook instances. The recommended next step is to add an optional exact run on restricted slices or smaller candidate grids, then append lower/upper bounds and gaps to `coverage_summary_by_budget.csv`.
