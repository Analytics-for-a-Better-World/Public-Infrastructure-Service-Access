# Fleur-Style First-Pass Analysis

Run date: 7 June 2026.

This pass mimics Fleur's analysis shape on fresh PISA data. It is not yet the full thesis experiment grid.

## Command

```powershell
py scripts\run_vietnam_fleur_style_analysis.py `
  --instances `
    C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_20km_threshold.npz `
    C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_50km_threshold.npz `
    C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_100km_threshold.npz `
  --budgets 20 40 60 80 100 200 `
  --local-search-budgets 20 40 `
  --randomized-budgets 20 `
  --randomized-repeats 2 `
  --grasp-max-iterations 3 `
  --grasp-time-limit-seconds 120 `
  --output-dir C:\local\Parvathy\Vietnam\fleur_style_analysis
```

The GRASP time limit is checked between iterations. A single long local-search or path-relinking iteration can therefore exceed the nominal limit.

## Output Folder

```text
C:\local\Parvathy\Vietnam\fleur_style_analysis
```

Key files:

- `instance_statistics.csv`
- `coverage_summary_by_budget.csv`
- `selected_candidates.csv`
- `coverage_by_budget.png`
- `greedy_marginal_gains.png`
- threshold-specific greedy, local-search, and randomized traces

## Headline Results

All totals include the existing-facility baseline plus incremental candidate coverage.

| Threshold | Budget | Method | Total covered | Incremental | Coverage |
|---:|---:|---|---:|---:|---:|
| 20 km | 20 | greedy | 63.278M | 8.990M | 63.67% |
| 20 km | 20 | greedy + first swap | 63.278M | 8.990M | 63.67% |
| 20 km | 20 | best short randomized GRASP + PR | 63.289M | 9.000M | 63.68% |
| 20 km | 40 | greedy + first swap | 67.782M | 13.494M | 68.20% |
| 20 km | 200 | greedy | 83.548M | 29.259M | 84.06% |
| 50 km | 20 | greedy | 90.355M | 6.878M | 90.91% |
| 50 km | 20 | greedy + first swap | 90.387M | 6.909M | 90.94% |
| 50 km | 20 | best short randomized GRASP + PR | 90.369M | 6.891M | 90.92% |
| 50 km | 40 | greedy + first swap | 93.214M | 9.737M | 93.79% |
| 50 km | 200 | greedy | 97.581M | 14.104M | 98.18% |
| 100 km | 20 | greedy | 97.398M | 3.475M | 97.99% |
| 100 km | 20 | greedy + first swap | 97.433M | 3.510M | 98.03% |
| 100 km | 20 | best short randomized GRASP + PR | 97.475M | 3.553M | 98.07% |
| 100 km | 40 | greedy + first swap | 97.726M | 3.804M | 98.32% |
| 100 km | 200 | greedy | 97.896M | 3.973M | 98.50% |

## Reading

- At 20 km, greedy is extremely fast and short randomized GRASP finds a tiny improvement at p=20.
- At 50 km, first-swap local search improves greedy; the short randomized GRASP pass did not beat greedy+swap yet.
- At 100 km, randomized GRASP plus path relinking beats greedy+swap for p=20 in this first pass, but it is much slower.
- Greedy to p=200 is useful as a coverage curve baseline, but the local-search and randomized grids still need longer thesis-style sweeps.

## Remaining Thesis-Style Work

- Run local search for budgets 60, 80, 100, and 200 where runtime permits.
- Increase randomized repeats and seeds, especially for 50 km and 100 km.
- Add sample and random-plus constructors to the analysis comparison.
- Add optional exact-solver comparisons on restricted or smaller instances, with lower bound, upper bound, gap, status, and solve time.
