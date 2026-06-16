# Fleur-Style Fast Replication

Run date: 7 June 2026.

This run repeats the Fleur-style analysis on fresh PISA Vietnam instances using the faster Vietnam-local algorithms:

- `--local-search first_sparse`
- `--path-relinking-method fast`

No Fleur `.npy`, pickle, or notebook-local data were used.

## Command

```powershell
py scripts\run_vietnam_fleur_style_analysis.py `
  --instances `
    C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_20km_threshold.npz `
    C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_50km_threshold.npz `
    C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_100km_threshold.npz `
  --budgets 20 40 60 80 100 200 `
  --local-search-budgets 20 40 60 80 100 200 `
  --randomized-budgets 20 40 60 80 `
  --randomized-repeats 3 `
  --grasp-max-iterations 5 `
  --grasp-time-limit-seconds 120 `
  --local-search first_sparse `
  --path-relinking-method fast `
  --output-dir C:\local\Parvathy\Vietnam\fleur_style_fast_replication
```

## Output Folder

```text
C:\local\Parvathy\Vietnam\fleur_style_fast_replication
```

Key outputs:

- `coverage_summary_by_budget.csv`
- `instance_statistics.csv`
- `selected_candidates.csv`
- `coverage_by_budget.png`
- `greedy_marginal_gains.png`
- threshold-specific greedy, local-search, and randomized GRASP traces

## Instance Baselines

| Threshold | Existing coverage | Incremental coverage still available | Candidate sites |
|---:|---:|---:|---:|
| 20 km | 54.62% | 45.38% | 3,126 |
| 50 km | 83.99% | 16.01% | 3,126 |
| 100 km | 94.50% | 5.50% | 3,126 |

This reproduces the central thesis pattern: the larger the service-distance threshold, the more Vietnam is already covered by existing facilities, and the smaller the remaining incremental opportunity for new candidates.

## Headline Results

All totals include existing-facility baseline coverage plus incremental candidate coverage.

| Threshold | Budget | Greedy total | Greedy + sparse local search | Best short GRASP + fast PR | Best method |
|---:|---:|---:|---:|---:|---|
| 20 km | 20 | 63.278M | 63.278M | 63.289M | GRASP + PR |
| 20 km | 40 | 67.782M | 67.782M | 67.770M | greedy |
| 20 km | 60 | 71.241M | 71.274M | 71.273M | local search |
| 20 km | 80 | 74.031M | 74.073M | 74.191M | GRASP + PR |
| 20 km | 100 | 76.386M | 76.428M | not run | local search |
| 20 km | 200 | 83.548M | 83.760M | not run | local search |
| 50 km | 20 | 90.355M | 90.387M | 90.371M | local search |
| 50 km | 40 | 93.186M | 93.214M | 93.250M | GRASP + PR |
| 50 km | 60 | 94.754M | 94.818M | 94.901M | GRASP + PR |
| 50 km | 80 | 95.623M | 95.758M | 95.841M | GRASP + PR |
| 50 km | 100 | 96.309M | 96.438M | not run | local search |
| 50 km | 200 | 97.581M | 97.658M | not run | local search |
| 100 km | 20 | 97.398M | 97.433M | 97.476M | GRASP + PR |
| 100 km | 40 | 97.708M | 97.726M | 97.735M | GRASP + PR |
| 100 km | 60 | 97.793M | 97.809M | 97.811M | GRASP + PR |
| 100 km | 80 | 97.838M | 97.842M | 97.844M | GRASP + PR |
| 100 km | 100 | 97.860M | 97.862M | not run | local search |
| 100 km | 200 | 97.896M | 97.896M | not run | greedy/local tie |

## Replicated Findings

The fresh-data fast run reproduces the main qualitative findings from Fleur's analysis:

- Greedy construction is a strong baseline and gives smooth diminishing-return curves.
- First-swap local search improves greedy, but the improvements are usually modest relative to total coverage.
- Randomized GRASP plus path relinking can beat greedy plus local search, especially at 50 km p=40/60/80 and 100 km p=20/40/60/80.
- Coverage saturates rapidly at 100 km because existing facilities already cover about 94.50% of population.
- The largest practical remaining room for improvement is the 20 km setting, where p=200 still covers only about 84.27% after local search.

## Method Notes

The randomized GRASP results are short-run checks: 3 repeats, 5 iterations per repeat, and randomized budgets 20/40/60/80 only. They are enough to reproduce the thesis-style pattern, but they are not final tuned stochastic experiments.

Exact Gurobi comparisons remain a planned extension. For the fresh national instances, exact runs should probably start from restricted slices or candidate subsets and use greedy/GRASP solutions as warm starts.
