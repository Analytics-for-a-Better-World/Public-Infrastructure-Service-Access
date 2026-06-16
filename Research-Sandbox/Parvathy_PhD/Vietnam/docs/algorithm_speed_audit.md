# Algorithm Speed Audit

This note records the June 2026 speed pass over the Vietnam-local algorithms. It does not use Fleur's old `.npy` or pickle data, and it does not modify `Research-Sandbox\approximated_tradeoff`.

## Algorithms Checked

- `budgeted_construct`: greedy, randomized RCL, sample, and random-plus construction.
- `improve_local_search`: original `first`, sparse `first_sparse`, and `none`.
- `path_relink` and `path_relink_fast`.
- `run_grasp`: construction plus local search plus optional path relinking.
- Fleur-style batch analysis wrappers that call the above algorithms.

The old exact Pyomo/Gurobi helpers in `approximated_tradeoff\src\mc_solvers.py` were inspected as reference code, but the Vietnam replication runners currently use heuristics rather than exact MIP solves.

## Findings

Construction is not the limiting component on the current 10 km candidate-grid instances. In wall-clock checks, `budgeted_construct` stayed around 0.04-0.47 seconds on 20 km, 0.29-1.24 seconds on 50 km, and 0.62-2.54 seconds on 100 km across budgets 20, 100, and 200. The small constructor cleanup now avoids building `np.flatnonzero(gain > 0)` for pure greedy choices.

The original first-swap local search remains the main avoidable bottleneck. On the p=20 benchmark, `first_sparse` matched the original objective and selected solution exactly, while reducing the 100 km local-search wall time from 35.86 seconds to 1.42 seconds including sparse-index setup. The Vietnam runners therefore now default to `--local-search first_sparse`; `--local-search first` remains available for reference validation.

Larger-budget local search is now practical, but still worth watching. With `first_sparse`, greedy-start local search took about 0.74 seconds on 20 km p=200, 16.17 seconds on 50 km p=200, and 0.68 seconds on 100 km p=200. The 50 km p=200 case is heavier because it accepts many more improving swaps.

Path relinking looked cheap at p=20, but the original version can become expensive at larger budgets. The Vietnam-local `path_relink_fast` preserves the original trace, objective, and selected solution on validation cases, while caching the removal side of each swap and replacing repeated `np.intersect1d` calls with a boolean recovery mask. On validation cases, it was about 2.65x faster for 20 km p=100, 1.90x faster for 50 km p=20, and 3.16x faster for 100 km p=20. The tiny 20 km p=20 case is a wash, where overhead dominates.

The larger fast path-relinking checks completed as follows:

| Instance | Budget | Fast path-relink wall time | Trace length |
|---|---:|---:|---:|
| 50 km | 100 | 1.76 s | 50 |
| 50 km | 200 | 26.00 s | 127 |
| 100 km | 100 | 4.75 s | 56 |
| 100 km | 200 | 6.25 s | 59 |

## Current Defaults

The GRASP runner now defaults to:

```text
--local-search first_sparse
--path-relinking-method fast
```

The Fleur-style analysis runner uses the same defaults. Both scripts still expose `--local-search first` and `--path-relinking-method original` for direct comparison with the reference behavior.

## Remaining Opportunities

- Add a dedicated benchmark for full GRASP variants across budgets, not only primitive algorithm kernels.
- Consider a sparse/vectorized path-relinking kernel for 50 km p=200 and larger candidate grids. The current fast relinker is much better than the original but still spends meaningful time in pairwise enter-facility gain evaluation.
- Keep construction as-is unless moving to denser candidate grids. It is already small compared with local search and path relinking.
- If exact MIP comparisons are needed later, prefer the native Gurobi formulation over Pyomo for repeated runs and add warm starts from greedy or GRASP solutions.
