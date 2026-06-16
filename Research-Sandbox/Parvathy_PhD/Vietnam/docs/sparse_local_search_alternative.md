# Sparse Local-Search Alternative

This is an experimental Vietnam-local alternative to the heavy first-swap local search in `approximated_tradeoff`.

It does not modify:

```text
Research-Sandbox\approximated_tradeoff
```

The implementation lives in:

```text
scripts\vietnam_sparse_local_search.py
```

It is exposed as:

```text
--local-search first_sparse
```

The Vietnam runners now default to:

```text
--local-search first_sparse
```

The original reference path remains available for validation:

```text
--local-search first
```

## Idea

The original `swap_first_improving` spends most of its time collecting closed facilities that touch newly uncovered households after a candidate facility is removed.

The alternative keeps the same first-improvement semantics, but replaces this Python-level candidate collection:

```text
for household in newly_uncovered:
    for facility in instance.facilities_of(household):
        ...
```

with a SciPy CSR row-sum over the household-to-facility matrix:

```text
A[newly_uncovered].sum(axis=0)
```

This moves the expensive candidate-discovery step into sparse matrix code.

## Benchmark

Command:

```powershell
py scripts\benchmark_local_search_variants.py `
  --instances C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_20km_threshold.npz `
              C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_50km_threshold.npz `
              C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_100km_threshold.npz `
  --budgets 20 `
  --output-csv C:\local\Parvathy\Vietnam\line_profiles\local_search_variant_benchmark.csv
```

Results, starting from the same greedy solution in each instance:

| Threshold | Variant | Incremental population | Moves | Search seconds | Wall seconds incl. setup | Same objective | Same solution |
|---:|---|---:|---:|---:|---:|---|---|
| 20 km | first | 8.990M | 0 | 0.072 | 0.107 | yes | yes |
| 20 km | first_sparse | 8.990M | 0 | 0.009 | 0.022 | yes | yes |
| 50 km | first | 6.909M | 2 | 0.747 | 0.790 | yes | yes |
| 50 km | first_sparse | 6.909M | 2 | 0.025 | 0.074 | yes | yes |
| 100 km | first | 3.510M | 17 | 35.530 | 35.862 | yes | yes |
| 100 km | first_sparse | 3.510M | 17 | 1.147 | 1.423 | yes | yes |

## Status

This is now the preferred local-search kernel for Vietnam experiments. It is still deliberately local to this folder; `Research-Sandbox\approximated_tradeoff` remains unchanged, and `--local-search first` can be used whenever a reference comparison is needed.
