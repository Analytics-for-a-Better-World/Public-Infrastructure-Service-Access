# Line Profiler First Pass

Run date: 7 June 2026.

The line profiler was run on the fresh 20 km Vietnam PISA max-cover instance. These timings include `line_profiler` tracing overhead and should be used to identify line-level hotspots, not to benchmark wall-clock performance.

## Command

```powershell
py scripts\profile_vietnam_algorithms.py `
  --instance-npz C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_20km_threshold.npz `
  --budget 20 `
  --grasp-budget 20 `
  --grasp-iterations 2 `
  --output-dir C:\local\Parvathy\Vietnam\line_profiles
```

## Outputs

```text
C:\local\Parvathy\Vietnam\line_profiles
```

The runner wrote `.lprof` and readable `.txt` reports for:

- `construct_greedy`
- `construct_randomized`
- `construct_sample`
- `construct_random_plus`
- `first_swap_local_search`
- `path_relinking`
- `grasp_with_path_relinking`

The profiling harness has since been extended to include:

- `first_sparse_local_search`
- `path_relinking_original`
- `path_relinking_fast`

It also wrote:

- `profile_manifest.json`
- `profile_summary.json`

## Summary

| Case | Profiled seconds | Incremental population | Notes |
|---|---:|---:|---|
| greedy construction | 0.179 | 8.990M | strongest construction result in this deterministic pass |
| randomized construction | 0.196 | 7.759M | one seeded randomized construction |
| sample construction | 0.198 | 8.035M | one seeded sample construction |
| random-plus construction | 0.136 | 6.799M | early random steps reduce quality in this seed |
| first-swap local search | 0.524 | 8.990M | no improving swap from greedy at 20 km, p=20 |
| path relinking | 0.005 | 8.990M | tiny on this pair of p=20 solutions |
| GRASP + path relinking | 8.759 | 8.913M | two iterations; dominated by local search |

## Hotspots

Construction:

- `budgeted_construct` spends most time in the loop over newly covered households while building `touched` and `weights`.
- In the greedy profile, the two hottest lines are the slice assignments to `touched[pos:nxt]` and `weights[pos:nxt]`.
- `np.add.at` is not the main construction cost in this run.

Local search:

- `swap_first_improving` is dominated by `collect_candidates(newly_uncovered)`.
- In the standalone local-search profile, candidate collection accounts for about 86% of `swap_first_improving`.
- In the full GRASP profile, candidate collection accounts for about 96% of `swap_first_improving`.

Path relinking:

- Path relinking is not a bottleneck at p=20 on the 20 km instance.
- Within `_swap_delta`, `np.intersect1d` is the largest single line, but the absolute time is small in this run.

## Implications

The next performance work should focus on local-search candidate collection before tuning GRASP:

- cache or precompute candidate neighborhoods for each removable facility;
- avoid repeated Python loops over `newly_uncovered` households;
- consider vectorized or sparse-matrix-based candidate collection;
- reuse static `base_gain` across repeated local-search calls inside GRASP where possible;
- profile again on the 50 km and 100 km instances after local-search candidate collection is improved.

Follow-up: `scripts\vietnam_sparse_local_search.py` implements the sparse-matrix candidate-collection alternative as `first_sparse`. Its first benchmark is documented in `docs\sparse_local_search_alternative.md`. The broader speed audit, including the fast path-relinking pass and larger-budget checks, is documented in `docs\algorithm_speed_audit.md`.
