# Vietnam Component Comparison Diagnostic

This note documents the Vietnam stroke-center comparison created after the
true `--snap-components 0,1` PISA rerun did not complete within the available
run window. The attempted constrained run reached:

```text
Building Pandana network from 25,683,257 nodes and 26,420,891 edges
```

and was stopped after prolonged CPU activity without further pipeline output.

## What Was Generated

The generated diagnostic compares:

- `all components`: the completed Vietnam all-components PISA run.
- `component filter 0,1`: the same all-components matrix after keeping only
  source-target rows where both already-snapped endpoints are on road-network
  components 0 or 1.

This is not the same as a true component-aware rerun, because points on minor
components are not re-snapped to components 0 or 1. It is a conservative
diagnostic showing how much of the selected-solution coverage depends on minor
components in the existing completed matrix.

Metrics:

- `road_distance`: routed distance on the road graph only.
- `total_dist`: source snap distance + road distance + target snap distance.

Thresholds:

- 20 km
- 50 km
- 100 km

Budgets:

- baseline with the 130 existing stroke centers
- greedy frontiers for 0, 20, 40, 60, 80, 100, 150, 175, and 200 added
  candidate sites
- sparse first-swap local search at 175 added candidate sites

## Main Output Folder

```text
C:\local\Parvathy\Vietnam\component_comparison
```

Copies are also placed in:

```text
C:\Users\joaqu\OneDrive - UvA\Parvathy\Vietnam_replication\results\component_comparison
Z:\shared stuff\Parvathy_Vietnam_replication_2026-06-23\component_comparison
```

Key files:

- `vietnam_existing_and_p175_access_summary.csv`
- `vietnam_frontier_by_budget.csv`
- `vietnam_component_metric_delta.csv`
- `vietnam_component_frontiers_by_snap_metric.pdf`
- `vietnam_existing_vs_p175_summary.pdf`
- `vietnam_component_delta_heatmap.pdf`
- `vietnam_p175_solution_map_totaldist_50km.pdf`

## Headline Diagnostic Result

At baseline, the component filter does not change the reported coverage in
these completed all-components matrices. At 175 added candidate sites, the
component-filter diagnostic changes coverage by:

- 0.00 percentage points at 20 km.
- -0.10 percentage points for 50 km road distance.
- -0.08 percentage points for 50 km total distance.
- -0.27 percentage points for 100 km road distance.
- -0.27 percentage points for 100 km total distance.

These figures should be interpreted as a diagnostic lower-bound/proxy, not as
the final answer for a full constrained-snap PISA rerun.
