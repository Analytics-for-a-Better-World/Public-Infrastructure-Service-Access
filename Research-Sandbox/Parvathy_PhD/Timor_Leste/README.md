# Timor-Leste reproducibility code

This folder contains the Timor-Leste scripts used for the 2026 component-aware access-optimization study.

The Timor-Leste case is the exact and sensitivity case. It is used to evaluate:

- drive-only versus drive-plus-walk mobility assumptions;
- simplified versus unsimplified networks;
- unrestricted versus component-aware snapping;
- mainland Timor-Leste, Oecusse-Ambeno, and Atauro as intended planning components;
- 10 km, 5 km, and 1 km candidate grids;
- exact Gurobi frontiers and heuristic quality;
- saturation budgets and approximate Pareto curves.

## Execution order

Run from the repository root or from this folder with the same Python environment used by `general_distances_per_country`.

1. Inspect component geography:

   ```powershell
   python tools\identify_timor_components.py
   ```

2. Generate component-aware network-profile pipeline cases:

   ```powershell
   python tools\run_timor_network_profile_sensitivity.py
   ```

   This produces drive-only and drive-plus-walk, simplified and unsimplified cases. The script records snap statistics and pipeline timings.

3. Compare unrestricted and component-aware snapping:

   ```powershell
   python tools\compare_timor_component_snapping.py
   ```

4. Build and solve saturation curves:

   ```powershell
   python tools\compute_timor_saturation_curves.py --threads 1 --time-limit 300
   ```

   Use `--skip-exact` for a heuristic-only dry run when Gurobi is unavailable.

5. Generate figures and result tables:

   ```powershell
   python tools\make_timor_network_profile_results_figures.py
   python tools\make_timor_sibuni_network_detail.py
   ```

## Notes

The scripts were copied from the 2026 local sandbox to make the exact workflow auditable. Some scripts contain constants for dated output directories. For a fresh reproduction, update the output root constants near the top of the script or use the script CLI options where present.

Large data products are not committed. Expected generated folders include:

```text
outputs/timor_component_geography_*
outputs/timor_network_profile_component012_*
outputs/timor_component_snapping_comparison_*
outputs/timor_component012_saturation_*
```

The result figures used in the integrated report are generated from those output folders.

## WorldPop 2026 exact-frontier replication

The latest exact benchmark uses WorldPop Global2 `R2025A v1` for 2026 and a
fixed Timor-Leste OSM snapshot. Run the complete, self-validating notebook:

```text
notebooks/timor_leste_worldpop_2026_full_exact_pareto_replication.ipynb
```

It documents and verifies the input hashes, reconstructs the three sparse
candidate-grid matrices when requested, and launches
`tools/run_timor_global2_2026_exact_pareto.py`. The campaign uses Gurobipy for
all nine combinations of 10, 5, and 1 km candidate grids with 2, 5, and 10 km
service thresholds. It solves every integer budget from zero through the first
exact saturation budget.

Each budget is durably checkpointed in two append-only files: `frontier.jsonl`
contains status, bounds, gap, warm-start provenance, timestamps, and detailed
timings; `solutions.jsonl` contains both internal indices and stable candidate
IDs for the corresponding exact solution. A resumed run skips previously
verified optimal budgets. Generated data and solver outputs remain outside Git
because of their size.
