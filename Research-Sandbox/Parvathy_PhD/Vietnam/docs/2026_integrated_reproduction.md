# Vietnam 2026 integrated reproduction runbook

This document extends the existing Vietnam replication folder with the 2026 integrated study code.

The Vietnam case is the national-scale case. It is used to evaluate:

- the public September 2025 Vietnam Stroke Association list with 170 records;
- geocoding and comparison against the earlier 130-record workbook;
- national road-driving distance matrices;
- component-aware snapping, including the TT 80 diagnostic;
- 10 km, 5 km, and 1 km candidate-grid sizes;
- adaptive chunking for large matrix writes;
- approximate Pareto construction on the 1 km grid using `abw_maxcover`;
- exact/heuristic comparison where exact optimization is feasible.

## Main scripts

```text
scripts/audit_vietnam_170_vs_130_geocoding.py
scripts/run_vietnam_road_pipeline_batch.py
scripts/run_vietnam_three_grid_batch.py
scripts/run_vietnam_1km_5km_approx_pareto.py
scripts/run_vietnam_5km_exact_selected_p.py
scripts/run_vietnam_fleur_no_baseline_s20.py
scripts/run_vietnam_tt80_cli_component_diagnostics.py
scripts/make_vietnam_component_figure.py
scripts/make_vietnam_tt80_pipeline_component_inset.py
scripts/make_vietnam_170_chunking_figure.py
```

## Typical execution order

1. Audit the 170-record geocoded list against the older 130-record workbook:

   ```powershell
   python scripts\audit_vietnam_170_vs_130_geocoding.py
   ```

2. Run the road-driving pipeline batch:

   ```powershell
   python scripts\run_vietnam_road_pipeline_batch.py --help
   ```

   Use the current 170-row geocoded CSV as `--source-table`, `--snap-components 0,1`, and `--aggregate-factor 5` for the integrated study settings.

3. Generate component diagnostics:

   ```powershell
   python scripts\run_vietnam_tt80_cli_component_diagnostics.py
   python scripts\make_vietnam_component_figure.py
   python scripts\make_vietnam_tt80_pipeline_component_inset.py
   ```

4. Build the 1 km, 5 km-threshold approximate Pareto result:

   ```powershell
   python scripts\run_vietnam_1km_5km_approx_pareto.py
   ```

5. Generate the chunking figure:

   ```powershell
   python scripts\make_vietnam_170_chunking_figure.py
   ```

## Data note

The public source list does not include coordinates. The geocoded CSV used by the 2026 run is a local derived input and should be stored with provenance in the local data archive, not committed blindly to Git unless licensing and privacy review are complete.

The source page for the 170-record public list was:

```text
https://hoidotquyvietnam.com/danh-sach-cac-benh-vien-co-don-vi-hoac-trung-tam-san-sang-cap-cuu-dot-quy-ban-cap-nhat-thang-09-2025/
```

Large generated matrices, parquet parts, and `.npz` optimization instances are not committed.
