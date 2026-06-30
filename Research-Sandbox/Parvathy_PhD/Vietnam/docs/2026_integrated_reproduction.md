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
scripts/geocode_vietnam_170_mapbox.py
scripts/run_vietnam_osm_health_pipeline_batch.py
scripts/run_vietnam_osm_health_approx_pareto.py
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

## 30 June 2026 update: geocoding audit and OSM health baseline

The September 2025 Vietnam Stroke Association list remains important public context, but it is not a coordinate-ready routing layer. The local Google-derived 170-row coordinate file and the Mapbox audit output are provider-derived data. They should not be committed to GitHub or treated as publication-ready open data until the storage and reuse conditions of the relevant provider account have been checked.

Policy pages consulted on 30 June 2026:

```text
https://developers.google.com/maps/documentation/geocoding/policies
https://cloud.google.com/maps-platform/terms/maps-service-terms
https://docs.mapbox.com/api/search/geocoding/
```

The Mapbox geocoding replication was run with `permanent=true`, `country=vn`, `language=vi`, and `limit=1`. The token is read from `MAPBOX_ACCESS_TOKEN`; do not hard-code or commit it.

```powershell
$env:MAPBOX_ACCESS_TOKEN = '<token from local secret store>'
python scripts\geocode_vietnam_170_mapbox.py `
  --input-csv <local-data-root>\vietnam_stroke_centers_170_vi_vnsa_2025_09_raw.csv
```

Quality decision from the audit: `do_not_use_as_primary_without_manual_correction`. The run had 39 failed rows, 75 review-level matches, and 33 large or major disagreements against the local Google-derived coordinates. Among comparable rows, the median Mapbox--Google separation was 1.343 km and the 90th percentile was 67.6 km.

For the reproducible Vietnam decision runs, use the open OSM health-amenity baseline:

```text
amenity=hospital
amenity=clinic
```

The batch runner is:

```powershell
python scripts\run_vietnam_osm_health_pipeline_batch.py `
  --fresh-root <local-vietnam-run-root> `
  --output-root <local-output-root>\vietnam_osm_health_agg5_20260630_s20 `
  --spacings 10000 5000 1000 `
  --amenity hospital clinic `
  --snap-components 0,1 `
  --aggregate-factor 5 `
  --max-total-dist 20000
```

Headline OSM health baseline results:

| Quantity | Value |
|---|---:|
| Raw OSM health features | 2,438 |
| Deduplicated OSM health amenities | 2,432 |
| Hospitals | 1,760 |
| Clinics | 672 |
| Snapped to component 0 | 2,426 |
| Snapped to component 1 | 6 |
| Amenity-to-population rows within 20 km | 7,564,396 |
| Population within 20 km road distance | 92.139% |

The aligned pipeline configurations completed in 00:07:08.383 for the 10 km configuration, 00:04:23.716 for 5 km, and 00:04:03.627 for 1 km after caches were available.

The OSM-baseline approximate Pareto runner is:

```powershell
python scripts\run_vietnam_osm_health_approx_pareto.py `
  --threshold-m 5000 `
  --run-output <local-vietnam-run-root>\vietnam_data\outputs `
  --output-root <local-output-root>\vietnam_osm_health_approx_pareto_20260630
```

At the 5 km threshold, the OSM hospital/clinic baseline covers 51.302% of the modeled population. After removing already covered demand from the residual objective, the approximate Pareto results were:

| Grid | Residual candidates | Residual arcs | Final coverage | Runtime |
|---:|---:|---:|---:|---:|
| 10 km | 3,077 | 238,636 | 62.647% | 00:00:02.471 |
| 5 km | 11,407 | 940,612 | 86.871% | 00:00:08.083 |
| 1 km | 186,719 | 16,778,081 | 99.050% | 00:00:53.113 |

The 1 km curve is flat from approximately `p=28723`; adding more candidate sites does not increase the modeled 5 km coverage. The approximation reports the pointwise maximum of greedy, zero-loss compaction, and regreedy.
