# Netherlands best healthcare CSV collection

This folder collects the best currently available Netherlands healthcare facility CSVs for Jack Martin's thesis work. The files are copied from `data/health_service_locations` so the pipeline inputs and audit layers are easy to find in one place.

## Recommended pipeline input

Use `netherlands_health_service_locations_study_layers_european_nl.csv` for distance-matrix runs.

It contains 2,541 geocoded facilities in the European Netherlands:

- 2,362 `general_practitioner` locations from OpenStreetMap GP/huisarts-oriented tags.
- 103 `huisartsenpost` locations from the official RIVM/VZinfo huisartsenspoedposten layer.
- 76 `hospital` locations from the official RIVM/VZinfo 24/7 SEH layer.

This is the best current study layer because it uses official sources for the Dutch urgent-care and emergency-care layers, while using OSM as the best available open GP point layer.

## Included CSVs

- `netherlands_health_service_locations_study_layers.csv`: recommended geocoded source table for the distance pipeline.
- `netherlands_health_service_locations_study_layers_european_nl.csv`: cleaned pipeline input restricted to finite coordinates in the European Netherlands; 2,541 rows. This excludes 106 Caribbean Netherlands / overseas OSM records that distort the national road-network run.
- `netherlands_hospitals_seh_official.csv`: official RIVM/VZinfo SEH hospital locations; 79 rows, including 76 open 24/7 and 3 with limited opening hours.
- `netherlands_huisartsenpost.csv`: official RIVM/VZinfo huisartsenspoedposten locations; 103 rows.
- `netherlands_general_practitioner.csv`: OSM-derived GP/huisarts layer; 2,465 rows. This is useful but likely incomplete relative to CBS/Nivel or Vektis/AGB.
- `netherlands_hospital_emergency_proxy.csv`: OSM hospital/emergency proxy layer; 212 rows. Use for validation or sensitivity checks, not as the preferred SEH layer.
- `netherlands_huisartsenpost_osm_proxy.csv`: OSM HAP proxy; 66 rows. Validation showed this is not a good substitute for the official HAP layer.
- `metadata.json`: source notes, recommended study file, and layer counts.

## Distance-pipeline run

Run this folder's recommended table as a custom source table only. Do not add generated candidates or default OSM amenities:

```powershell
py run_pipeline.py netherlands `
  --sources table `
  --source-table "C:\local\GIT\Public-Infrastructure-Service-Access\Research-Sandbox\Master_Theses\DSBA_2025_2026\Jack Martin\best_healthcare_csvs\netherlands_health_service_locations_study_layers_european_nl.csv" `
  --source-lon-column longitude `
  --source-lat-column latitude `
  --source-id-column facility_id `
  --destinations population `
  --max-total-dist 150000 `
  --matrix-output-mode split `
  --save-map `
  --map-basemap voyager-no-labels `
  --map-basemap-alpha 0.65 `
  --map-roads true
```

The matrix should be interpreted as population-to-existing-healthcare access only. It does not include newly generated candidate sites.

## Completed pipeline run

Completed on 25 May 2026 with:

- `--sources table`
- `--destinations population`
- `--aggregate-factor 20`
- `--max-total-dist 150000`
- `--matrix-output-mode split`
- `--network-backend osmium`
- no generated candidates
- no default OSM amenity extraction

Outputs:

- Matrix: `C:\local\Download_Depot\netherlands_data\outputs\distance_matrix_src_table_dst_population_pop_1_sample_1_max_none_agg_20_maxdist_150000_amenity_amenity_all-dst_population-src_table_netherlands_health_service_locations_d979d68da5_no_candidates.parquet`
- Sources: `C:\local\Download_Depot\netherlands_data\outputs\sources_pop_1_sample_1_max_none_agg_20_maxdist_150000_amenity_amenity_all-dst_population-src_table_netherlands_health_service_locations_d979d68da5_no_candidates.parquet`
- Targets: `C:\local\Download_Depot\netherlands_data\outputs\targets_pop_1_sample_1_max_none_agg_20_maxdist_150000_amenity_amenity_all-dst_population-src_table_netherlands_health_service_locations_d979d68da5_no_candidates.parquet`
- Manifest: `C:\local\Download_Depot\netherlands_data\outputs\run_manifest_pop_1_sample_1_max_none_agg_20_maxdist_150000_amenity_amenity_all-dst_population-src_table_netherlands_health_service_locations_d979d68da5_no_candidates.yaml`
- Context figure: `C:\local\Download_Depot\netherlands_data\figures\netherlands_healthcare_all_layers_context_european_nl_resolution_10000m.png`

Run size:

- 2,541 healthcare source points.
- 17,781 aggregated population target points.
- 21,522,195 retained source-target distance rows under the 150 km cap.
