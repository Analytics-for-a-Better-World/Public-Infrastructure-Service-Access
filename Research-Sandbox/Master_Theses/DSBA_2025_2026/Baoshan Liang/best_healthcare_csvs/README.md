# Portugal best healthcare CSV collection

This folder collects the best currently available Portugal healthcare facility CSVs for Baoshan Liang's thesis work. The files are copied from `data/health_service_locations` so the pipeline inputs and audit layers are easy to find in one place.

## Recommended pipeline input

Use `portugal_health_service_locations_pipeline_with_permanent_care_geocoded.csv` for distance-matrix runs.

It contains 1,712 geocoded facilities:

- 94 `hospital_emergency_official` locations from the SNS Transparency Portal urgency dataset.
- 1,597 `primary_care` locations from the best current OSM-derived primary-care proxy.
- 21 `permanent_care_geocoded` SAP/SAC/permanent-care locations from curated official pages, geocoded on 25 May 2026.

This is the best current geocoded source table because the emergency layer is official and facility-level, the primary-care layer remains the best available open point layer found so far, and the middle SAP/SAC layer is now represented by a small curated geocoded set.

## Included CSVs

- `portugal_health_service_locations_official_preferred.csv`: recommended geocoded source table for the distance pipeline.
- `portugal_hospital_emergency_official_sns.csv`: official SNS hospital emergency locations; 94 rows.
- `portugal_primary_care.csv`: OSM-derived primary-care proxy; 1,597 rows.
- `portugal_permanent_care_curated_official_pages.csv`: first curated official-page list for SAP/SAC/permanent-care services; 21 rows, but most rows still need geocoding and should not be used directly as pipeline sources yet.
- `portugal_permanent_care_curated_official_pages_geocoded.csv`: geocoded version of the curated SAP/SAC/permanent-care list; 21 rows, 12 geocoded by Nominatim and 9 by Google Maps fallback.
- `portugal_health_service_locations_pipeline_with_permanent_care_geocoded.csv`: recommended pipeline input combining official ED, OSM primary care, and geocoded permanent care; 1,712 rows.
- `portugal_primary_care_official_sns_area_counts.csv`: SNS area-level primary-care counts; 52 rows. Useful for validation/context, not facility-level distance computation.
- `portugal_sns_urgency_facility_locations_official.csv`: official urgency source extract with facility-level coordinates; 94 rows.
- `portugal_health_service_locations_study_layers.csv`: broader exploratory OSM-derived study layer; 1,950 rows. Keep for sensitivity checks, but prefer `portugal_health_service_locations_official_preferred.csv`.
- `metadata.json`: source notes, recommended study file, and layer counts.

## Distance-pipeline run

Run this folder's recommended table as a custom source table only. Do not add generated candidates or default OSM amenities:

```powershell
py run_pipeline.py portugal `
  --sources table `
  --source-table "C:\local\GIT\Public-Infrastructure-Service-Access\Research-Sandbox\Master_Theses\DSBA_2025_2026\Baoshan Liang\best_healthcare_csvs\portugal_health_service_locations_pipeline_with_permanent_care_geocoded.csv" `
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

The matrix should be interpreted as population-to-existing-healthcare access only. It does not include newly generated candidate sites. The SAP/permanent-care layer is represented by the curated geocoded rows, but should still be documented as less complete than the official hospital-emergency layer.

## Completed pipeline run

Completed on 25 May 2026 with:

- `--sources table`
- `--destinations population`
- `--aggregate-factor 20`
- `--max-total-dist 150000`
- `--matrix-output-mode split`
- `--network-backend pyrosm`
- no generated candidates
- no default OSM amenity extraction

Outputs:

- Matrix: `C:\local\Download_Depot\portugal_data\outputs\distance_matrix_src_table_dst_population_pop_1_sample_1_max_none_agg_20_maxdist_150000_amenity_amenity_all-dst_population-src_table_portugal_health_service_locations_pi_9220fa425f_no_candidates.parquet`
- Sources: `C:\local\Download_Depot\portugal_data\outputs\sources_pop_1_sample_1_max_none_agg_20_maxdist_150000_amenity_amenity_all-dst_population-src_table_portugal_health_service_locations_pi_9220fa425f_no_candidates.parquet`
- Targets: `C:\local\Download_Depot\portugal_data\outputs\targets_pop_1_sample_1_max_none_agg_20_maxdist_150000_amenity_amenity_all-dst_population-src_table_portugal_health_service_locations_pi_9220fa425f_no_candidates.parquet`
- Manifest: `C:\local\Download_Depot\portugal_data\outputs\run_manifest_pop_1_sample_1_max_none_agg_20_maxdist_150000_amenity_amenity_all-dst_population-src_table_portugal_health_service_locations_pi_9220fa425f_no_candidates.yaml`
- Context figure: `C:\local\Download_Depot\portugal_data\figures\portugal_healthcare_all_layers_context_resolution_5000m.png`

Run size:

- 1,712 healthcare source points.
- 23,486 aggregated population target points.
- 10,494,966 retained source-target distance rows under the 150 km cap.
