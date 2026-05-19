# Portugal healthcare data exploration notes

These notes summarize the first data exploration after the 18 May 2026 meeting and point to the sandbox files that can be reviewed or reused for distance-computation experiments.

## Main sandbox files

- `download_portugal_sns_urgency_official.py` downloads or refreshes the official SNS urgency source data.
- `fetch_portugal_sns_urgency_monitoring.py` fetches the SNS urgency monitoring data used during exploration.
- `build_portugal_health_service_layers.py` builds the Portugal healthcare service layers.
- `build_portugal_permanent_care_curated_official.py` contains the first curated attempt at the permanent-care/SAP layer.
- `display_portugal_health_service_layers.ipynb` displays the generated layers on a map.
- `pt_healthcare_facilities_for_pipeline.csv` is the compact pipeline-ready file shared for review.
- `data/health_service_locations/portugal_health_service_locations_official_preferred.csv` is the broader official-preferred study file.
- `data/health_service_locations/metadata.json` records sources, layer definitions, and counts.

## Pipeline-ready file

The compact pipeline-ready file `pt_healthcare_facilities_for_pipeline.csv` contains 176 facilities:

- 94 hospital emergency department locations.
- 52 primary-care facilities.
- 23 SAP-related services.
- 7 basic urgent-care facilities.

Each record includes an ID, name, type, tier, address fields where available, latitude, longitude, coordinate status, and source notes.

## Broader official-preferred study file

The broader `portugal_health_service_locations_official_preferred.csv` currently contains 1,691 facilities:

- 1,597 primary-care locations, still OSM-derived.
- 94 official hospital-emergency locations from the SNS Transparency Portal urgency data.

## Validation results

The official Portugal urgency facilities were compared with OSM hospital locations:

- 94 official urgency facilities matched.
- Median nearest OSM hospital distance: 0.089 km.
- 90th percentile nearest OSM hospital distance: 4.56 km.
- 78 of 94 were within 1 km.
- 84 of 94 were within 5 km.

This confirms that OSM hospitals are useful as a cross-check, but the official SNS urgency layer is preferred for the emergency-care layer.

## Current interpretation

Portugal has a solid official hospital-emergency layer. The main open issue is the middle urgent-care layer: the Serviços de Atendimento Permanente / Atendimento Permanente services do not appear to be available as a single stable downloadable facility-level list. The current SAP-related layer is therefore partly curated and should be documented carefully.

Primary care also remains less straightforward: the SNS Transparency `Unidades Funcionais` source appears to provide aggregate CSP-area records rather than clean individual facility-level points. For now, the broader primary-care layer remains OSM-derived, while the compact pipeline file uses the curated set prepared for immediate distance-analysis experiments.

## Reproducing the layer build

Run the relevant scripts from this folder:

```powershell
py download_portugal_sns_urgency_official.py
py build_portugal_health_service_layers.py
py build_portugal_permanent_care_curated_official.py
```

Then inspect:

```powershell
pt_healthcare_facilities_for_pipeline.csv
data/health_service_locations/portugal_health_service_locations_official_preferred.csv
data/health_service_locations/metadata.json
```

The display notebook can be used for a visual sanity check of the resulting layers.
