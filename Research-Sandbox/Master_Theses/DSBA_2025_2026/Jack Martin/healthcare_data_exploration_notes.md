# Netherlands healthcare data exploration notes

These notes summarize the first data exploration after the 18 May 2026 meeting and point to the sandbox files that can be reviewed or reused for distance-computation experiments.

## Main sandbox files

- `build_netherlands_health_service_layers.py` builds the Netherlands healthcare service layers.
- `display_netherlands_health_service_layers.ipynb` displays the generated layers on a map.
- `data/health_service_locations/netherlands_health_service_locations_study_layers.csv` is the recommended study input file.
- `data/health_service_locations/netherlands_health_service_locations_all_layers.csv` keeps the broader collected layers.
- `data/health_service_locations/metadata.json` records sources, layer definitions, and counts.
- `Drenthe_centriods.csv` is Jack's OSM sample for Drenthe.

## Preliminary counts

The recommended Netherlands study layer currently contains 2,647 facilities:

- 2,465 general-practitioner locations from OpenStreetMap.
- 103 huisartsenspoedposten from the official RIVM/VZinfo layer.
- 79 SEH hospital locations from the official RIVM/VZinfo layer.

The official SEH layer contains 76 hospitals open 24/7 and 3 with limited opening hours. An OSM hospital emergency proxy contained 212 locations, but the official SEH layer is preferred for the thesis.

## Validation results

Official SEH locations matched very closely to OSM hospital locations:

- 76 official 24/7 SEH locations matched.
- Median nearest OSM hospital distance: 0.085 km.
- 90th percentile nearest OSM hospital distance: 0.269 km.
- 76 of 76 were within 1 km.

The huisartsenspoedposten comparison shows that OSM is not a reliable replacement for the official HAP layer:

- 103 official HAP locations.
- 66 OSM HAP proxy locations.
- Median nearest OSM proxy distance: 6.74 km.
- 90th percentile nearest OSM proxy distance: 31.00 km.
- 42 official HAP locations were within 1 km of an OSM proxy.
- 47 official HAP locations were within 5 km of an OSM proxy.

## Current interpretation

For the Netherlands, the official huisartsenspoedposten and SEH layers are strong enough to start distance computations. The GP layer is still OSM-derived and may cover only about half of all GP practice locations. Access to CBS/Nivel microdata would be preferable for a complete official GP-practice coordinate layer.

The practical recommendation is to proceed with the current Netherlands study layer for first distance experiments, while documenting the GP-source limitation and continuing to investigate CBS/Nivel or Vektis/AGB access.

## Reproducing the layer build

Run the build script from this folder:

```powershell
py build_netherlands_health_service_layers.py
```

Then inspect:

```powershell
data/health_service_locations/netherlands_health_service_locations_study_layers.csv
data/health_service_locations/metadata.json
```

The display notebook can be used for a visual sanity check of the resulting layers.
