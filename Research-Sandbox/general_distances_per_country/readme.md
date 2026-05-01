# Distance Pipeline

Build population-to-facility distance tables for a country using WorldPop population rasters, OpenStreetMap road networks, existing health facilities, and optional grid-based candidate facility sites.

The code is organized as a reusable pipeline plus a notebook walkthrough. Expensive stages are cached so repeated runs can reuse downloaded files, parsed networks, population points, snapped locations, and distance matrices.

---

# Overview

The pipeline performs these steps:

1. Load a country configuration.
2. Download or reuse the country OSM PBF and WorldPop raster.
3. Build a driving road network from OSM.
4. Convert WorldPop raster cells into population target points.
5. Extract existing health-related facilities from OSM.
6. Generate optional regular-grid candidate facility sites from the country boundary.
7. Optionally remove candidate sites on water and filter candidates too far from the road network.
8. Optionally build a context map.
9. Snap population targets, existing facilities, and candidate sites to road-network nodes.
10. Combine existing facilities and candidates into one source table.
11. Compute a sparse target-source distance table.

The distance table is sparse: source-target pairs are first limited by a crow-fly distance threshold from the country config, then pairs without a valid network path are removed. `--max-total-dist` can apply an additional total-distance filter.

---

# Usage

Run from this directory:

```bash
python run_pipeline.py <country> [options]
```

Examples:

```bash
python run_pipeline.py timor_leste
python run_pipeline.py tls --population-threshold 1 --sample-fraction 1
python run_pipeline.py prt --sample-fraction 0.1 --save-map
python run_pipeline.py vnm --aggregate-factor 10 --amenity hospital clinic
```

The notebook `testdrive_general_country.ipynb` mirrors the same pipeline stages with inspectable intermediate objects. Unlike the CLI, the notebook also writes selected parquet outputs at the end.

---

# Countries

Known country identifiers include:

- `netherlands`, `nld`, `nl`
- `portugal`, `prt`, `pt`
- `timor_leste`, `timor-leste`, `tls`, `tl`
- `vietnam`, `viet_nam`, `vnm`, `vn`
- `laos`, `lao`, `la`
- `luxembourg`

If a country module is missing, `load_cfg()` can attempt to generate one through the OpenAI helper in `distance_pipeline/use_openai.py`. Existing country configs do not need generation.

---

# CLI Options

## General

```bash
--log-file PATH
```
Write console logs to a file.

```bash
--force-recompute
```
Ignore matching caches and rebuild cached stages.

```bash
--quiet
```
Reduce console output.

## Population

```bash
--population-threshold FLOAT
```
Minimum population value retained from the raster. Default: `1.0`.

```bash
--sample-fraction FLOAT
```
Randomly sample retained population points. Must be in `(0, 1]`. Default: `1.0`.

```bash
--max-points INT
```
Optional cap on the number of population points.

## Aggregation

```bash
--aggregate-factor INT
```
Aggregate raster cells by summing non-overlapping square blocks before converting them to points. Overrides the country config.

```bash
--no-aggregate
```
Disable raster aggregation, even if the country config defines one.

Resolution order:

1. `--no-aggregate` disables aggregation.
2. `--aggregate-factor` overrides the config.
3. Otherwise the pipeline uses `cfg.aggregate_factor`.

## Facilities

```bash
--amenity hospital clinic doctors
```
Restrict OSM facility extraction to specific `amenity=*` values. If omitted, the pipeline uses the default health-related amenity list from `load_health_facilities()`.

```bash
--no-healthcare-tag
```
Disable the broad `healthcare=*` OSM tag filter and use only the selected/default amenity values.

## Candidate Generation

```bash
--candidate-grid-spacing-m FLOAT
```
Grid spacing for candidate facility sites in meters. If omitted, the country config value is used. If both are `None`, candidate generation is disabled.

```bash
--candidate-max-snap-dist-m FLOAT
```
Maximum allowed snapping distance from a candidate site to the road network, in meters. If omitted, the country config value is used.

Country configs can also control whether candidates on water are excluded and whether boundary points are included.

## Distance Computation

```bash
--max-total-dist FLOAT
```
Keep only rows whose `total_dist` is less than or equal to this value, in meters.

## Map Output

```bash
--build-map
```
Build the context map stage. By itself this renders and closes the figure unless paired with `--save-map` or `--show-map`.

```bash
--save-map
```
Save the context map. This now triggers map building automatically.

```bash
--show-map
```
Display the context map interactively. This now triggers map building automatically.

```bash
--map-path PATH
```
Custom output path for the saved map.

```bash
--map-dpi INT
```
Resolution for saved maps. Default: `300`.

---

# Outputs

The CLI currently computes the pipeline, writes caches, and prints the head of the distance matrix. The notebook additionally writes parquet files under:

```text
<cfg.BASE_DIR>/outputs/
```

Notebook output files:

- `population_<run_tag>.parquet`
- `existing_sources_<run_tag>.parquet`
- `distance_matrix_<run_tag>.parquet`

The distance matrix columns are:

- `target_id`
- `source_id`
- `source_nearest_node`
- `target_nearest_node`
- `target_to_road_dist`
- `road_distance`
- `source_to_road_dist`
- `total_dist`

All distance values are in meters.

---

# Caching

Caches are written under:

```text
<cfg.BASE_DIR>/cache/
```

Cache keys include the runtime inputs that affect each stage:

- population threshold
- sample fraction
- maximum population points
- aggregation factor
- facility amenity filters
- healthcare-tag mode
- candidate grid spacing
- candidate snap-distance filter
- candidate presence
- distance threshold
- maximum total distance
- projected EPSG code where relevant

Use `--force-recompute` when you want to rebuild even if a matching cache exists.

---

# Data Model

Targets are population points. Sources are existing health facilities plus optional candidate facility sites.

Important assumptions:

- Tables carry an `ID` column.
- Tables used for distance computation are indexed by `ID`.
- IDs are unique and consistent with the index.
- Snapped target rows contain `nearest_node` and `dist_snap_target`.
- Snapped source rows contain `nearest_node` and `dist_snap_source`.

---

# Dependencies

There is currently no pinned environment file in this folder. The main libraries used by the pipeline include:

- `geopandas`
- `matplotlib`
- `numpy`
- `pandas`
- `pandana`
- `polars`
- `pyrosm`
- `rasterio`
- `scipy`
- `shapely`
- `contextily`

For reproducible runs, add a `requirements.txt`, `pyproject.toml`, or environment file before using this pipeline on a fresh machine.

---

# License

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project is documented as MIT licensed. The MIT license is a permissive license: you may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, provided the copyright notice and license text are included with substantial portions of the software.

If this folder is distributed independently, include a full `LICENSE` file next to this README.
