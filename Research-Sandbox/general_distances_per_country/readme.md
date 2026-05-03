# Distance Pipeline

Build population-to-facility distance tables for a country using WorldPop population rasters, OpenStreetMap road networks, existing facilities, and optional grid-based candidate facility sites.

The code is organized as a reusable pipeline plus a notebook walkthrough. Expensive stages are cached so repeated runs can reuse downloaded files, parsed networks, population points, snapped locations, and distance matrices.

---

# Overview

The pipeline performs these steps:

1. Load a country configuration.
2. Download or reuse the country OSM PBF and WorldPop raster.
3. Build a driving road network from OSM.
4. Convert WorldPop raster cells into population target points.
5. Extract existing default facilities from OSM.
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
python run_pipeline.py timor_leste --amenity school
py run_pipeline.py vietnam --source-layer table --source-table "C:\Users\joaqu\OneDrive - UvA\Bureaublad\stroke-facs-100-en.xlsx" --source-lon-column Longitude --source-lat-column Latitude --source-id-column ID --max-total-dist 150000 --aggregate-factor 10 --save-map
```

The notebook `testdrive_general_country.ipynb` mirrors the same pipeline stages with inspectable intermediate objects. Both the CLI and notebook write parquet outputs using the same run-tag helper.

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
Restrict OSM facility extraction to specific `amenity=*` values. If omitted, the pipeline uses the default amenity list from `load_facilities()`. The current defaults are health-oriented, but the loader itself is service-agnostic.

```bash
--deduplicate-amenities true
```
OSM amenities are deduplicated by default after non-point geometries are converted to representative points. Set `--deduplicate-amenities false` to keep the raw OSM amenity features for auditing or sensitivity analysis. When a nearby point/node and polygon/centroid have the same normalized name and amenity, the point/node feature is kept because it is usually the more intentional routing location. Unnamed amenities are deduplicated more conservatively by amenity and a small projected spatial cell.


## Custom source and destination tables

The pipeline is being generalized so matrix sources and destinations can come from more than the default population-to-facility setup. The intended layer vocabulary is:

- `population`: gridded WorldPop-derived points, normally used as demand targets
- `amenities`: OSM features selected by `--amenity`
- `candidates`: generated regular-grid candidate sites
- `custom`: a user-provided table with coordinates

Custom point tables may be CSV, Excel, parquet, or GeoJSON. They should contain an `ID` column when stable identifiers matter and either `Longitude`/`Latitude`, `lon`/`lat`, `lng`/`lat`, `x`/`y`, or point geometry. Explicit coordinate columns can be supplied with `--source-lon-column`, `--source-lat-column`, and `--source-id-column`. Optional `population`, `demand`, `weight`, or `headcount` columns are used as demand weights; otherwise a unit weight is assumed. At the CLI, `--source-layer table --source-table <path>` uses the supplied table as the source layer and WorldPop-derived population points as destinations. Candidate-grid sources are disabled for table-source runs so the matrix reflects the provided sources only. This supports use cases such as a supplied spreadsheet of facilities or a later matrix where supplied points and amenity sets are mixed.

## Routing utilities

`distance_pipeline.routing` contains the first reusable routing helpers for experiments beyond coverage matrices:

- `add_edge_speeds()` annotates pyrosm OSM road edges with speed and travel-time columns, using parseable `maxspeed` tags when available and highway-class defaults otherwise.
- `build_networkx_graph()` builds a directed NetworkX graph from the loaded OSM nodes and edges, for route reconstruction and plotting.
- `route_between_nodes()` reconstructs a shortest route between two snapped network nodes under a selected edge weight such as `length` or `travel_time_s`.
- `plot_tsp_routes()` renders one or more TSP route layers over a contextily basemap and saves article-ready PNG figures.
- `symmetric_tsp_via_gurobi_sparse()` adapts the MO-book lazy-subtour Gurobi TSP formulation to sparse edge-cost dictionaries, so capped distance matrices do not need to be expanded to dense all-pairs arrays.
- `directed_tsp_via_gurobi_sparse()` solves the corresponding directed sparse TSP on ordered arcs from the matrix, with one incoming and one outgoing arc per stop and lazy directed subtour cuts.

The initial Netherlands routing driver builds an institution-to-institution matrix for OSM `amenity=university` and `amenity=college`, capped at 100 km, solves sparse undirected and directed TSP variants with Gurobi when the retained graph supports a tour, and writes the stops, matrix, tours, summary, route geometries, and route figures. The undirected TSP uses symmetric road distances. The directed distance TSP uses ordered road-distance arcs. The fastest-route variants rebuild the ordered arc costs from shortest travel times on an OSM graph whose edge speeds come from parseable `maxspeed` tags or highway-class defaults, then solve a separate directed TSP on those travel-time costs. The conservative fastest variant repeats that optimization after reducing every edge speed to 80% of its nominal value. The summary reports total route length, total travel time, average driven speed, and route changes between the nominal and conservative fastest tours. If the local Overleaf article figures folder exists, the route figures are saved there automatically; otherwise pass `--figures-dir <path>`.

```bash
python routing_experiments.py netherlands --max-total-dist 100000
```

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

The CLI and notebook write parquet outputs under:

```text
<cfg.BASE_DIR>/outputs/
```

Output files:

- `population_<run_tag>.parquet`
- `existing_sources_<run_tag>.parquet`
- `sources_<run_tag>.parquet`
- `distance_matrix_<run_tag>.parquet`
- `run_manifest_<run_tag>.yaml`

`existing_sources_<run_tag>.parquet` contains only the OSM amenities selected by the amenity filter. `sources_<run_tag>.parquet` contains the full source layer used by the matrix, including existing amenities and generated candidate sites when candidates are enabled. Use `sources_<run_tag>.parquet` for optimization plots or any model whose selected source IDs may include candidates.

Large distance matrices are built with Polars in the CLI and written directly to parquet. This avoids converting very large matrices to pandas before writing, which can require several additional gigabytes of memory.

The run tag records the population settings, aggregation factor, facility filters, candidate settings, and distance filter so different runs do not overwrite each other.

The run manifest is the reproducibility record for a run. It stores:

- source URLs for the OSM PBF and WorldPop raster
- resolved country configuration values
- runtime settings and resolved parameters
- local input and output paths
- file sizes and modification timestamps
- SHA256 checksums for the input and output files
- the current pipeline git commit when available

This matters because Geofabrik `*-latest.osm.pbf` extracts are moving snapshots of OpenStreetMap. A run on a fresh machine may download different OSM road and facility data than an earlier run. WorldPop inputs are more explicitly versioned by year and filename, but the manifest still records the exact local file bytes used for the run.

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
- candidate grid spacing
- candidate snap-distance filter
- candidate presence
- distance threshold
- maximum total distance
- projected EPSG code where relevant

Use `--force-recompute` when you want to rebuild even if a matching cache exists.

---

# Data Model

Targets are population points. Sources are existing facilities plus optional candidate facility sites.

Important assumptions:

- Tables carry an `ID` column.
- Tables used for distance computation are indexed by `ID`.
- IDs are unique and consistent with the index.
- Snapped target rows contain `nearest_node` and `dist_snap_target`.
- Snapped source rows contain `nearest_node` and `dist_snap_source`.

---

# Dependencies

There is currently no pinned environment file in this folder. Use a NumPy 1.x environment for now: Pandana ships compiled extension modules that were built against the NumPy 1.x C API, and importing those wheels under NumPy 2 can fail with binary-compatibility errors. In practice, install `numpy<2` before installing Pandana unless you are using a Pandana build that explicitly supports NumPy 2.

The main libraries used by the pipeline include:

- `geopandas`
- `matplotlib`
- `networkx`
- `numpy`
- `pandas`
- `pandana`
- `polars`
- `pyarrow`
- `pyyaml`
- `pyrosm`
- `rasterio`
- `scipy`
- `shapely`
- `contextily`

Routing experiments additionally use `networkx` for route reconstruction and `gurobipy` for the sparse TSP models.

For reproducible runs, add a `requirements.txt`, `pyproject.toml`, or environment file before using this pipeline on a fresh machine.

---

# Optimization Example

Vietnam health-service candidate additions can be solved from a completed pipeline run with:

```bash
python solve_vietnam_health_max_cover.py
```

The script reads the Vietnam `population_*`, `sources_*`, and `distance_matrix_*` parquet outputs, filters the matrix to the 1 km coverage threshold lazily, fixes existing health amenities open, and uses `Research-Sandbox/approximated_tradeoff/src/mc_solvers.py` to choose 100 additional candidate sites with Gurobi.

---

# License

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project is documented as MIT licensed. The MIT license is a permissive license: you may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, provided the copyright notice and license text are included with substantial portions of the software.

If this folder is distributed independently, include a full `LICENSE` file next to this README.
