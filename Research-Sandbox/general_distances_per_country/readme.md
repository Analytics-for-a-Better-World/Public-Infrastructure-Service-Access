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
py run_pipeline.py netherlands --map-only --candidate-grid-spacing-m 10000 --map-basemap voyager-no-labels --map-basemap-alpha 0.65
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
- `indonesia`, `idn`, `id`
- `nusa_tenggara`, `indonesia_nusa_tenggara`, `idn_nusa_tenggara`
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

```bash
--worldpop-year INT
```
Override the population year from the country config.

```bash
--worldpop-dataset global1
```
Select the WorldPop dataset family. `global1` uses the archived 2000-2020 tree. `global2` uses the 2015-2030 release/version tree.

```bash
--worldpop-release R2025A --worldpop-version v1 --worldpop-resolution 100m --worldpop-constrained true
```
Select a WorldPop Global2 release. For example, `--worldpop-dataset global2 --worldpop-year 2025 --worldpop-release R2025A --worldpop-version v1 --worldpop-resolution 100m --worldpop-constrained true` resolves to a constrained country raster named like `vnm_pop_2025_CN_100m_R2025A_v1.tif`.

```bash
--worldpop-filename FILE.tif
```
Use an explicit raster filename while keeping the generated WorldPop URL structure.

```bash
--worldpop-url URL
```
Use an explicit raster download URL.

```bash
--worldpop-path PATH
```
Use an existing local raster and skip downloading WorldPop. This is the strongest reproducibility option when an archived raster has already been stored locally.

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


## Source and Destination Layers

The default matrix is still WorldPop-derived population destinations against OSM amenity sources plus generated candidate sites when candidate generation is configured. The CLI now also accepts composable source and destination layers:

```bash
--sources amenities candidates
--destinations population
```

Supported layer values and aliases:

- `population` or `pop`: gridded WorldPop-derived points
- `amenities`, `amenity`, or `osm`: OSM features selected by `--amenity`
- `candidates`, `candidate`, or `grid`: generated regular-grid candidate sites
- `table` or `custom`: a user-provided table with point coordinates

The legacy `--source-layer amenity|table` option remains available. `--source-layer amenity` maps to the old default behavior, and `--source-layer table` uses only the supplied table as sources.

Custom point tables may be CSV, Excel, parquet, GeoJSON, GeoPackage, or shapefile. They should contain an `ID` column when stable identifiers matter and either `Longitude`/`Latitude`, `lon`/`lat`, `lng`/`lat`, `x`/`y`, or point geometry. IDs may be numeric or string-valued; the pipeline preserves them as identifiers when snapping to road nodes. Optional `population`, `demand`, `weight`, or `headcount` columns are used as demand weights when the table is used as a destination layer; otherwise a unit weight is assumed.

Source-table columns:

```bash
--source-table PATH
--source-lon-column longitude
--source-lat-column latitude
--source-id-column ID
```

Destination-table columns:

```bash
--destination-table PATH
--destination-lon-column longitude
--destination-lat-column latitude
--destination-id-column ID
```

If `--destinations table` is used without `--destination-table`, the pipeline reuses `--source-table`. This is convenient for table-to-table matrices.

When several source or destination layers are requested, the default output is one combined distance matrix with `source_type` and `target_type` columns. Use `--matrix-output-mode split` to compute and write one matrix for each nonempty source/destination layer pair, or `--matrix-output-mode both` to compute the pairs separately and also concatenate a combined file. For example, sources `amenities candidates` and destinations `population table` can produce separate `amenities -> population`, `amenities -> table`, `candidates -> population`, and `candidates -> table` parquet files. In `split` mode, the pipeline avoids building the full `[sources] x [destinations]` intermediate matrix. The run manifest lists every matrix path that was written.

Examples:

```bash
py run_pipeline.py vietnam --sources table --source-table "C:\path\stroke-facs-100-en.xlsx" --source-lon-column longitude --source-lat-column latitude --source-id-column TT --destinations population --aggregate-factor 10 --max-total-dist 150000
```

```bash
py run_pipeline.py netherlands --sources table amenities candidates --source-table institutions.xlsx --source-lon-column lon --source-lat-column lat --destinations table --destination-table institutions.xlsx --max-total-dist 100000
```

```bash
py run_pipeline.py nusa_tenggara --sources candidates --destinations table --destination-table "C:\local\GIT\route-the-meals\geocoding\draft\17_routing_targets_enhanced.csv" --destination-lon-column routing_lon --destination-lat-column routing_lat --destination-id-column source_id --bbox 118.8 -11.1 125.4 -7.1 --candidate-grid-spacing-m 2000 --candidate-max-snap-dist-m 1000 --max-total-dist 150000
```

```bash
py run_pipeline.py luxembourg --sources candidates --destinations population --candidate-grid-spacing-m 500 --max-total-dist 500
```

## Per-island distribution planning

`nusa_tenggara_distribution_pipeline.py` turns a generated candidate-to-school distance matrix into independent island-level distribution-planning outputs. It loads a run manifest, splits the school demand table by inferred island, filters the relevant sparse matrix rows, solves a 5-facility location-allocation model per island, assigns each school to an opened candidate, constructs greedy service-route geometries for each opened center, and writes island-specific outputs.

Example using the 2 km candidate matrix:

```bash
py nusa_tenggara_distribution_pipeline.py --manifest "C:\local\Download_Depot\indonesia_nusa_tenggara_data\outputs\run_manifest_pop_1_sample_1_max_none_agg_10_maxdist_150000_amenity_amenity_all-dst_table_17_routing_targets_enhanced-src_candidates_candidates_spacing_2000_maxsnap_1000.yaml" --school-table "C:\local\GIT\route-the-meals\geocoding\draft\17_routing_targets_enhanced.csv" --facilities-per-island 5 --max-candidates-per-target 50 --time-limit 300 --route-on-road true --max-road-route-legs-per-center 150 --output-prefix nusa_tenggara_distribution --figure-dir "C:\Users\joaqu\Dropbox\Apps\Overleaf\Real Life Distance Generator\figures"
```

Outputs are written below `<country data>/outputs/<output-prefix>/`. Each island receives its own folder with `selected_distribution_centers`, `demand_allocations`, `route_geometries`, `summary_statistics.csv`, and `diagnostics.log`. A combined `summary_statistics_all_islands.csv` and `pipeline_diagnostics.log` are written at the output-prefix root. To redraw the combined solution from already saved island outputs without rerunning the optimization, use `--plot-only true --output-prefix <output-prefix> --figure-dir <figure-dir>`.

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

```bash
--candidate-exclude-water {true,false}
```
Override whether generated candidate sites that fall on OSM water bodies are removed. If omitted, the country config value is used. Cache filenames record the effective choice as `no_water` or `water_allowed`.

Country configs can also control whether boundary points are included.

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
--map-only
```
Build and save the context map, then stop before source/target snapping and distance-matrix computation. This is useful for checking map styling or source/destination layers without running the full matrix stage. In map-only mode, generated candidate sites are drawn before road-node snapping so the command does not need to build the Pandana network.

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

```bash
--map-basemap {voyager-no-labels,voyager,positron-no-labels,positron}
```
Choose the context-map tile style. The default is `voyager-no-labels`, which provides a subtle but recognizable map background without competing with the plotted road network.

```bash
--map-basemap-alpha FLOAT
```
Opacity for the context-map basemap, from `0` to `1`. Default: `0.52`.

```bash
--map-roads {true,false}
```
Control whether context maps include the OSM road overlay. The default is `true`. For very large countries, `--map-roads false` can produce a quick visual check using the basemap plus population/source/candidate points without parsing the national road network.

```bash
--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT
```
Subset a run to a longitude/latitude bounding box. This is useful for island groups or regional runs within large countries. For OSM road-network loading, the bbox is passed to `pyrosm.OSM(..., bounding_box=[min_lon, min_lat, max_lon, max_lat])`, so network and road-map extraction can avoid parsing the full national graph. Bbox-specific node, edge, and road caches include the bbox in their filenames. For example, `--bbox 115 -12 128 -6` focuses Indonesia on Bali/Nusa Tenggara/Timor-like extents.

For a map-only visual check with a more visible but still restrained basemap:

```bash
py run_pipeline.py netherlands --map-only --candidate-grid-spacing-m 10000 --map-basemap voyager-no-labels --map-basemap-alpha 0.65
```

---

# Outputs

The CLI and notebook write parquet outputs under:

```text
<cfg.BASE_DIR>/outputs/
```

Output files:

- `population_<run_tag>.parquet`
- `targets_<run_tag>.parquet`
- `existing_sources_<run_tag>.parquet`
- `sources_<run_tag>.parquet`
- `distance_matrix_<run_tag>.parquet`
- `run_manifest_<run_tag>.yaml`

`targets_<run_tag>.parquet` contains the full destination layer used by the matrix. `population_<run_tag>.parquet` is kept for backward compatibility; when population is a destination layer it contains the snapped population targets, otherwise it mirrors the selected targets. `existing_sources_<run_tag>.parquet` contains non-candidate sources such as OSM amenities, table sources, or population-as-source layers. `sources_<run_tag>.parquet` contains the full source layer used by the matrix, including generated candidate sites when candidates are enabled. Use `sources_<run_tag>.parquet` for optimization plots or any model whose selected source IDs may include candidates.

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

This matters because Geofabrik `*-latest.osm.pbf` extracts are moving snapshots of OpenStreetMap. A run on a fresh machine may download different OSM road and facility data than an earlier run. WorldPop inputs are organized by product family, year, resolution, release, and version. The pipeline defaults to the older Global1-style country configs already in the repository, but runs can now pin Global2 products explicitly with `--worldpop-dataset global2`, `--worldpop-release`, `--worldpop-version`, `--worldpop-resolution`, and `--worldpop-constrained`. For complete control, use `--worldpop-url` or `--worldpop-path`; the manifest records the resolved URL/path and the exact local file bytes used for the run.

The distance matrix columns are:

- `target_id`
- `source_id`
- `source_nearest_node`
- `target_nearest_node`
- `target_to_road_dist`
- `road_distance`
- `source_to_road_dist`
- `total_dist`
- `target_type`, when multiple or typed destination layers are used
- `source_type`, when multiple or typed source layers are used

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
