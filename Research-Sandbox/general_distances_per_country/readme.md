# Distance Pipeline

Build sparse road-distance matrices from online geospatial data. The pipeline combines:

- OpenStreetMap road networks from Geofabrik `.osm.pbf` extracts
- WorldPop population rasters
- OSM amenities such as schools, clinics, hospitals, or other service points
- optional user-provided point tables
- optional regular-grid candidate locations

The main entry point is:

```powershell
py run_pipeline.py <country> [options]
```

---

# Important Pandana And NumPy Warning

Pandana is currently the routing engine used for road-network shortest paths. Pandana `0.7` and earlier wheels were built against the NumPy 1.x C API. If you install NumPy 2.x with Pandana `<=0.7`, importing or using Pandana can fail with binary-compatibility errors.

Use a NumPy 1.x environment unless you know that your Pandana build explicitly supports NumPy 2:

```powershell
py -m pip install "numpy<2"
py -m pip install pandana
```

At startup, the pipeline checks the installed package versions. If it detects `pandana<=0.7` with `numpy>=2`, it writes a clear warning to the console before the Pandana import is used by the pipeline.

---

# What The Pipeline Does

For a configured country or region, the CLI can:

1. Download or reuse the configured OSM PBF and WorldPop raster.
2. Extract a drivable road network from OSM.
3. Convert WorldPop raster cells to weighted population points.
4. Extract OSM amenities selected by `--amenity`.
5. Load custom point tables as sources, destinations, or both.
6. Generate regular-grid candidate locations.
7. Optionally remove candidate locations on water.
8. Snap sources and destinations to the road network.
9. Compute sparse or dense road-distance matrices.
10. Write parquet outputs and a YAML run manifest.
11. Optionally draw a context map.

The pipeline caches expensive intermediate stages under each country's configured data folder.

---

# Setup

There is no pinned environment file in this folder yet. A practical environment should include:

```text
numpy<2
pandas
geopandas
shapely
rasterio
pyrosm
pandana
polars
pyarrow
pyyaml
contextily
matplotlib
scipy
networkx
openpyxl
```

The optional `osmium` package enables the streaming OSM backend:

```powershell
py -m pip install osmium
```

Optimization and routing experiments that live next to this pipeline may need additional packages such as `gurobipy`, `pyomo`, or `highspy`. Those experiments should document their own extra requirements in their own README or notes.

---

# First Smoke Test

Run a small test before launching a full country matrix:

```powershell
cd C:\local\GIT\Public-Infrastructure-Service-Access\Research-Sandbox\general_distances_per_country
py run_pipeline.py timor_leste --sources amenities --destinations population --amenity hospital clinic doctors --max-points 10 --no-aggregate --max-total-dist 5000 --matrix-output-mode split
```

This keeps only 10 population points and writes a small split matrix. It is useful for checking the Python environment, OSM loading, WorldPop loading, snapping, Pandana, parquet output, and manifests.

---

# Countries

Country aliases are resolved by `distance_pipeline/config_loader.py` and the modules in `countries/`.

Common examples include:

- `timor_leste`, `timor-leste`, `tls`, `tl`
- `vietnam`, `viet_nam`, `vnm`, `vn`
- `luxembourg`
- `netherlands`, `nld`, `nl`
- `portugal`, `prt`, `pt`
- `laos`, `lao`, `la`
- `indonesia`, `idn`, `id`
- `poland`, `pol`, `pl`
- `switzerland`, `che`, `ch`

If a country module is missing, `load_cfg()` can attempt to generate one through the optional OpenAI helper. Existing country configs do not require an OpenAI key.

---

# Basic CLI Pattern

```powershell
py run_pipeline.py <country> `
  --sources amenities candidates `
  --destinations population `
  --amenity hospital clinic doctors `
  --random-seed 42 `
  --candidate-grid-spacing-m 5000 `
  --candidate-max-snap-dist-m 5000 `
  --max-total-dist 150000 `
  --matrix-output-mode split `
  --save-map
```

Use `py` on Windows if that is the Python launcher on your machine.

---

# Sources And Destinations

The pipeline is layer based. Sources and destinations can be selected independently.

Supported layer names:

- `population` or `pop`: WorldPop-derived demand points
- `amenities`, `amenity`, or `osm`: OSM amenities selected by `--amenity`
- `candidates`, `candidate`, or `grid`: generated grid candidates
- `table` or `custom`: a user-provided point table

Default behavior:

```powershell
--sources amenities candidates
--destinations population
```

Examples:

```powershell
py run_pipeline.py timor_leste --sources amenities --destinations population --amenity school
```

```powershell
py run_pipeline.py vietnam --sources table --source-table "C:\path\facilities.xlsx" --source-lon-column longitude --source-lat-column latitude --source-id-column ID --destinations population --aggregate-factor 10 --max-total-dist 150000
```

```powershell
py run_pipeline.py netherlands --sources table amenities --destinations table --source-table institutions.xlsx --destination-table institutions.xlsx --max-total-dist 100000
```

---

# Custom Point Tables

Custom point tables may be CSV, Excel, parquet, GeoJSON, GeoPackage, or shapefile.

The loader can auto-detect common coordinate columns such as:

- `Longitude` / `Latitude`
- `longitude` / `latitude`
- `lon` / `lat`
- `lng` / `lat`
- `x` / `y`

You can also specify them:

```powershell
--source-table PATH --source-lon-column longitude --source-lat-column latitude --source-id-column ID
```

```powershell
--destination-table PATH --destination-lon-column longitude --destination-lat-column latitude --destination-id-column ID
```

If `--destinations table` is used without `--destination-table`, the pipeline reuses `--source-table`. Optional `population`, `demand`, `weight`, or `headcount` columns are used as destination weights.

---

# Population Options

```powershell
--population-threshold FLOAT
```

Minimum raster value retained as a population point. Default: `1.0`.

```powershell
--sample-fraction FLOAT
```

Randomly sample retained population points. Must be in `(0, 1]`.

```powershell
--max-points INT
```

Cap the number of population points. Useful for smoke tests.

```powershell
--random-seed INT
```

Random seed used when `--sample-fraction` or `--max-points` selects a subset of population points. The default is `42`. With the same raster bytes, threshold, aggregation, sample fraction, max-points cap, and seed, the sampled population points are deterministic. The seed is included in population caches, snapped-population caches, matrix caches, output filenames, and the run manifest.

```powershell
--aggregate-factor INT
```

Aggregate raster cells by summing non-overlapping square blocks before converting to points.

```powershell
--no-aggregate
```

Disable aggregation even if the country config defines one.

WorldPop can be overridden with:

```powershell
--worldpop-year INT
--worldpop-dataset global1|global2
--worldpop-release R2025A
--worldpop-version v1
--worldpop-resolution 100m|1km
--worldpop-constrained true|false
--worldpop-url URL
--worldpop-path PATH
```

The strongest reproducibility option is `--worldpop-path`, because it points to a local raster whose exact bytes are recorded in the run manifest.

---

# Amenity Options

```powershell
--amenity hospital clinic doctors
```

Restrict OSM extraction to selected `amenity=*` values. If omitted, the current default amenities in `load_facilities()` are used.

```powershell
--deduplicate-amenities true|false
```

OSM amenities are deduplicated by default after non-point geometries are converted to representative points. When a nearby point/node and polygon/centroid have the same normalized name and amenity, the point/node feature is kept because it is usually the more intentional routing location.

---

# Candidate Grid Options

```powershell
--candidate-grid-spacing-m FLOAT
```

Grid spacing for generated candidate sites. If omitted, the country config value is used. If both are `None`, candidate generation is disabled.

```powershell
--candidate-max-snap-dist-m FLOAT
```

Maximum allowed distance from a candidate site to the road network. Candidates farther than this value are removed before matrix computation.

```powershell
--candidate-exclude-water true|false
```

Override whether candidate sites on OSM water polygons are removed. If omitted, the country config value is used.

Note: `--map-only` draws generated candidates before network snapping, so it is useful for visual grid checks but does not apply the candidate snap-distance filter.

---

# Matrix Options

```powershell
--max-total-dist FLOAT
```

In sparse mode, keep only rows whose `total_dist` is less than or equal to this value in meters. The same value also tightens the spatial prefilter used before road routing. In dense mode, entries above this cap are kept in the matrix but set to `inf`.

```powershell
--matrix-shape sparse|dense
```

- `sparse`: default long table with one row per retained finite source-target pair.
- `dense`: wide target-by-source matrix. The index is `target_id`, the columns are `source_id`, and unreachable paths are written as `inf`.

Dense mode deliberately computes every selected target/source pair. The user is responsible for choosing reasonable sources, destinations, bboxes, population caps, aggregation, and distance caps.

```powershell
--dense-component-matrices true|false
```

When dense mode is enabled, write three additional wide matrices:

- `origin_stitch`: source-to-road snapping distance
- `destination_stitch`: destination-to-road snapping distance
- `road_distance`: shortest road-network distance between snapped nodes

The dense total matrix is:

```text
destination_stitch + road_distance + origin_stitch
```

```powershell
--matrix-output-mode combined|split|both
```

- `combined`: compute one matrix for all selected source and destination layers.
- `split`: compute and write one matrix per nonempty source-layer/destination-layer pair.
- `both`: compute the pairs separately, write them, and also concatenate a combined matrix.

Use `split` when several source and destination layers would otherwise create a very large combined intermediate.

---

# Map Options

```powershell
--save-map
--show-map
--map-only
```

`--save-map` and `--show-map` trigger context-map construction. `--map-only` saves the map and stops before snapping and matrix computation.

```powershell
--map-path PATH
--map-dpi INT
--map-basemap voyager-no-labels|voyager|positron-no-labels|positron|osm
--map-basemap-alpha FLOAT
--map-roads true|false
--map-legend-loc "center left"
--map-legend-bbox-to-anchor 1.02 0.5
```

By default, the context-map legend is placed outside on the right with
`--map-legend-loc "center left" --map-legend-bbox-to-anchor 1.02 0.5`.
Use a standard Matplotlib legend location and anchor pair to move it, for
example `--map-legend-loc "lower left" --map-legend-bbox-to-anchor 0 0`.

For a fast visual check:

```powershell
py run_pipeline.py <country> --map-only --sources amenities --destinations population --amenity school --map-basemap voyager-no-labels --map-basemap-alpha 0.65
```

---

# Regional Runs

```powershell
--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT
```

Subset geometry layers to a longitude/latitude bounding box. For OSM road-network loading, the bbox is passed to the selected network backend so regional runs can avoid retaining a full national graph. Bbox-specific node, edge, and road caches include the bbox in their filenames.

---

# Network Backend

```powershell
--network-backend pyrosm|osmium|auto
```

- `pyrosm`: default behavior.
- `osmium`: optional streaming backend using Python `osmium`.
- `auto`: use `osmium` when installed, otherwise `pyrosm`.

The `osmium` backend is intended for large PBF extracts where full `pyrosm` extraction is memory intensive. In this version, selecting `--network-backend osmium` also uses a streaming `osmium` path for OSM amenity extraction. Amenity extraction is done in two passes: the first pass finds matching amenity nodes and ways, and the second pass collects only the node coordinates needed to reconstruct those matching ways. This avoids asking pyosmium to keep a full node-location index for the whole country just to extract a small amenity subset. Backend-specific caches are kept separate from the historical `pyrosm` caches. For publication runs, record the backend, bbox, aggregation, and distance cap in the run notes.

Large national extracts should usually avoid random sampling in final runs. Prefer the smallest aggregation that keeps the candidate target/source pair set computationally feasible, and report the retained WorldPop headcount from the log or run manifest. For example, health-facility accessibility capped at 100 km can be run as:

```powershell
py run_pipeline.py switzerland `
  --sources amenities `
  --destinations population `
  --amenity hospital clinic doctors `
  --aggregate-factor 10 `
  --max-total-dist 100000 `
  --matrix-output-mode split `
  --network-backend osmium
```

```powershell
py run_pipeline.py poland `
  --sources amenities `
  --destinations population `
  --amenity hospital clinic doctors `
  --aggregate-factor 50 `
  --max-total-dist 100000 `
  --matrix-output-mode split `
  --network-backend osmium
```

---

# Connectivity Diagnostics

```powershell
--diagnose-connectivity true|false
```

When enabled, the pipeline computes weak connected components of the loaded road graph before matrix computation. It logs the number of components and the largest component sizes, then adds a `component_id` column to:

- `targets_<run_tag>.parquet`
- `population_<run_tag>.parquet`
- `existing_sources_<run_tag>.parquet`
- `sources_<run_tag>.parquet`

Diagnostic runs add a `_connectivity` suffix to the output run tag so the labeled point tables are not overwritten by an otherwise identical non-diagnostic run. They also write:

- `connectivity_components_<run_tag>.parquet`

Component IDs are ordered by size, so `component_id = 0` is the largest road-network component. A source and destination being in different weak components means no road path can exist between them. Being in the same weak component is necessary but not always sufficient for directed routing, because one-way restrictions can still affect reachability.

```powershell
--snap-components COMPONENTS
```

Restrict snapping to selected weak connected components. Component IDs use the same ordering as the diagnostic output: `0` is the largest component, `1` the second largest, and so on. The option accepts comma-separated component IDs and ranges:

```powershell
--snap-components 0
--snap-components 0,2,5
--snap-components 0-3,7
```

If omitted, snapping uses all road-network nodes, which preserves the historical behavior. Use this option when small isolated road fragments should not be eligible snap targets, for example:

```powershell
py run_pipeline.py vietnam `
  --sources amenities `
  --destinations population `
  --amenity hospital clinic `
  --aggregate-factor 30 `
  --sample-fraction 0.02 `
  --snap-components 0
```

Component-restricted snapping changes snapped nearest nodes, so the selected components are included in snapped-table caches, matrix caches, output filenames, and the run manifest. The full road graph is still used for routing after snapping.

Small diagnostic example:

```powershell
py run_pipeline.py timor_leste `
  --sources amenities `
  --destinations population `
  --amenity hospital clinic doctors `
  --max-points 3 `
  --no-aggregate `
  --max-total-dist 5000 `
  --diagnose-connectivity true
```

This writes the usual source, destination, population, matrix, and manifest outputs, plus `connectivity_components_<run_tag>.parquet`. The snapped point outputs can then be grouped by `component_id` to see whether origins or destinations are isolated on small road-network components.

---

# Outputs

Outputs are written below:

```text
<cfg.BASE_DIR>/outputs/
```

Typical files:

- `population_<run_tag>.parquet`
- `targets_<run_tag>.parquet`
- `existing_sources_<run_tag>.parquet`
- `sources_<run_tag>.parquet`
- `distance_matrix_<run_tag>.parquet`
- `distance_matrix_src_<source>_dst_<target>_<run_tag>.parquet`
- `connectivity_components_<run_tag>.parquet` when `--diagnose-connectivity true`
- `run_manifest_<run_tag>.yaml`

`targets_<run_tag>.parquet` contains the full destination layer. `population_<run_tag>.parquet` is retained for backward compatibility. `existing_sources_<run_tag>.parquet` contains non-candidate source layers. `sources_<run_tag>.parquet` contains all source layers, including candidates when used.

Distance matrix columns:

- `target_id`
- `source_id`
- `source_nearest_node`
- `target_nearest_node`
- `target_to_road_dist`
- `road_distance`
- `source_to_road_dist`
- `total_dist`
- `target_type`
- `source_type`

All distance values are meters.

The run manifest records the input URLs or paths, resolved settings, output paths, file metadata, checksums, and the current git commit when available.

---

# Caching

Caches are written below:

```text
<cfg.BASE_DIR>/cache/
```

Cache keys include the relevant runtime settings, including population settings, random seed, aggregation, amenity filters, candidate settings, bbox, backend, distance threshold, and maximum total distance.

Sparse matrix construction also maintains a reusable road-node-pair cache below:

```text
<cfg.BASE_DIR>/cache/node_pair_distances/
```

This cache stores `(target_nearest_node, source_nearest_node, road_distance)` tuples induced by snapped source and destination layers. The final matrix outputs are unchanged, but later runs with overlapping snapped road-node pairs can reuse the cached road distances and compute only missing pairs. Unreachable road-node pairs are stored as `inf` internally so they are not recomputed; they are still omitted from the usual sparse distance matrix.

Use:

```powershell
--force-recompute
```

to rebuild matching cached stages.

---

# Notebook

`testdrive_general_country.ipynb` mirrors the main pipeline stages with inspectable intermediate objects. Keep notebook behavior consistent with the CLI when changing pipeline semantics.

---

# Other Experiments In This Folder

Some optimization, routing, and analysis scripts currently live next to the pipeline. They are not part of the core distance-matrix CLI. Their detailed setup, models, and case-study commands should be documented in specific README files or notes next to those scripts, not in this top-level pipeline README.

---

# License

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project is documented as MIT licensed. The MIT license is permissive: you may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, provided the copyright notice and license text are included with substantial portions of the software.

If this folder is distributed independently, include a full `LICENSE` file next to this README.
