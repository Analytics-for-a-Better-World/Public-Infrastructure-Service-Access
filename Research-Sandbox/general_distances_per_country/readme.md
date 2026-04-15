````markdown
# Distance Pipeline CLI

This command line entry point runs the full distance pipeline for a selected country configuration.

It orchestrates:

- downloading required source data
- loading and caching the OSM road network
- converting WorldPop raster data to population points
- extracting health facilities from OSM
- generating candidate facility sites
- building a context map
- snapping sources and targets to network nodes
- computing a source to target distance matrix

The script is intended to be the main executable wrapper around the `distance_pipeline` package.

---

## What the pipeline does

Given a country code such as `tls`, `prt`, or `nld`, the pipeline:

1. loads the corresponding country configuration
2. downloads OSM and WorldPop input files if needed
3. builds or loads cached network data
4. classifies roads for map visualization
5. converts population raster cells into point targets
6. loads existing health facilities
7. converts non point facility geometries to points
8. builds candidate sites and snaps them to the network
9. produces a context map
10. snaps population and facilities to network nodes
11. computes network distances from sources to targets
12. stores intermediate and final outputs in cache

---

## Main entry point

The script exposes a CLI interface through:

```python
main(country_code: str, settings: PipelineSettings) -> None
````

and is normally run via:

```bash
python path/to/script.py <country_code> [options]
```

Example:

```bash
python run_pipeline.py tls --save-map --show-map --candidate-grid-spacing-m 5000
```

---

## Required package structure

The script depends on the following internal modules:

```text
countries.base
distance_pipeline.cache
distance_pipeline.candidate_builder
distance_pipeline.config_loader
distance_pipeline.distance_matrix
distance_pipeline.facilities
distance_pipeline.io
distance_pipeline.network
distance_pipeline.pipeline_support
distance_pipeline.population
distance_pipeline.settings
distance_pipeline.snapping
distance_pipeline.source_tables
distance_pipeline.viz
```

---

## Inputs

The pipeline expects a valid country configuration resolved by:

```python
load_cfg(country_code)
```

The configuration must provide, at minimum:

* country name
* base directory
* projected CRS
* OSM PBF URL and local path
* WorldPop URL and local path
* plotting title
* distance threshold for the matrix

Typical examples of accepted country codes:

* `portugal`
* `netherlands`
* `timor_leste`
* `prt`
* `nld`
* `tls`

---

## Outputs

Depending on the settings and cache state, the pipeline produces:

* downloaded source files
* cached network nodes and edges
* classified roads
* population points
* health facilities
* point converted facilities
* candidate sites
* snapped targets and sources
* source target distance matrix
* optional context map image

---

## Command line arguments

### Positional argument

#### `country_code`

Country config module name.

Example:

```bash
python run_pipeline.py tls
```

---

### Optional arguments

#### `--force-recompute`

Ignore caches and rebuild all cached steps.

```bash
python run_pipeline.py tls --force-recompute
```

#### `--save-map`

Save the context map to disk.

```bash
python run_pipeline.py tls --save-map
```

#### `--show-map`

Display the context map interactively.

```bash
python run_pipeline.py tls --show-map
```

#### `--map-path`

Optional custom path for the saved map.

```bash
python run_pipeline.py tls --save-map --map-path figures/tls_map.png
```

#### `--map-dpi`

DPI used when saving the map.

```bash
python run_pipeline.py tls --save-map --map-dpi 400
```

#### `--population-threshold`

Minimum population threshold for raster to point conversion.

```bash
python run_pipeline.py tls --population-threshold 5
```

#### `--sample-fraction`

Sampling fraction for raster to point conversion, must be in `(0, 1]`.

```bash
python run_pipeline.py tls --sample-fraction 0.25
```

#### `--max-points`

Maximum number of population points to retain.

```bash
python run_pipeline.py tls --max-points 50000
```

#### `--max-total-dist`

Optional maximum total distance retained in the matrix output.

```bash
python run_pipeline.py tls --max-total-dist 25000
```

#### `--candidate-grid-spacing-m`

Optional candidate site grid spacing in meters.

```bash
python run_pipeline.py tls --candidate-grid-spacing-m 5000
```

#### `--candidate-max-snap-dist-m`

Optional maximum snapping distance for candidate sites, in meters.

```bash
python run_pipeline.py tls --candidate-max-snap-dist-m 1000
```

#### `--quiet`

Reduce console output.

```bash
python run_pipeline.py tls --quiet
```

---

## Validation rules

The CLI arguments are validated before the pipeline runs.

The following conditions apply:

* `population_threshold >= 0`
* `0 < sample_fraction <= 1`
* `max_points > 0` when provided
* `max_total_dist > 0` when provided
* `candidate_grid_spacing_m > 0` when provided
* `candidate_max_snap_dist_m > 0` when provided
* `map_dpi > 0`

Invalid values raise `ValueError`.

---

## Pipeline flow

The execution order is approximately:

```text
load country config
↓
initialize cache manager
↓
download PBF and WorldPop files
↓
load or build OSM network data
↓
build Pandana network
↓
classify roads
↓
convert WorldPop raster to population points
↓
load health facilities
↓
convert facility geometries to points
↓
build candidate sites
↓
build map facility layer
↓
plot context map
↓
snap population targets to nodes
↓
snap existing facilities to nodes
↓
combine existing and candidate sources
↓
compute distance matrix
↓
set known categories
↓
print summary
```

---

## Caching behavior

The pipeline uses `CacheManager` extensively to avoid recomputing expensive steps.

Cached steps include:

* network data
* classified roads
* population points
* health facilities
* point converted facilities
* snapped targets
* snapped hospitals
* distance matrix

You can force regeneration with:

```bash
python run_pipeline.py tls --force-recompute
```

---

## Candidate sites

Candidate sites are generated through:

```python
build_candidate_sites(
    cfg=cfg,
    settings=settings,
    cache=cache,
    nodes=nodes,
)
```

Their spacing is resolved with:

```python
resolve_candidate_grid_spacing(cfg, settings)
```

These candidates may then be combined with existing facilities before distance computation.

When candidates are present, the matrix cache filename is modified to include:

```text
_with_candidates
```

This prevents accidental collision with an existing only matrix.

---

## Context map

The script can generate a context map through:

```python
plot_context_map(
    roads=roads,
    population_points=population_points,
    health_centers=map_facilities,
    title=cfg.PLOT_TITLE,
    output_path=context_map_path if settings.save_context_map else None,
    dpi=settings.context_map_dpi,
    show=settings.show_context_map,
    verbose=settings.verbose,
)
```

Map path resolution is handled by:

```python
build_context_map_path(...)
```

The displayed facilities layer may include both:

* existing health centers
* candidate sites

depending on the pipeline settings and generated outputs.

---

## Distance matrix

The final matrix is built by:

```python
compute_distances(
    targets=population,
    sources=sources_for_matrix,
    distance_threshold_largest=cfg.DISTANCE_THRESHOLD_KM,
    network=network,
    max_total_dist=settings.max_total_dist,
    verbose=settings.verbose,
)
```

This matrix uses:

* snapped population points as targets
* existing and optionally candidate facilities as sources
* network distances over the Pandana graph

The resulting dataframe is then post processed with:

```python
set_known_categories(matrix_df)
```

---

## Console output

In verbose mode, the script reports useful diagnostics such as:

* country being processed
* timing for major steps
* number of existing sources
* number of candidate sources
* context map path
* total runtime

It also prints:

```python
print(matrix_df.head())
```

so you immediately see the first rows of the computed distance matrix.

---

## Example runs

### Basic run

```bash
python run_pipeline.py tls
```

### Save the context map

```bash
python run_pipeline.py tls --save-map
```

### Save and display the map

```bash
python run_pipeline.py tls --save-map --show-map
```

### Recompute everything from scratch

```bash
python run_pipeline.py tls --force-recompute
```

### Use a custom candidate spacing

```bash
python run_pipeline.py tls --candidate-grid-spacing-m 5000
```

### Limit population points for testing

```bash
python run_pipeline.py tls --max-points 10000 --sample-fraction 0.2
```

### Quiet mode for batch runs

```bash
python run_pipeline.py tls --quiet
```

---

## Dependencies

External dependencies used directly in this script include:

* `argparse`
* `pathlib`
* `time`

It also relies on the dependencies required by the imported `distance_pipeline` modules, typically including geospatial and network packages such as:

* `geopandas`
* `pandas`
* `numpy`
* `pandana`
* `contextily`
* OSM parsing utilities

---

## Assumptions

This script assumes:

* the selected country configuration is valid
* URLs for PBF and WorldPop data are reachable
* the input data can be downloaded or already exists locally
* the downstream pipeline functions handle their own internal validation
* the cache paths are writable

---

## Typical use case

This pipeline is especially useful for accessibility analysis and facility location workflows where you want to:

* represent population demand spatially
* combine current facilities with candidate sites
* compute travel distance based coverage
* generate a map for quality checking and reporting

---

## Notes for developers

A few design choices are worth noting:

* `settings_from_args()` cleanly separates CLI parsing from runtime settings
* `CacheManager` keeps expensive geospatial steps reproducible and efficient
* map generation is optional and controlled entirely through settings
* candidate and existing sources are merged late, just before matrix computation
* cache paths depend on relevant parameter choices, which helps avoid accidental reuse

---

## Suggested future improvements

Potential extensions for this entry point include:

* exporting the final matrix to parquet or csv
* optional logging to file
* structured progress reporting
* dry run mode
* explicit configuration validation before downloads
* optional profiling summary per stage
* optional map suppression when running headless by default

---

````

A couple of code improvements stand out in this script.

First, `print(matrix_df.head())` is unconditional. If you want `--quiet` to be truly quiet, it should probably be wrapped in:

```python
if settings.verbose:
    print(matrix_df.head())
````

Second, `candidate_grid_spacing_m` is resolved and used for the map path, but the resolved value is not otherwise printed unless verbose mode is extended. It may be useful to report it explicitly in the final summary.

Here is an updated **README.md** section you can append, covering:

* your exact CLI example
* how to display a PNG in markdown
* the role of OpenAI in your pipeline (country config generation)

---

````markdown
## Example usage

Run the full pipeline for Timor Leste, display the map, include candidate sites, and restrict distances:

```bash
python run_pipeline.py tls --show-map --candidate-grid-spacing-m=5000 --max-total-dist=300
````

This will:

* generate candidate facility locations on a 5 km grid
* compute distances up to 300 km
* display the context map interactively

---

## Example output map

If you save a map using `--save-map`, you can include it in documentation like this:

```markdown
![Context map example](figures/east-timor-latest.osm_context_map_resolution_5000m.png)
```

Tips:

* use relative paths so the image renders on GitHub
* keep images in a `figures/` or `docs/` folder
* prefer PNG for maps, as it preserves detail

---

## OpenAI integration

This project uses the OpenAI API to automatically generate country configuration modules.

### Purpose

The function:

```python
generate_country_config_module(...)
```

creates a new country config file with:

* ISO codes
* country name and slug
* projected CRS (EPSG)
* WorldPop filename

This avoids manual setup when adding new countries.

---

### How it works

The pipeline sends a structured prompt to an OpenAI model:

```text
Return only valid JSON.
...
```

The model returns a JSON payload like:

```json
{
  "iso3": "TLS",
  "iso2": "TL",
  "country_name": "Timor-Leste",
  "country_slug": "timor_leste",
  "projected_epsg": 32751,
  "worldpop_filename": "tls_ppp_2020.tif"
}
```

This is then:

1. parsed and validated
2. normalized (for example EPSG extraction)
3. written into a Python module

---

### Generated module structure

Each generated config looks like:

```python
CFG = build_config(
    {
        'iso3': 'TLS',
        'iso2': 'TL',
        'country_name': 'Timor-Leste',
        'country_slug': 'timor_leste',
        'projected_epsg': 32751,
        'distance_threshold_km': 300.0,
        'geofabrik_region': 'asia',
        'worldpop_filename': 'tls_ppp_2020.tif',
        'plot_title_suffix': 'roads by class, population points, and health facilities',
        'candidate_grid_spacing_m': 5000.0,
        'candidate_max_snap_dist_m': 5000.0,
    }
)
```

---

### Robust EPSG parsing

The helper:

```python
parse_epsg(value)
```

ensures that outputs like:

* `32648`
* `"32648"`
* `"EPSG:32648"`

are all safely converted to:

```python
32648
```

---

### Why use OpenAI here

* removes manual lookup of CRS and naming conventions
* ensures consistent config structure across countries
* accelerates onboarding of new geographies
* reduces human error in repetitive setup

---

### Notes

* The model is constrained to return **JSON only**
* No free text is accepted
* Validation happens after parsing
* Existing configs are not overwritten unless explicitly allowed

---

## Typical workflow with OpenAI

```bash
# 1. Generate config
python -c "from your_module import generate_country_config_module; generate_country_config_module('laos', 'countries')"

# 2. Run pipeline
python run_pipeline.py laos --save-map
```

---

## Design philosophy

OpenAI is used only for:

* metadata generation
* configuration scaffolding

It is **not used** for:

* geospatial computation
* routing
* distance calculation
* map rendering

All analytical steps remain fully deterministic and reproducible.

```

---
```
