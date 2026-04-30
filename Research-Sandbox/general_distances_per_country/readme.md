# Distance Pipeline

This project builds population points, generates candidate facility locations, and computes distance matrices using a road network.

The pipeline is designed to be modular and efficient, with caching at every stage.

---

# Overview

The pipeline performs the following steps:

1. Load country configuration
2. Build population points from WorldPop data
3. Load existing facilities
4. Generate candidate sites (grid-based)
5. Optionally plot a context map
6. Snap all points to the road network
7. Combine facilities and candidates into sources
8. Compute pairwise distances

The output is a full distance matrix between population points and sources.

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

Run the pipeline:

```bash
python run_pipeline.py <country> [options]
```

Examples:

```bash
python run_pipeline.py nld
python run_pipeline.py prt --sample-fraction 0.1 --save-map
python run_pipeline.py tls --aggregate-factor 10
```

---

# Countries

Supported country identifiers:

- `nld`, `netherlands`
- `prt`, `portugal`
- `tls`, `timor_leste`

---

# CLI Options

## General

```bash
--force-recompute
```
Ignore cache and recompute all steps.

```bash
--quiet
```
Reduce logging output.

---

## Population

```bash
--population-threshold FLOAT
```
Minimum population per pixel.

```bash
--sample-fraction FLOAT
```
Randomly sample population points.

```bash
--max-points INT
```
Maximum number of population points.

---

## Aggregation

```bash
--aggregate-factor INT
```
Aggregate WorldPop raster cells before converting to points.

```bash
--no-aggregate
```
Disable aggregation, even if defined in the country configuration.

Behavior:
- CLI overrides config
- If not provided, uses `cfg.aggregate_factor`
- If disabled, aggregation is not applied

---

## Distance computation

```bash
--max-total-dist FLOAT
```
Maximum allowed total distance (filters results).

---

## Candidate generation

```bash
--candidate-grid-spacing-m FLOAT
```
Spacing between candidate sites in meters.

If not provided, falls back to country configuration.

```bash
--candidate-max-snap-dist-m FLOAT
```
Maximum snapping distance when attaching candidates to the road network.

---

## Map output

```bash
--save-map
```
Save context map.

```bash
--show-map
```
Display map interactively.

```bash
--map-path PATH
```
Custom path for saved map (overrides default cache path).

```bash
--map-dpi INT
```
Resolution of saved map.

---

# Pipeline Details

## Population points

Population points are generated from WorldPop rasters using:

- population threshold filtering
- optional sampling
- optional aggregation

Aggregation groups raster cells before point creation, reducing resolution and improving performance.

---

## Candidate sites

Candidate sites are generated using a regular grid.

Parameters:
- grid spacing
- snapping distance to road network

---

## Snapping

All points are snapped to the nearest road network node:

- population points → targets
- facilities + candidates → sources

Distances include:

- point to road distance
- road network distance
- total distance

---

## Sources

Sources are constructed by combining:

- existing facilities
- generated candidate sites

This combined table is used in distance computation.

---

## Distance matrix

Distances are computed between:

- targets (population points)
- sources (facilities + candidates)

Output includes:

- `pop_id`
- `source_id`
- `pop_to_road_dist`
- `road_distance`
- `source_to_road_dist`
- `total_dist`

Optionally filtered using `--max-total-dist`.

---

# Caching

Each pipeline step is cached.

Cache keys depend on parameters such as:

- population threshold
- sampling fraction
- max points
- aggregation factor
- candidate grid spacing
- distance thresholds

Changing any parameter triggers recomputation of affected steps.

---

# Important assumptions

- All tables must contain an `ID` column
- DataFrames are indexed by `ID`
- IDs must be unique and consistent

---

# Notes

- The pipeline produces a full distance matrix
- No aggregation (e.g. nearest facility) is performed
- Downstream analysis or optimization is expected

---

# Dependencies

Main libraries:

- numpy
- pandas
- scipy
- pandana

---

# License

MIT
