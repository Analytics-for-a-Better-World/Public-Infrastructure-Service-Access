# Comparisons against GitHub `general_distances_per_country`

Reference read: https://github.com/Analytics-for-a-Better-World/Public-Infrastructure-Service-Access/tree/main/Research-Sandbox/general_distances_per_country

The GitHub `main` version describes a reusable distance pipeline that builds population-to-facility distance tables from WorldPop rasters, OSM road networks, existing facilities, and optional grid candidate sites. The documented core steps are: load config, download/reuse OSM and WorldPop, build road network, convert population raster to points, extract facilities, generate candidates, snap targets/sources to road nodes, combine sources, and compute a sparse target-source distance table.

## Comparison 1: GitHub Main vs Local `general_distances_per_country`

Local comparison command: `git diff --stat origin/main -- Research-Sandbox/general_distances_per_country` after fetching `origin main`.

| area | result |
|---|---:|
| modified files | 12 |
| insertions | 1334 |
| deletions | 229 |
| largest changed file | `run_pipeline.py` with 951 diff lines |

Changed files:

| file | status | main kind of change |
|---|---|---|
| `countries/base.py` | modified | country config extended for new runtime/data options |
| `distance_pipeline/cache.py` | modified | cache keys now include bbox and richer source/candidate signatures |
| `distance_pipeline/candidate_builder.py` | modified | candidate generation behavior expanded |
| `distance_pipeline/config_loader.py` | modified | more aliases/config resolution |
| `distance_pipeline/manifest.py` | modified | manifest records more resolved runtime information |
| `distance_pipeline/network.py` | modified | bbox and map/network support changes |
| `distance_pipeline/settings.py` | modified | settings now include candidate water override, basemap, bbox, map options |
| `distance_pipeline/snapping.py` | modified | small snapping behavior/interface changes |
| `distance_pipeline/source_tables.py` | modified | ID/table handling adjustments |
| `distance_pipeline/viz.py` | modified | richer context map rendering controls |
| `readme.md` | modified | documents new CLI/use cases |
| `run_pipeline.py` | modified | new composable source/destination layer orchestration |

## Comparison 2: Capability Surface

| capability | GitHub main documented baseline | local pipeline | WFP draft architecture |
|---|---|---|---|
| Core population-to-facility matrix | yes | yes | yes, via Luxembourg adapter notebook |
| OSM road network + Pandana | yes | yes | yes, wrapped as `RealPandanaRouter` |
| Candidate sites | yes | yes, with more explicit water/boundary controls | yes for Luxembourg case |
| Cached expensive stages | yes | yes, with richer cache keys | yes conceptually via `ArtifactRunner`; real case delegates heavy caches to existing pipeline |
| Custom table sources | documented on GitHub | expanded locally into composable layers | architecture supports shared data handles; not fully productionized as CLI yet |
| Custom table destinations | partial/generalized locally | yes locally via `--destinations table` | architecture design supports it; notebook basic tests prove same table can be source and target once in memory |
| Same table as source and target | not explicit in GitHub baseline | local code has ID prefixing helpers | explicitly tested in `architecture_basic_tests.ipynb` |
| Router interchangeability | GitHub has Pandana matrix and separate NetworkX routing experiments | local still operationally Pandana for matrix, NetworkX utilities for routing experiments | explicit strategy boundary; Pandana adapter implemented, NetworkX/R5 not yet implemented |
| Persistence formats | GitHub outputs Parquet and caches pickle | local still Parquet/pickle outputs, custom tables accept CSV/Excel/Parquet/GIS | repository/codec abstraction sketched; in-memory notebook repo used for tests |
| Geocoding pipeline | not a core GitHub pipeline feature | table coordinate loading exists | modular geocoding package scaffold exists; not yet tied into full real case |
| Visualization | context maps and routing figures | richer map-only/basemap options | Luxembourg notebook includes input map, snap histograms, matrix histogram |

## Comparison 3: Local Pipeline Additions over GitHub Main

Important local-only additions found in the diff:

- `--sources` and `--destinations` layer vocabulary with aliases for `population`, `amenities`, `candidates`, and `table`.
- `layer_signature()` and richer cache/run tags so matrices distinguish source/destination layer combinations.
- `with_prefixed_ids()` and layer-specific prefixes to avoid ID collisions when the same data can be used in multiple roles.
- Separate source-table and destination-table arguments.
- WorldPop runtime override options: year, dataset family, release, version, resolution, constrained flag, filename, URL, and local path.
- `--bbox` filtering for bounded experiments.
- `--map-only`, basemap choice, basemap alpha, and road overlay settings.
- Candidate water exclusion override and richer candidate cache naming.

## Comparison 4: Luxembourg School Case Results

The detailed numerical report is in `luxembourg_architecture_comparison.md`. Summary:

| use case | metric | earlier persisted output | current WFP architecture/cache | discrepancy |
|---|---:|---:|---:|---:|
| aggregate factor 10 | population rows | 4800 | 4800 | 0 |
| aggregate factor 10 | existing schools | 394 | 391 | -3 |
| aggregate factor 10 | all sources | 490 | 487 | -3 |
| aggregate factor 10 | distance matrix rows | 885 | 879 | -6 |
| full resolution | population rows | 102061 | 102061 | 0 |
| full resolution | existing schools | 394 | 391 | -3 |
| full resolution | all sources | 490 | 487 | -3 |
| full resolution | distance matrix rows | 65583 | not recomputed | current architecture-compatible full matrix cache missing |

The three earlier school IDs missing after the current deduplicated architecture path are:

- `185207530`
- `268229700`
- `1461344068`

Likely cause: current local/architecture path uses `facility_points_*_dedup_v1` and layer-signature source cache keys, while the earlier persisted outputs appear to come from an older source/facility cache version.

## Comparison 5: WFP Architecture Fit vs GitHub Pipeline

What the WFP draft improves:

- Data identity is explicit: `DatasetKey` and `DatasetHandle` make reuse testable.
- The same data object can be source and target without duplicate loading.
- Artifact decorators separate stage contract from execution/persistence.
- Router is a strategy boundary, so Pandana can be replaced later.
- Notebook examples are smaller and more modular than the monolithic pipeline walkthrough.

What is still missing or incomplete:

- Production storage codecs are sketched, but notebooks use an in-memory repository.
- NetworkX and R5 routing strategies are not implemented yet.
- The geocoding pipeline is scaffolded but not yet used in the real Luxembourg computation.
- Full parity with local `run_pipeline.py` composable source/destination CLI is not yet implemented as a WFP runner API.
- Full-resolution Luxembourg matrix was not recomputed through the new current cache key because that cache is absent and expensive to build.

## Recommended Next Comparisons

1. Run the full-resolution Luxembourg architecture matrix once with the current cache key, then compare row-level matrix differences against the earlier `65583`-row output.
2. Add a WFP notebook case for table-as-source and table-as-destination using a small synthetic table to validate source/target role reuse beyond the unit-style basic notebook.
3. Add a candidate-only Luxembourg use case matching the newer local README example: `--sources candidates --destinations population --candidate-grid-spacing-m 500 --max-total-dist 500`.
4. Add a NetworkX strategy prototype for a small bounded case and compare its distances to Pandana on identical snapped nodes.
