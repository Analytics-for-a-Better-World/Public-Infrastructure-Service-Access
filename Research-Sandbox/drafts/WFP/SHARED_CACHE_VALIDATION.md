# Shared Cache Validation Workflow

This draft validates that the original `general_distances_per_country` implementation
and the reengineered WFP architecture use one reproducible cache, not parallel hidden
state.

## Reference Basis

- Manuscript cases: `C:\Users\joaqu\Dropbox\Apps\Overleaf\Real Life Distance Generator`
- Original implementation: `C:\local\GIT\Public-Infrastructure-Service-Access\Research-Sandbox\general_distances_per_country`
- GitHub reference: https://github.com/Analytics-for-a-Better-World/Public-Infrastructure-Service-Access/tree/main/Research-Sandbox/general_distances_per_country
- Reengineered draft: `C:\local\GIT\Public-Infrastructure-Service-Access\Research-Sandbox\drafts\WFP`

## Shared Cache Rule

The shared download/cache root is the original pipeline's `CountryConfig.base_root`:

```text
C:\local\Download_Depot
```

Both implementations must use the country configs and `CacheManager` from the original
pipeline when working with real OSM, WorldPop, Natural Earth, network, snapping, and
matrix artifacts. The WFP architecture must not create a second real-data cache under
the WFP draft folder.

## Cache Classes

Downloaded artifacts:

- Geofabrik `.osm.pbf` extracts, e.g. `luxembourg-latest.osm.pbf`
- WorldPop rasters, e.g. `lux_ppp_2020.tif`
- Natural Earth boundaries archive, `ne_10m_admin_0_countries.zip`

Generated/reusable artifacts:

- parsed OSM nodes and edges
- classified roads
- OSM facility extracts and point conversions
- deduplicated facility points
- population point tables
- snapped population/source/candidate tables
- candidate grids and snapped candidates
- sparse distance matrices
- parquet outputs and YAML manifests

## Case Set from the Manuscript

The harness in `tools/validate_shared_cache.py` currently defines these cases:

| case | status in WFP architecture | purpose |
|---|---|---|
| `luxembourg_schools_full` | partially implemented; full matrix cache may need rebuild | descriptive school access, unaggregated |
| `luxembourg_schools_agg10` | implemented and executed | compact school access check |
| `timor_leste_health_agg8_10km` | not yet implemented as WFP case | max-cover input matrix |
| `vietnam_health_agg10_150km` | not yet implemented as WFP case | large max-cover input matrix |
| `nusa_tenggara_schools_candidates_2km` | not yet implemented as WFP case | table destinations and candidate sources |
| `luxembourg_aed_network_nodes` | not yet implemented as WFP case | network-node set covering |

## Commands

Inventory expected downloads and current file status:

```powershell
py tools\validate_shared_cache.py inventory --output diagnostics\shared_cache_inventory.json
```

Write the same inventory as the shared YAML manifest schema:

```powershell
py tools\validate_shared_cache.py manifest --output diagnostics\shared_cache_manifest.yaml
```

Include SHA-256 hashes for existing downloaded files:

```powershell
py tools\validate_shared_cache.py inventory --hash --output diagnostics\shared_cache_inventory_hashed.json
```

Restore missing downloaded files without overwriting existing ones:

```powershell
py tools\validate_shared_cache.py restore-downloads
```

Print original implementation commands without executing:

```powershell
py tools\validate_shared_cache.py run-original
```

Execute one original implementation case:

```powershell
py tools\validate_shared_cache.py run-original --case luxembourg_schools_agg10 --execute
```

## Agreement Criteria

Exact agreement is expected for:

- downloaded artifact paths
- downloaded artifact names
- configured URLs
- cache root
- output schemas for matrix columns
- row counts when both implementations use the same data snapshot and preprocessing flags
- nearest road node IDs when the same nodes/edges and snapping inputs are used

Numerical tolerance:

- coordinate equality: exact after reading cached tables, otherwise within `1e-9` degrees
  for lon/lat or `1e-6` projected meters for CRS-transformed values
- distance equality: exact for cached matrix reuse, otherwise absolute tolerance `1e-6`
  meters for repeated computations with the same backend
- summary percentages: `1e-6` absolute tolerance

Differences that must be explained:

- moving Geofabrik `latest` snapshots
- WorldPop release/version/path changes
- facility deduplication changes
- role-prefixed IDs versus unprefixed legacy IDs
- bbox or candidate-grid parameter changes
- missing-path filtering differences
- NetworkX/R5/Pandana backend changes

## Diagnostics to Record

For every validation run, record:

- cache inventory JSON
- original command
- WFP command/notebook cell or adapter used
- country config values
- downloaded paths, URLs, sizes, mtimes, optional hashes
- CRS and bounding box for every geometry table
- row counts for population, facilities, candidates, snapped tables, source tables,
  target tables, matrix
- network node/edge counts
- valid and invalid path counts
- matrix schema
- matrix summary statistics
- output manifests

## Shared Manifest Schema

Both implementations should write YAML with the same top-level schema:

```yaml
schema_version: wfp-access-manifest/v1
manifest_kind: pipeline_run
created_utc: '2026-05-15T...Z'
implementation:
  name: general_distances_per_country
  role: original
  root: C:/local/GIT/Public-Infrastructure-Service-Access/Research-Sandbox/general_distances_per_country
code:
  git_commit: ...
case:
  country_code: luxembourg
  country_name: Luxembourg
  iso3: LUX
cache:
  root: C:/local/Download_Depot
  country_dir: C:/local/Download_Depot/luxembourg_data
inputs:
  osm_pbf:
    role: download:geofabrik_pbf
    path: C:/local/Download_Depot/luxembourg_data/luxembourg-latest.osm.pbf
    url: https://download.geofabrik.de/europe/luxembourg-latest.osm.pbf
parameters:
  runtime_settings: {}
  resolved: {}
intermediate_artifacts: {}
outputs: {}
diagnostics: {}
```

The original implementation keeps legacy aliases such as `country_config`,
`runtime_settings`, `resolved_parameters`, and `input_files` so current downstream
scripts keep working, but the shared fields above are authoritative.

## Current Known Difference

For Luxembourg schools, the earlier persisted output has 394 existing schools while
the current deduplicated WFP/original-cache path has 391. The missing earlier IDs are
documented in `luxembourg_architecture_comparison.md`.
