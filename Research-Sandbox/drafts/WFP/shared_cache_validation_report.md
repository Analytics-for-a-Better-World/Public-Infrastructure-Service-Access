# Shared cache validation report

## Cache Root

Both implementations use the original pipeline cache root through `CountryConfig.base_root` and `CacheManager`:

```text
C:\local\Download_Depot
```

## Restored Downloads

The deleted download cache was rebuilt from source with `py tools\validate_shared_cache.py restore-downloads`. A second restore pass printed `Using existing file` for every artifact, confirming unchanged downloads are not redownloaded.

## Shared YAML Manifest Agreement

Both implementations now use the shared top-level YAML schema:

```yaml
schema_version: wfp-access-manifest/v1
manifest_kind: ...
created_utc: ...
implementation: {}
code: {}
case: {}
cache: {}
inputs: {}
parameters: {}
intermediate_artifacts: {}
outputs: {}
diagnostics: {}
```

Validation artifacts:

- `diagnostics/shared_cache_manifest.yaml`: WFP/reengineered cache-inventory manifest.
- `diagnostics/original_shared_manifest_sample.yaml`: original-pipeline sample run manifest.

Both were checked to contain the same required shared fields and the same
`schema_version`. The original manifest also keeps legacy aliases
(`country_config`, `runtime_settings`, `resolved_parameters`, `input_files`) so
existing downstream scripts continue to work while new tooling uses the shared
fields.

| artifact role | path | size bytes |
|---|---|---:|
| download:geofabrik_pbf | `C:\local\Download_Depot\luxembourg_data\luxembourg-latest.osm.pbf` | 46680278 |
| download:worldpop_raster | `C:\local\Download_Depot\luxembourg_data\lux_ppp_2020.tif` | 2215890 |
| download:natural_earth_boundaries | `C:\local\Download_Depot\luxembourg_data\cache\boundaries\ne_10m_admin_0_countries.zip` | 4930492 |
| download:geofabrik_pbf | `C:\local\Download_Depot\east-timor_data\east-timor-latest.osm.pbf` | 17751588 |
| download:worldpop_raster | `C:\local\Download_Depot\east-timor_data\tls_ppp_2020.tif` | 8975646 |
| download:natural_earth_boundaries | `C:\local\Download_Depot\east-timor_data\cache\boundaries\ne_10m_admin_0_countries.zip` | 4930492 |
| download:geofabrik_pbf | `C:\local\Download_Depot\vietnam_data\vietnam-latest.osm.pbf` | 323467599 |
| download:worldpop_raster | `C:\local\Download_Depot\vietnam_data\vnm_ppp_2020.tif` | 196897622 |
| download:natural_earth_boundaries | `C:\local\Download_Depot\vietnam_data\cache\boundaries\ne_10m_admin_0_countries.zip` | 4930492 |
| download:geofabrik_pbf | `C:\local\Download_Depot\indonesia_nusa_tenggara_data\nusa-tenggara-latest.osm.pbf` | 172512654 |
| download:worldpop_raster | `C:\local\Download_Depot\indonesia_data\idn_ppp_2020_UNadj.tif` | 924379381 |
| download:natural_earth_boundaries | `C:\local\Download_Depot\indonesia_nusa_tenggara_data\cache\boundaries\ne_10m_admin_0_countries.zip` | 4930492 |

## Feasible Cross-Implementation Run Completed

Ran the original implementation case:

```powershell
py tools\validate_shared_cache.py run-original --case luxembourg_schools_agg10 --execute
```

Then executed the WFP notebook `notebooks/luxembourg_school_case_architecture.ipynb` against the same shared cache.

| metric | original from rebuilt cache | WFP architecture notebook | agreement |
|---|---:|---:|---|
| population points / targets | 4800 | 4800 | exact |
| existing schools | 390 | 390 | exact |
| total sources | 486 | 486 | exact |
| distance matrix rows | 876 | 876 | exact |
| candidate sites | 96 | 96 | exact |

The WFP notebook reuses the same current matrix cache produced by the original implementation, so repeated runs do not recompute or redownload unchanged source data.

## Differences vs Earlier Manuscript Snapshot

The rebuilt cache downloads current Geofabrik `latest` files. The manuscript snapshot recorded May 2026 earlier artifacts with Luxembourg school/source counts of 394/490 and aggregate matrix count 885. The rebuilt current snapshot gives 390/486/876. This is a data-version difference from a moving OSM extract, not a disagreement between implementations on the same cache.

## Remaining Work

- Implement WFP architecture cases for Timor-Leste health, Vietnam health, Nusa Tenggara candidate-to-school, and Luxembourg AED.
- Run the heavy original cases only when their runtime/storage budget is acceptable. The validation harness has their commands and expected manuscript summaries.
- Add row-level matrix comparators once each WFP case exists.
- Capture hashes in `shared_cache_inventory_hashed.json` when long checksum diagnostics are needed.
