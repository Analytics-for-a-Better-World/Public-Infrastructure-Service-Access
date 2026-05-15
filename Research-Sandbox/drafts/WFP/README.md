# WFP scalable access pipeline draft

This is a first architecture draft for a lightweight, extensible routing and geocoding pipeline.

The guiding idea is a scoped `DataContext`: one run owns one data registry, one artifact runner,
one repository, and strategy registries for routing, geocoding, validation, and optimization.
Within that context, identical datasets are loaded once and reused by reference.

## Design rules

- Dataframes are not the contract; schemas and typed IDs are the contract.
- Heavy tables are loaded through `DatasetRegistry` and reused by `DatasetKey`.
- Decorators describe artifact intent; plain functions do the actual work.
- `ArtifactRunner` owns cache lookup, schema validation, persistence, and registration.
- Routing engines are strategies: Pandana, NetworkX, R5, OSRM, or Valhalla can be swapped.
- Geocoding is a sequence of stages: normalization, OSM amenity matching, Nominatim, reference data, fallback, validation, confidence.
- Optional dependencies are imported only inside adapters.
- Every serious run writes a YAML manifest with inputs, parameters, fingerprints, schemas, engine versions, and artifacts.
- Internal tables should prefer Parquet/Arrow; CSV/XLSX are interchange formats; YAML is for config and manifests.

## Package layout

```text
src/wfp_access/
  artifacts/      artifact specs, decorators, runner, fingerprints
  core/           context, registries, shared contracts
  data/           dataset keys, handles, registry, dataframe protocol
  geocoding/      geocoding stage contracts and pipeline
  optimization/   facility-location strategy contracts
  routing/        router strategy contracts and registry
  storage/        repository and storage codec contracts
examples/
  timor_leste_schools.yaml
tests/
  golden_minidata.py
tools/
  validate_shared_cache.py
```

## Shared cache validation

Real-data validation deliberately reuses the original pipeline cache root:

```text
C:\local\Download_Depot
```

The reengineered notebooks and validation harness should use the original country
configs and `CacheManager` for OSM, WorldPop, Natural Earth, parsed networks,
snapping caches, candidate caches, and distance matrices. This keeps original and
reengineered runs pointed at one reproducible cache instead of creating hidden
local state in the WFP draft folder.

See `SHARED_CACHE_VALIDATION.md` for the workflow and:

```powershell
py tools\validate_shared_cache.py inventory --output diagnostics\shared_cache_inventory.json
py tools\validate_shared_cache.py manifest --output diagnostics\shared_cache_manifest.yaml
py tools\validate_shared_cache.py restore-downloads
```

## Decorator principle

Decorators are metadata, not magic:

```python
@derived_artifact(name="distance_matrix", schema=DistanceMatrixSchema, format_role="table")
def compute_distances(origins, destinations, router):
    return router.route_many(origins, destinations)
```

The function can still be tested directly. In a pipeline, run it through the context:

```python
distances = ctx.runner.run(compute_distances, origins, destinations, router)
```

## Extension principle

New engines register against small contracts:

```python
ctx.routers.register("pandana", PandanaRouter)
ctx.routers.register("networkx", NetworkXRouter)
ctx.geocoders.register("nominatim", NominatimStage)
ctx.storage.register("parquet", ParquetCodec)
```

The core pipeline should never import `pandana`, `networkx`, `r5py`, `googlemaps`, `pandas`, or `polars` directly.
