# Consistency With `general_distances_per_country`

Checked against GitHub `origin/main` on 2026-05-25.

## Original Features Reflected Here

| Original capability | Promoted scalable module |
| --- | --- |
| Shared YAML manifest schema | `scalable_distances.manifest` |
| Shared cache validation workflow | `tools/validate_shared_cache.py` |
| OSM PBF and WorldPop path/URL naming | `scalable_distances.config.CountryDataSources` |
| Optional OSM/PBF backend direction | `scalable_distances.geospatial.detect_geospatial_backend()` reports `npyosmium` |
| Split matrix output keys and paths | `scalable_distances.matrix` |
| `combined`, `split`, `both` matrix modes | `scalable_distances.write_distance_matrix()` |
| Routing strategy boundary | `scalable_distances.routing` |
| Geocoding stage boundary | `scalable_distances.geocoding` |

## Local Original Change Not Yet On GitHub

The local original file
`Research-Sandbox/general_distances_per_country/countries/base.py` contains a
WorldPop extension that is not present on GitHub `origin/main` yet. The promoted
package includes the same source-resolution fields:

- `worldpop_dataset`
- `worldpop_release`
- `worldpop_version`
- `worldpop_resolution`
- `worldpop_constrained`
- `worldpop_url`
- `worldpop_path`

That means the promoted API can describe both current GitHub source naming and
the local WorldPop `global2`/override extension without depending on the original
implementation internals.

## Remaining Non-Equivalence

This folder is not yet a full production runner. It does not independently
download data, parse road networks, convert rasters, extract facilities, snap
points, or compute Pandana distances end to end. Those stages remain delegated to
the original implementation while the scalable package stabilizes the API,
manifest, cache, data, geocoding, routing, and matrix contracts.
