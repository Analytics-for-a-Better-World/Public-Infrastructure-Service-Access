# Scalable General Distances Per Country

This folder is the promoted reengineering draft for
`Research-Sandbox/general_distances_per_country`. It keeps the original pipeline's
country-data semantics and shared cache discipline, but exposes the new work as a
small Python API first and a CLI second.

For a compact notebook demonstration of the new API, see
`notebooks/luxembourg_api_illustration.ipynb`.

The current goal is to provide a production-capable runner while keeping stable
contracts explicit: data source resolution, shared-cache manifests, lightweight
data reuse, interchangeable routing engines, modular geocoding, and scalable
matrix outputs.

## What Is Consistent With The Original

The promoted version mirrors the current GitHub `main` behavior in these areas:

- One shared cache location for original and reengineered runs.
- YAML manifests for reproducibility and version comparison.
- OSM PBF and WorldPop source naming conventions.
- Optional OSM/PBF backend detection, including `npyosmium`.
- Matrix output modes: `combined`, `split`, and `both`.
- Stable split matrix keys such as `distance_matrix_src_amenities_dst_population`.
- Pandas and Polars transparency for matrix splitting/writing.
- Lazy strategy boundaries for routing engines. NetworkX is available as a
  pure-Python shortest-path strategy; Pandana is imported only when the Pandana
  strategy is selected.
- Modular geocoding stages that operate on the same data objects used by routing.

The package now owns a full production runner for the standard country-distance
flow: download/reuse source artifacts, parse OSM roads, convert WorldPop rasters
to target points, extract OSM facilities, snap points, compute routes, and write
matrix outputs.

## Install For Local Use

From this folder:

```powershell
py -m pip install -e .
```

Optional extras:

```powershell
py -m pip install -e .[osm,geospatial,routing,optimization,dev]
```

The `osm` extra declares `npyosmium`, used for fast OSM/PBF parsing compatibility
with the newer original pipeline backend direction.

## API Quick Start

The primary interface is `scalable_distances`.

```python
from pathlib import Path

import pandas as pd

from scalable_distances import (
    create_context,
    describe_backends,
    describe_country_sources,
    write_distance_matrix,
)

ctx = create_context("luxembourg_school_run", root=Path("data/runs/luxembourg"))

backend_versions = describe_backends()

sources = describe_country_sources(
    country_slug="luxembourg",
    iso3="LUX",
    base_dir=Path(r"C:\local\Download_Depot\luxembourg_data"),
    worldpop_dataset="global1",
    worldpop_year=2020,
)

matrix = pd.DataFrame(
    [
        {
            "source_id": "school_1",
            "target_id": "pop_1",
            "source_type": "amenities",
            "target_type": "population",
            "total_dist": 820.0,
        },
        {
            "source_id": "candidate_1",
            "target_id": "pop_1",
            "source_type": "candidates",
            "target_type": "population",
            "total_dist": 910.0,
        },
    ]
)

outputs = write_distance_matrix(
    matrix,
    output_dir=Path("outputs"),
    run_tag="luxembourg_agg10",
    mode="both",
)

print(backend_versions)
print(sources)
print(outputs.paths)
```

`mode="combined"` writes one parquet matrix. `mode="split"` writes one parquet
file per `source_type` and `target_type` pair. `mode="both"` writes both forms.

## CLI Quick Start

The CLI is intentionally thin and mirrors the API.

```powershell
py -m scalable_distances.cli backends
```

Resolve source URLs and paths without downloading:

```powershell
py -m scalable_distances.cli sources `
  --country-slug luxembourg `
  --iso3 LUX `
  --base-dir C:\local\Download_Depot\luxembourg_data
```

Run the split-matrix smoke test:

```powershell
py -m scalable_distances.cli split-smoke --mode both
```

If installed as an editable package, the same commands are available through:

```powershell
scalable-distances backends
scalable-distances split-smoke --mode both
```

Run a full country pipeline:

```powershell
py -m scalable_distances.cli run `
  --country-slug luxembourg `
  --iso3 LUX `
  --base-dir C:\local\Download_Depot\luxembourg_data `
  --output-dir C:\local\Download_Depot\luxembourg_data\outputs `
  --run-tag luxembourg_networkx `
  --amenity school `
  --router networkx `
  --matrix-output-mode both
```

Use `--router pandana` only in deployments where Pandana is installed and
compatible with the NumPy version. Pandana is never imported by the package
unless that strategy is selected.

## Optimization Backends

Facility-location optimization is exposed through the package API and uses lazy
solver imports. The default solver mode is `auto`: try Gurobi first, and if
Gurobi is missing or not licensed, solve the same MILP through Pyomo's portable
HiGHS interface.

```python
from scalable_distances.optimization import (
    FacilityLocationConfig,
    solve_facility_location_by_island,
)

solution = solve_facility_location_by_island(
    "Sabu",
    demand_table=facility_inputs,
    candidate_table=facility_candidates,
    config=FacilityLocationConfig(solver="auto", setup_km=2.0, mip_gap=0.001),
)
print(solution.summary)
```

Install the open-source fallback with:

```powershell
py -m pip install -e .[optimization]
```

The fallback follows the Pyomo/HiGHS stack used in the MO-book TSP notebook:
`appsi_highs`/HiGHS with `highspy` as the Python backend. The implementation
uses Pyomo's direct APPsi `Highs` interface so test and notebook output capture
remain quiet. Use `solver="pyomo-highs"` to force the open-source path.

## Architecture

```text
src/scalable_distances/
  api.py          public Python API facade
  artifacts/     artifact specs, decorators, fingerprints, runner
  config/        country source-data and cache naming rules
  core/          run context and registries
  data/          dataset handles, schemas, frame protocol
  geocoding/     geocoding stage contracts and pipeline
  geospatial/    optional backend version detection
  io/            deterministic source download/reuse helpers
  matrix/        combined/split matrix output contracts
  network/       OSM PBF road network parsing
  optimization/  facility-location strategy contracts
  pipeline/      full production runner
  population/    WorldPop raster-to-point conversion
  routing/       router strategy contracts and registry
  storage/       repository and codec contracts
tools/
  validate_shared_cache.py
notebooks/
  architecture_basic_tests.ipynb
  luxembourg_school_case_architecture.ipynb
```

## Shared Cache Contract

Real-data validation should reuse the original pipeline cache root:

```text
C:\local\Download_Depot
```

The new package must not create hidden real-data caches inside this folder. It can
write diagnostics, manifests, smoke outputs, and small test artifacts under
`diagnostics/`, which is ignored by Git and regenerated on demand.

## Stability Status

Production-capable:

- API facade imports and backend detection compile.
- Country source resolution covers current original naming rules and the local
  WorldPop global2/local-path extension.
- Full runner performs download, network parsing, raster conversion, facility
  extraction, snapping, routing, and output writing.
- NetworkX router smoke test does not import Pandana.
- Split matrix output smoke checks pass for pandas and, when installed, Polars.
- The promoted folder is a sibling of the original pipeline and no generated
  diagnostics are committed.

Remaining implementation choices:

- Candidate-site generation is still a separate extension point rather than a
  default stage in the production runner.
- Full Luxembourg/Timor/Vietnam/Nusa Tenggara real-data benchmark runs should be
  rerun after each source-data refresh because Geofabrik and WorldPop snapshots
  are moving inputs.
