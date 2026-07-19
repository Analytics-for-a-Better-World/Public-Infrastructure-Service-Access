# abw_maxcover

`abw_maxcover` is the reusable maximum-cover optimization core used by the
Analytics for a Better World public-infrastructure service-access experiments.
It is deliberately independent of any country, thesis, notebook, or data
pipeline. Pipelines build `MaxCoverInstance` objects; this package solves or
approximates budgeted maximum-cover curves and returns typed dataclasses.

The planned PyPI distribution name is `abw-maxcover`; the Python import name is
`abw_maxcover`.

## What belongs here

- Canonical weighted maximum-cover instances in compact CSR-style arrays.
- Exact Pareto curves through solver modules.
- Scalable approximate Pareto curves through shared greedy, compact,
  re-greedy, randomized, GRASP-style, local-search, and path-relinking logic.
- Greedy deployment sequencing from a fixed largest-budget solution.
- Lightweight IO and validation helpers.

What does not belong here: country-specific preprocessing, GIS snapping,
paper/deck scripts, notebook state, pandas-heavy reporting, or application
code. Those layers should depend on `abw_maxcover`, not live inside it.

## Dependency policy

The core heuristic path imports only NumPy. Specialized packages are imported
late, inside the functions that need them:

- `gurobipy` is imported only by Gurobi exact solves.
- `pyomo` is imported only by Pyomo exact solves.
- SciPy is optional and used only by the sparse local-search accelerator.
- pandas is not used by the optimization core. Results expose dataclasses and
  primitive records; analysis scripts may convert those records to DataFrames.

This keeps the package importable on machines that only need pure Python/NumPy
heuristics.

## Module layout

```text
abw_maxcover/
  __init__.py           public API
  instance.py           MaxCoverInstance and builders
  results.py            result and curve dataclasses
  exact.py              Gurobi/Pyomo exact solvers
  heuristics.py         approximate Pareto orchestration
  deployment.py         optimize-then-greedy deployment curves
  pareto.py             exact/approximate curve API and comparison helpers
  validation.py         consistency checks
  io.py                 JSON/CSV-oriented primitive IO helpers
  _budgets.py           budget-order normalization
  _incremental_core.py  shared coverage-state and marginal-gain engine
```

The private `_incremental_core.py` module is intentionally central. Greedy,
compact, re-greedy, randomized construction, local search, path relinking,
deployment ordering, and exact-solver warm starts all reuse the same coverage
state, marginal-gain, add/drop/swap, and prefix logic.

Budget lists may be unsorted. Public functions sort internally for reuse and
solver efficiency, then report results back in the caller's requested order.

## Small working example

```python
import numpy as np

from abw_maxcover import (
    HeuristicConfig,
    approximate_pareto_curve,
    build_instance,
    greedy_deployment_sequence,
)

weights = np.array([10, 7, 5, 4, 3], dtype=np.int64)

# Demand-to-facility rows: demand i can be covered by these candidate ids.
ij = [
    [0, 1],
    [0, 2],
    [1, 2],
    [2, 3],
    [3],
]

# Facility-to-demand rows: facility j covers these demand ids.
ji = [
    [0, 1],
    [0, 2],
    [1, 2, 3],
    [3, 4],
]

instance = build_instance(
    weights,
    ij,
    ji,
    name="toy",
    validate_consistency=True,
)

config = HeuristicConfig(
    constructors=("greedy", "compact", "regreedy", "randomized"),
    randomized_repeats=2,
    local_search="first",
    seed=7,
)

curve = approximate_pareto_curve(instance, [3, 1, 2], config=config)
for result in curve.results:
    print(result.budget, result.method, result.objective, result.solution)

largest = curve.results[0]  # budget 3, because requested order is preserved
deployment = greedy_deployment_sequence(instance, largest.solution, budgets=[0, 1, 2, 3])
for result in deployment.results:
    print("deploy", result.budget, result.objective, result.solution)
```

Exact solves use the same instance and result schema:

```python
from abw_maxcover import GurobiConfig, exact_pareto_curve

exact = exact_pareto_curve(
    instance,
    [1, 2, 3],
    gurobi_config=GurobiConfig(time_limit_seconds=60, mip_gap=1e-8),
)
```

The Gurobi call requires the optional `gurobipy` dependency and a working
license. The package can still be imported and heuristic curves can still run
without it.
