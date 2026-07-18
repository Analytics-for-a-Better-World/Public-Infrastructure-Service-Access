# abw-maxcover

`abw-maxcover` is the reusable maximum-cover optimization package incubated in
the Analytics for a Better World public-infrastructure service-access work. It
is built for experiments where demand points have weights, candidate facilities
cover subsets of demand, and the analyst needs exact or scalable approximate
coverage curves over many budgets.

The Python import name is `abw_maxcover`.

## Design principles

- Keep the package country-agnostic and data-pipeline-agnostic.
- Use compact dataclasses and primitive records for optimization results.
- Keep pandas out of the optimization core; convert records to DataFrames only
  in reporting scripts.
- Import solver packages only inside solver functions. Machines that only need
  heuristics should not need Gurobi, Pyomo, pandas, or SciPy.
- Reuse one incremental coverage engine across greedy construction, compacting,
  re-greedy refill, randomized construction, local search, path relinking,
  deployment sequencing, and exact-solver warm starts.
- Treat unsorted budget lists efficiently by solving in sorted order internally
  and returning results in the user-requested order.

## Installation

From this repository:

```powershell
python -m pip install -e Research-Sandbox\abw_maxcover
```

The core package requires only NumPy. Optional extras are available for sparse
local search, exact solvers, and development:

```powershell
python -m pip install -e Research-Sandbox\abw_maxcover[sparse,gurobi,dev]
```

## Module map

```text
abw_maxcover/
  instance.py           MaxCoverInstance and builders
  results.py            MaxCoverResult, MaxCoverCurve, comparison records
  exact.py              Gurobi and Pyomo exact Pareto solvers
  heuristics.py         scalable approximate Pareto orchestration
  deployment.py         optimize-then-greedy deployment sequencing
  pareto.py             public exact/approximate curve API
  validation.py         consistency checks
  io.py                 JSON/CSV-oriented primitive IO helpers
  _budgets.py           budget-order normalization
  _incremental_core.py  shared coverage-state and marginal-gain engine
```

The implementation is scalable because it keeps the current coverage state and
incremental gains close to the core algorithms. Adding, dropping, and swapping
facilities updates only affected demand and candidate rows. The same logic is
used by deterministic greedy, compact, re-greedy, randomized construction,
GRASP-style multi-starts, local search, path relinking, and deployment curves,
so improvements in the core benefit every higher-level strategy.

Path relinking is deliberately bounded for large solutions. Exhaustively
evaluating every exit--entry pair at every step has cubic worst-case growth in
the symmetric-difference size and is not suitable when budgets reach thousands
of facilities. The default heuristic therefore:

- ranks candidate exits by current loss and entries by current gain;
- evaluates exact swap deltas inside a 16-by-16 candidate beam;
- refreshes that ranking every eight accepted path steps; and
- stops after 64 path steps while retaining the best state visited.

Within each beam, exit losses, entry gains, and facility incidence rows are
computed once per path step. Exact overlap corrections use a reusable demand
mask rather than repeated pairwise set intersections. The best path position
is recorded and reconstructed once, avoiding full coverage-vector copies at
every incumbent improvement.

These controls are exposed as `path_relinking_candidate_width`,
`path_relinking_refresh_interval`, and `path_relinking_max_steps` on
`HeuristicConfig`. Set the width and step limit to `None` only for a small
instance where exhaustive relinking is intentionally required.

Long-running approximate curves can pass `result_callback` to
`approximate_pareto_curve`. The callback receives the best result immediately
after each internally sorted budget is complete, allowing experiment scripts
to persist frontier points and selected facilities without waiting for the
entire curve.

`HeuristicConfig` also exposes `local_search_max_moves` and
`local_search_time_limit_seconds`. These optional limits make the intensification
effort explicit on national instances; the incremental greedy result remains in
the portfolio and therefore preserves the standard weighted maximum-coverage
guarantee even when local search is truncated.

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

largest = max(curve.results, key=lambda result: result.budget)
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

The exact call requires the relevant optional solver dependency and license.
The package can still be imported and heuristic curves can still run without
those dependencies.

## Development checks

```powershell
python -m pytest -q Research-Sandbox\abw_maxcover
python Research-Sandbox\abw_maxcover\examples\small_working_example.py
```
