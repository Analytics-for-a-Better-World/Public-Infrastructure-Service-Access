# Public API

The stable API is exported from `abw_maxcover`.

## Instances

- `MaxCoverInstance`
- `build_instance`
- `build_instance_from_facility_map`

The canonical representation uses consecutive integer ids internally. External
applications should retain their own id mapping in instance metadata or a
separate adapter table.

## Approximation

- `HeuristicConfig`
- `approximate_pareto_curve`
- `greedy_construct`
- `select_by_marginal_gain`
- `SparseSwapLocalSearch`
- `path_relink_fast`

Budgets are normalized and solved in ascending order internally. Results are
returned in the requested order. A `result_callback` can checkpoint each
completed budget.

## Exact optimization

- `GurobiConfig`
- `PyomoConfig`
- `exact_pareto_curve`
- `solve_gurobi_curve`
- `solve_pyomo_curve`

Exact solvers consume the same `MaxCoverInstance` as the heuristics. Solver
imports are delayed until the corresponding function is called. Exact results
retain status, incumbent objective, upper bound, MIP gap, selected facilities,
model time, and solve time.

## Deployment and comparison

- `greedy_deployment_sequence`
- `optimize_then_greedy_deployment`
- `compare_curves`
- `best_by_budget`

`MaxCoverResult`, `MaxCoverCurve`, and `CurveComparison` convert to primitive
records for downstream presentation. pandas is not used in the optimization
core.
