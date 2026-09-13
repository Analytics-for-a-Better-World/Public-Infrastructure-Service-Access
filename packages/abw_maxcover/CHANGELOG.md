# Changelog

All notable changes to `abw-maxcover` are recorded here.

## Unreleased

- Fix `parsimonious=True` being silently discarded on the Gurobi path: the
  per-facility penalty was set through `addVars(obj=...)` and then overwritten
  by `setObjective`, so saturating budgets returned redundant facilities.
- Fix `upper_bound` on parsimonious exact solves: both solver paths now add the
  worst-case penalty back before flooring, so the reported bound is a valid
  certificate on integer coverage and can no longer sit below the incumbent.
- Add exact-solver regression tests that run when `gurobipy` or Pyomo with
  HiGHS is available and are skipped otherwise.
- Fix `greedy_deployment_sequence` dropping pool facilities that add no
  coverage: they now follow the greedy order with a zero marginal gain, so the
  final step lists the whole pool and marginal gains sum to the objective.
- Fix `exact_pareto_curve(solver="pyomo")` silently dropping `result_callback`:
  `solve_pyomo_curve` now accepts the callback and invokes it after every
  solved budget, matching the Gurobi path.
- Fix `local_search_moves` meaning different things per method: it now always
  counts accepted local-search swaps. The plain greedy record reports zero
  instead of its construction step count, compact records report the swaps of
  the search they compacted instead of the number of dropped facilities, and
  regreedy records report the swaps of both local-search phases.

## 0.2.0 - 2026-07-19

- Promote the package from `Research-Sandbox` to `packages/abw_maxcover`.
- Document the distinction between complete value frontiers, nested deployment
  sequences, pointwise envelopes, and selected-budget refinement.
- Use one raw-CSR incremental state across greedy construction, zero-loss
  compaction, regreedy, local search, randomized construction, and bounded path
  relinking.
- Add callback-based checkpointing for long approximate and exact curves.
- Keep Gurobi, Pyomo, pandas, SciPy, and reporting dependencies optional.
- Add package-level CI, build validation, citation metadata, invariant tests,
  brute-force exact checks, and reproducible scaling benchmarks.

## 0.1.0 - 2026-06-01

- Initial incubator release used in the Timor-Leste and Vietnam studies.
