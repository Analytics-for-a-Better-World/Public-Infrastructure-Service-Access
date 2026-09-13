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
- Fix `mip_gap` reporting: both exact paths now derive it from the coverage
  objective and coverage upper bound with one definition, and report `None`
  instead of `inf` when there is no incumbent. Gurobi's own gap is kept in
  `metadata["solver_mip_gap"]`.
- Fix the Pyomo path raising when the solver finds no feasible solution (for
  example fixed facilities exceeding the budget): solutions are loaded only
  when one exists, and the result reports the termination condition with no
  incumbent, matching the Gurobi path.
- Rename `assume_unique_sorted` to `assume_unique` on `build_instance` and
  `build_instance_from_facility_map`: rows never needed to be sorted, only free
  of duplicates. The old keyword still works with a `DeprecationWarning`.
  `validate_consistency=True` and `validate_instance` now also reject rows with
  duplicate entries, which would otherwise double-count weights in gains.
- Reject negative demand weights when constructing `MaxCoverInstance`. The
  greedy guarantee and the positive-gain filters assume nonnegative weights;
  the algorithm note previously described weights as signed.
- Honour `HeuristicConfig(randomized_repeats=0)`: randomized constructors are
  now skipped instead of silently running once.
- Remove SciPy remnants: the unreachable SciPy fallback in `run_heuristics`,
  the always-`None` `household_facility_matrix` argument of
  `SparseSwapLocalSearch`, and the claim in the packaged README that SciPy
  powers the sparse local search. The core is NumPy-only.

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
