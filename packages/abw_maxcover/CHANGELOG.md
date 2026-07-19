# Changelog

All notable changes to `abw-maxcover` are recorded here.

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
