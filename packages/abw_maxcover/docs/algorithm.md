# Algorithmic design

## Fixed instance

Let `I` be weighted demand, `J` candidate facilities, and `N(j)` the demand
covered by facility `j`. The package optimizes the weighted coverage function

```text
f(S) = sum(w[i] for i covered by at least one j in S).
```

This function is normalized, monotone, and submodular for nonnegative weights.
The full-universe greedy prefix therefore attains the classical
`1 - (1 - 1/p)^p >= 1 - 1/e` approximation guarantee at every budget `p`.
The guarantee is conditional on the supplied candidate set and coverage
relation; it does not certify the fidelity of an upstream geographic model.

## Sparse state

`MaxCoverInstance` stores both sides of the bipartite incidence relation in CSR
arrays:

- demand-to-facility: `ij_indptr`, `ij_indices`;
- facility-to-demand: `ji_indptr`, `ji_indices`;
- demand weights: signed 64-bit integers.

For a selected set `S`, the incremental state maintains a coverage count for
each demand point. Adding or removing a facility touches only its incidence
row. Reverse incidence identifies the candidate gains affected when a demand
point changes between covered and uncovered.

With `n` demand points, `q` candidates, `m` incidences, and `s` positive-gain
greedy selections, the current scan-based greedy implementation costs
`O(m + s q)` time and `O(n + q + m)` memory. A direct add or remove operation
costs `O(degree(j))`. One-swap local search still has a combinatorial worst-case
neighborhood, but each evaluated update uses local sparse degrees rather than
rebuilding coverage over all `n` demand points.

## Computational products

The package deliberately distinguishes:

1. a single-budget solution;
2. an exact pointwise frontier, whose optimal sets need not be nested;
3. a full-universe greedy sequence, whose prefixes are nested;
4. an optimize-then-greedy deployment sequence ending at a selected solution;
5. a pointwise approximation envelope, which is feasible at every budget but
   need not be one irreversible deployment sequence.

The fast complete-frontier mode runs greedy to saturation, deletes only
zero-loss redundant facilities, regreedily orders the compact support, and
returns the pointwise maximum of the original and reordered curves. Because the
original greedy prefix remains available, the envelope retains its guarantee.

Selected-budget refinement can add sparse first-improvement local search,
randomized restricted-candidate construction, and bounded path relinking. These
methods reuse the same add, remove, and exact swap primitives. They improve
fixed-budget quality at a higher computational cost and are not included in the
theoretical guarantee.
