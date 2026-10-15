# Experimental persistent-memory search

`run_iterated_search` is an opt-in search for one facility budget. Existing
`run_heuristics` behavior and defaults remain unchanged. This is our composition
of existing package operators, not a reproduction of the authors' AILS-PR code.

```python
from abw_maxcover import IteratedSearchConfig, run_iterated_search

result = run_iterated_search(
    instance,
    70,
    config=IteratedSearchConfig(time_limit_seconds=60, seed=20261015),
)
```

The three stages can be tested separately:

| Configuration | Behavior |
|---|---|
| `perturb=False, adaptive=False` | Seeded restarts with a persistent elite pool |
| `perturb=True, adaptive=False` | Also perturb and repair the accepted reference solution |
| `perturb=True, adaptive=True` | Also adapt the number of removed sites |

The initial solution is greedy followed by the existing sparse single-swap local
search. The pool retains distinct site sets, protects its best solution, and
evicts a redundant non-best member when full. A random elite guide is used for
path relinking; improved relinked solutions receive local search.

Perturbation removes a seeded random subset of selected sites. The existing
randomized constructor repairs the retained set, followed by local search. Every
fifth iteration is a fresh restart by default. Late acceptance compares the new
objective with the current objective and a rolling history of accepted reference
objectives; the best-found solution is retained separately.

Adaptive perturbation changes removal strength by one according to the site-set
distance between the perturbed/local-searched solution and its reference. The
target defaults to 10% of the budget. Low distance increases strength; distance
above twice the target decreases it. This rule is an experimental choice.

Time is checked between iterations, so the last iteration can exceed the nominal
limit. Initial construction is always performed, including with a zero-second
limit. `max_iterations` supports fixed-work tests; deterministic reproducibility
requires that the wall limit does not bind. Configuration, iterations, elapsed
search time and final strength are returned in result metadata. An optional
callback supplies objective, distance, acceptance, restart and strength traces.

The implementation speedup underneath this search preserves candidate ordering
and fixed-work results in the tested cases. It caches marginal gains and computes
the correction for tentative removal in batches. It uses O(E) additional temporary
storage, where E is the number of coverage incidences; large dense instances may
need a different memory strategy. Wall-limited outputs can change after a speedup.

Validation distinguishes implementation equivalence from algorithm improvements:
the speedup uses a frozen pre-change local-search oracle, random/edge cases and
fixed-work city workloads. The opt-in algorithm has seeded reproducibility,
feasibility, pool and controlled acceptance/adaptation tests. Runtime and quality
claims belong to their saved experimental reports, not this API description.
