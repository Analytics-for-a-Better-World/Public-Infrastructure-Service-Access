# Optional solver-free research strategies

`ResearchSearchConfig` and `run_research_search` expose the strongest recorded
Python core from the geographic Kings development experiments. Existing package
strategy defaults are unchanged. Search needs NumPy, not Java or Gurobi.

```python
from abw_maxcover import ResearchSearchConfig, run_research_search

result = run_research_search(
    instance,
    80,
    config=ResearchSearchConfig(seed=171, time_limit_seconds=60),
    facility_coordinates=coordinates,
    facility_neighbors=radius_neighbors,
)
```

The preset uses eight random construction/descent starts, an elite pool of up to
40 solutions, adaptive random/concentric perturbations, threshold acceptance,
faithful Java-style local tie handling, dynamic path relinking, and residual
component recombination. Small components are optimized by exact-cardinality
subset menus; large components retain feasible parent choices. This combination
is an experimental adaptation, not a verbatim whole-Java trajectory reproduction.
All intermediate coverage scores use the original input graph and weights.

The entry point requires `1 <= budget < n_facilities`, nonnegative weights whose
sum fits int64, finite latitude/longitude coordinates of shape `(n_facilities,2)`,
and ordered radius neighborhoods. Every neighborhood must contain its own site
and consist of sorted, unique integer facility IDs. Neighborhoods guide search;
they must not be confused with demand coverage rows. Road distances belong in
the coverage incidence; changing the proximity guide does not change the score.
`abw_maxcover.java_ils.load_java_sparse` reads the existing ordered Java sparse
snapshot when that format is available.

`initial_starts`, `seed_capacity`, `faithful_local`, `dynamic_relink`,
`component_relink`, `exact_component_cap`, `seed`, and the time limit are exposed
on the config. Other inherited exploratory options are disabled by default.
`iterations` with an ample time allowance and `fake_clock=True` supports finite
trajectory checks, not a competitive timing protocol. Different time allowances
can change threshold acceptance and therefore the search trajectory.

## Optional near-neutral candidate-pool repair

For a complete run, set `pool_repair=True`; the default `repair_fraction=.1`
reserves the final tenth of the SAME total allowance. The first phase receives
less time, so preserving its result does not guarantee matching a core given the
entire allowance. `repair_extra_cap=24` controls alternative candidates.

An existing solution can instead be improved explicitly:

```python
from abw_maxcover import repair_saved_solution

repaired = repair_saved_solution(
    instance, list(result.solution), time_limit_seconds=1, extra_cap=24
)
```

The repair ranks exact single-swap changes separately for each outgoing site,
selects distinct promising alternatives, and removes only dominated extras.
Every original site stays in the pool. Bounded variable-elimination DP chooses
an exact-cardinality subset using original integer coverage. Width, entry,
operation, matrix and deadline caps protect resources. An incomplete calculation
retains the parent. A completed calculation is exact only in its chosen pool;
no global optimality certificate is produced. Diagnostics are in
`metadata['pool_search']`; no strict improvement is also a valid outcome.

## Preparation caching and CEM/beam research tools

`abw_maxcover.regional_repair.repair_region` accepts `memo=RepairMemo(capacity)`
and `cache_mode='prepare'`, `'exact'`, or `'both'` (default `'off'`). Immutable
model snapshots invalidate the cache on change. Returned site lists are private.
Preparation expiry is not cached. Only completed, untruncated, no-improvement
DP results may be reused as conditional proofs. `beam=None` disables truncation.
The workspace is sequential and not thread safe; use one memo per worker.

`abw_maxcover.cem_beam.BeamSession` retains a live queue, unfinished batch,
dominance map, model, RNG and memory-loss bound across `run` calls. Each call
accepts a finite monotonic deadline and a new expansion allowance. It is an
in-memory object, not an on-disk restart checkpoint. Inputs are integer coverage
masks and nonnegative integer demand weights; selected indices refer to the
ordered mask list. CEM's clipped, smoothed frequencies are fitted only when the
retained elite archive changes. This changes learning behaviour as well as cost.
Probabilities guide scheduling and samples; they never enter pruning bounds.
The APIs are low-level experimental tools rather than a new default strategy.

```python
from time import perf_counter
from abw_maxcover.cem_beam import BeamSession

session = BeamSession([1, 2, 3], [2, 3], 1, [0], agenda_limit=1024)
first = session.run(perf_counter() + .1, max_expansions=1)
continued = session.run(perf_counter() + .1, max_expansions=100)
```

Bounds concern the supplied model, or a fixed regional model after removing
already-fixed coverage. Discarded agenda states remain represented by a lost
upper bound; exhaustion after memory loss does not imply certification. Use
`certified`/`gap`, not the stop reason alone. Search counters are cumulative per
session; aggregate the last snapshot, or differences between calls.
`session.active_seconds` excludes suspended time; `stats.seconds` includes it.

## Evidence and limits

Recorded geographic Kings p=80 development runs used Windows 11, Python 3.14.6,
NumPy 2.4.6, CPU affinity `[0]`, and one BLAS thread. Historical manifests record
24 logical processors and RAM; the CPU model is not in those manifests and is
not reconstructed from today's hardware. Fresh Python timing includes algorithm
setup/search/callbacks, excludes common input loading/imports, and selects only
independently rescored events at or before the 60-second cutoff.

| Scope | Coverage | Evidence |
|---|---:|---|
| Best fresh Python 60-second run | 2,467,674 | Seed 2026121102, path relinking event at 52.5328665 s |
| Own saved solution plus extra-time pool repair | 2,467,690 | Five simultaneous replacements; not a fresh 60-second result |
| Strongest separately recorded Java run | 2,467,701 | Feasible reference, not a certified optimum |

Coverage is population weight served out of the Kings graph's total input weight
2,504,700. The frozen-source hash and final port hashes are recorded separately
in `research-source-provenance.json`. Port hashes normalize text line endings to
LF; historical source hashes bind the original experiment bytes. The released
wrapper adds strict validation and optional finishing; the historical result is
not a benchmark of that wrapper.
The frozen best core is checked on two synthetic 24-iteration trace fixtures.
Finite trace agreement does not prove universal implementation equivalence.

The subsequent CEM screen comprised 24 extra-time runs: three own saved parents,
two seeds, four versions, 20 seconds each. Every final score was unchanged.
Selective descent ran 545 times, completed 541 times, and recovered 265 of 512
worse starts to the parent score without exceeding it. Changed-only fitting
performed 3,650 fits versus 43,936 for the original; expanded states were
714,143 versus 689,477. An unrelated scheduling experiment was active at launch,
and trajectories/policies differed: these are not isolated speed estimates.
The machine snapshot for that screen records an Ultra 9 285HX, 24 cores/logical
processors, 63.46 GiB visible RAM and Balanced power mode. Fit-plus-polish also
skipped known models and changed slice allocation, so its contrast is not a pure
polishing-effect measurement. These are adaptively developed single-instance
results, not a general ranking or superiority claim.

`research-evidence.json` contains compact source-bound records. Original-union
rescoring, finite exhaustive tests, mutation controls and separate in-project AI
reviews support the scoped claims. In-project AI review is not outside peer
review. Full large-instance search agendas were not independently replayed.
