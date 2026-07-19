# Mathematical Programming Computation artifact

This repository directory is the canonical software artifact for the manuscript
on engineering fast heuristics for large-scale maximum-covering location
problems. It separates three evidence layers:

1. `tests/` checks implementation invariants, optional-dependency isolation,
   sparse serialization, and approximation quality against brute-force optima.
2. `benchmarks/` records portable synthetic scaling and exact quality audits.
3. Application campaigns under `Research-Sandbox/Parvathy_PhD` construct and
   analyze the Timor-Leste and Vietnam instances through the same public API.

Every reported campaign should record the Git commit, package version, Python and
NumPy versions, solver version, solver parameters, random seed, hardware, input
checksums, and separate construction and solution times. See
[`docs/reproducibility.md`](docs/reproducibility.md) for the required manifest.

For archival submission, create a repository release from the exact manuscript
commit and deposit that release together with immutable benchmark inputs and raw
result manifests in a DOI-backed archive. The paper should cite both the Git
commit and the archive DOI; a moving branch name is not sufficient provenance.
