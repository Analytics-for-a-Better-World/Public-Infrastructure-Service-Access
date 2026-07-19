# Reproducibility protocol

Computational papers should preserve more than the final objective values.

## Required provenance

Record for every campaign:

- `abw-maxcover` version and full Git commit;
- Python, NumPy, solver, and operating-system versions;
- physical or virtual hardware description;
- instance source, license, hash, dimensions, and realized incidence count;
- random seed and number of repeats;
- requested budgets and internally executed budget order;
- heuristic configuration, including all local-search and relinking limits;
- exact solver time limit, status, incumbent, upper bound, and MIP gap;
- instance-build, construction, refinement, model-build, and solve times.

Do not label a time-limited incumbent as an optimum. When the solver stops
without proof, compare a heuristic with the solver upper bound for a valid
certificate and report the solver incumbent only as another feasible solution.

## Timing boundaries

Report separately:

1. upstream data and coverage construction;
2. conversion to `MaxCoverInstance`;
3. complete-frontier generation;
4. selected-budget calibration or refinement;
5. exact model construction and optimization.

The package benchmark scripts write JSON manifests and CSV results. Long-running
country scripts should also use `result_callback` to durably checkpoint every
completed budget and its selected facilities.

## MPC artifact layout

A review artifact should contain:

```text
artifact/
  README.md
  environment.json
  instances/          # or immutable download instructions and hashes
  configs/
  results/
  figures/
  manuscript/
```

Generated figures and tables should read durable result files rather than
copying values from console output. Redistribution restrictions for literature
instances must be documented explicitly.
