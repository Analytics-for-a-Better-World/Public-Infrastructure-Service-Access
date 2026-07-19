# Shared Parvathy PhD helpers

This folder contains scripts shared by the Timor-Leste and Vietnam studies.

The shared helpers are mostly thin orchestration or reporting utilities around:

- `Research-Sandbox/general_distances_per_country` for distance matrices;
- `packages/abw_maxcover` for optimization;
- local output folders that are deliberately not committed to Git.

Use the country-specific README files for execution order. The shared scripts are copied here to make the exact 2026 report/deck workflow auditable without searching a Codex sandbox.

## Important scripts

```text
tools/collect_environment_metadata.py
tools/run_pipeline_fresh_root.py
tools/run_pipeline_fresh_root_profile.py
tools/run_fresh_network_pipeline_cases.py
tools/run_pipeline_with_partition_resume.py
tools/run_pipeline_fresh_root_with_partition_resume.py
tools/reassemble_parquet_parts.py
tools/compact_node_pair_cache.py
tools/make_candidate_facility_grid_maps.py
```

The pipeline helpers were used to launch repeatable country runs, manage fresh output roots, and reassemble chunked parquet matrices.
