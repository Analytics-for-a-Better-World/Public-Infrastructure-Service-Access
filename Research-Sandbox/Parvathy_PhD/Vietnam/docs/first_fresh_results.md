# First Fresh Vietnam Results

Date: 7 June 2026

## Fresh PISA Pipeline

Command variant completed:

- sources: `table candidates`
- destinations: `population`
- aggregate factor: `10`
- candidate spacing: `10,000 m`
- candidate snap cap: `5,000 m`
- retained maximum total distance: `150,000 m`
- matrix output: split sparse
- context map: skipped on the successful rerun because plotting all roads was the bottleneck

Key pipeline outputs:

- output folder: `C:\local\Parvathy\Vietnam\fresh_downloads\vietnam_data\outputs`
- log: `C:\local\Parvathy\Vietnam\logs\fresh_table_candidate_10kmgrid_150km_no_map.log`
- manifest: `run_manifest_*candidates_spacing_10000_maxsnap_5000_connectivity.yaml`
- candidate-to-population matrix: 61,729,130 distance rows, 789.3 MB parquet
- table-to-population matrix: 3,334,738 distance rows, 47.7 MB parquet
- population points: 408,838
- candidate sites after snapping/filtering: 3,131 in the pipeline; 3,126 have retained threshold coverage in the built instances

## Fresh CSR Instances

All instances use the current PISA outputs, not Fleur local `.npy` or pickle data.

| Threshold | Candidate count | Existing covered population | Incremental population available |
|---:|---:|---:|---:|
| 20 km | 3,126 | 54,288,625.792 | 45,102,520.369 |
| 50 km | 3,126 | 83,477,466.809 | 15,913,679.228 |
| 100 km | 3,126 | 93,922,648.534 | 5,468,497.544 |

## First Budget-20 Heuristic Checks

Greedy construction plus first-improvement local search:

| Threshold | Best incremental population | Total covered population |
|---:|---:|---:|
| 20 km | 8,989,532.124 | 63,278,157.916 |
| 50 km | 6,909,266.016 | 90,386,732.825 |
| 100 km | 3,510,260.626 | 97,432,909.160 |

Short randomized GRASP plus local search plus path relinking on the 50 km instance:

| Threshold | Budget | Iterations | Best incremental population | Total covered population |
|---:|---:|---:|---:|---:|
| 50 km | 20 | 5 | 6,898,143.545 | 90,375,610.354 |

This short GRASP run verifies that the GRASP/path-relinking implementation runs on the fresh CSR instance. It is not yet a tuned or exhaustive GRASP experiment.
