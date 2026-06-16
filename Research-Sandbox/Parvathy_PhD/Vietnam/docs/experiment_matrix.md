# Vietnam Fresh-Data Experiment Matrix

All executable experiments in this folder use PISA-generated parquets or CSR `.npz` files built from those parquets. Historical Fleur `.npy` and pickle files are deliberately out of scope for replication runs.

## Pipeline Runs

Start with a current 10 km candidate grid:

- sources: `table candidates`
- destinations: `population`
- source table: `C:\Users\joaqu\OneDrive - UvA\share\Vietnam\stroke-facs-100-en.xlsx`
- grid spacing: 10,000 m
- snap cap: 5,000 m
- retained matrix cap: 150,000 m
- matrix output: split sparse

If the 10 km run behaves well, add a 5 km candidate-grid run with a 2,500 m snap cap.

## Threshold Instances

Build separate max-cover instances from the same retained matrix:

- 20 km threshold
- 50 km threshold
- 100 km threshold

The builder treats `source_type = table` as existing stroke coverage and `source_type = candidates` as potential new facilities. Existing-covered demand is assigned zero incremental weight in the optimization instance.

## Budgets

Use:

```text
20, 40, 60, 80, 100, 200
```

For 100 km threshold, prioritize:

```text
20, 40, 60, 80
```

## Heuristic Runs

Minimum checks:

- greedy construction only;
- greedy plus first-improvement local search;
- randomized GRASP plus first-improvement local search;
- randomized GRASP plus first-improvement local search and path relinking.

Recommended GRASP settings:

- `--constructor randomized`
- `--rcl-size 25`
- `--time-limit-seconds 300`
- `--max-pool 8`
- seeds `42, 43, 44`

## Validation Outputs

For each run keep:

- instance metadata JSON;
- trace CSV;
- best-solution JSON;
- selected candidate source IDs;
- baseline covered population;
- incremental covered population;
- total covered population after adding candidates.

## Fleur-Style Analysis Outputs

In addition to single-run heuristic JSON/CSV files, run:

```powershell
py scripts\run_vietnam_fleur_style_analysis.py `
  --instances C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_20km_threshold.npz `
              C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_50km_threshold.npz `
              C:\local\Parvathy\Vietnam\optimization\vietnam_10kmgrid_100km_threshold.npz `
  --budgets 20 40 60 80 100 200 `
  --local-search-budgets 20 40 60 80 `
  --randomized-budgets 20 `
  --randomized-repeats 3 `
  --grasp-max-iterations 5 `
  --output-dir C:\local\Parvathy\Vietnam\fleur_style_analysis
```

This creates:

- `instance_statistics.csv`;
- `coverage_summary_by_budget.csv`;
- greedy marginal traces by threshold;
- local-search traces by threshold and budget;
- randomized GRASP repeat traces;
- selected fresh candidate IDs;
- coverage and marginal-gain plots.

The next exact-solver replication step is to add optional Gurobi or Pyomo runs for small/restricted instances and append lower bound, upper bound, gap, status, and solve time columns.
