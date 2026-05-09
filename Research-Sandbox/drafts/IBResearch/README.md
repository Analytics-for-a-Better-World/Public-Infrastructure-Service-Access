# IB Timetabling Experiments

This folder contains the Python source used for the IB timetabling heuristic,
MILP, large-neighborhood search, validation, and clean-run summaries described
in the accompanying manuscript.

The data are not committed to GitHub. Put the private input files in a local
`data/` folder before running the scripts:

- `M24 exam names and block lengths.csv`
- `exam_days3.csv`
- `Exam Pairs ABW-2.csv`

Toy-instance scripts also expect the toy CSV files used in Antony Furlong's
thesis replication workflow, if those experiments are run.

## Environment

The clean reruns were made with:

- Python 3.12.10
- gurobipy 13.0.2
- Gurobi Optimizer 13.0.2
- Windows 11 Pro on a Dell Precision 3560, Intel Core i7-1185G7, 64 GB RAM

Install the Python dependencies in your preferred environment:

```powershell
py -m pip install pandas matplotlib gurobipy
```

You also need a working Gurobi license.

## Main Files

- `src/anthony_model.py`: data preparation, Antony MILP builder, objective
  evaluation, MIP starts, timetable extraction, and proof-strengthening switches.
- `src/full_heuristic.py`: deterministic constructive heuristic and full
  timetable validation.
- `src/lns_improvement.py`: MILP-based large-neighborhood search, including
  date-window and conflict-neighbor neighborhoods, load-aware selection, and
  guarded acceptance.
- `src/spread_improvement.py`: secondary spread diagnostics and exploratory
  spread-improvement model.
- `src/ib_graphics.py`: heatmaps and agenda visualizations.
- `solution_value_standalone.py`: standalone objective function for use in
  external notebooks.
- `run_full_heuristic.py`: construct a full-instance feasible timetable.
- `run_lns_improvement.py`: run full-instance LNS improvements.
- `run_full_mip_experiment.py`: run seeded full-instance MILP proof experiments.
- `run_toy_mip_experiment.py`: run toy heuristic/MILP and strengthening variants.
- `run_toy_antony_verbatim.py`: run the Appendix-style toy MILP replication.
- `summarize_clean_runs.py`: summarize clean logs, histories, and solution
  diagnostics, and regenerate the comparison plots.

## Reproducing the Clean Pipeline

Run commands from this folder. The `-B` flag avoids writing Python bytecode
caches.

```powershell
py -B run_full_heuristic.py `
  --max-rounds 2 `
  --output clean_runs_20260508\full_heuristic_rounds2.csv
```

```powershell
py -B run_lns_improvement.py `
  --start clean_runs_20260508\full_heuristic_rounds2.csv `
  --output clean_runs_20260508\full_lns_23day_8x90.csv `
  --history-output clean_runs_20260508\full_lns_23day_8x90_history.csv `
  --nb-days 23 --iterations 8 --subjects 10 --time-limit 90 `
  --neighborhood-sizes 6,10,14,6,10,14,12,10 `
  --strategy-cycle worst_subjects,worst_pairs,same_slot_clashes,crowded_days,date_window,conflict_neighbors,same_slot_clashes,worst_subjects `
  --fix-mode-cycle exact,date_window_slots_free,selected_days_slots_free `
  --no-adaptive-time
```

```powershell
py -B run_lns_improvement.py `
  --start clean_runs_20260508\full_lns_nb34_recommended_6x90.csv `
  --output clean_runs_20260508\full_lns_nb34_guarded_6x120.csv `
  --history-output clean_runs_20260508\full_lns_nb34_guarded_6x120_history.csv `
  --nb-days 34 --iterations 6 --subjects 8 --time-limit 120 `
  --neighborhood-sizes 12,10,12,8,14,12 `
  --strategy-cycle load_aware_date_window,date_window,conflict_neighbors,date_window,load_aware_date_window,date_window `
  --fix-mode-cycle date_window_slots_free,selected_days_slots_free,date_window_slots_free `
  --solution-pool-size 10 --no-adaptive-time `
  --load-acceptance-tolerance 50000 `
  --max-spread-regression 0 `
  --target-max-day-exams 5 --target-max-slot-exams 3 `
  --enforce-subject-exam-order --symmetry 2
```

```powershell
py -B run_full_mip_experiment.py `
  --start clean_runs_20260508\full_lns_nb34_guarded_6x120.csv `
  --output clean_runs_20260508\full_mip_guarded_order_sym2_10min_timetable.csv `
  --progress-output clean_runs_20260508\full_mip_guarded_order_sym2_10min_progress.csv `
  --log-output clean_runs_20260508\full_mip_guarded_order_sym2_10min.log `
  --plot-output clean_runs_20260508\full_mip_guarded_order_sym2_10min_bounds.png `
  --nb-days 34 --time-limit 600 --objective-mode formal `
  --enforce-subject-exam-order --symmetry 2
```

Two experimental strengthening mechanisms can also be activated in
`run_full_mip_experiment.py`. The callback heuristic inspects incumbent
solutions during the MILP solve, tries deterministic same-day morning/afternoon
slot swaps, validates each candidate, and injects an improving schedule with
Gurobi's `cbSetSolution`. Dense-cluster cuts select high-overlap exam clusters,
solve each cluster as a small sub-MIP, and add a valid lower-bound inequality
for that cluster's contribution to the full objective.

```powershell
py -B run_full_mip_experiment.py `
  --start clean_runs_20260508\full_lns_nb34_guarded_6x120.csv `
  --output clean_runs_20260508\full_mip_callback_dense_5min_timetable.csv `
  --progress-output clean_runs_20260508\full_mip_callback_dense_5min_progress.csv `
  --log-output clean_runs_20260508\full_mip_callback_dense_5min.log `
  --plot-output clean_runs_20260508\full_mip_callback_dense_5min_bounds.png `
  --nb-days 34 --time-limit 300 --objective-mode formal `
  --enforce-subject-exam-order --symmetry 2 --mip-focus 3 --cuts 2 `
  --callback-slot-swap-heuristic `
  --dense-cluster-cuts 1 --dense-cluster-size 8 --dense-cluster-time-limit 10
```

Overlapping dense-cluster cuts can be tested by adding
`--dense-cluster-overlap` and a novelty threshold. The cluster selector keeps a
candidate overlapping cluster only when enough of its positive pair mass has not
already appeared in earlier clusters.

```powershell
py -B run_full_mip_experiment.py `
  --start clean_runs_20260508\full_lns_nb34_guarded_6x120.csv `
  --output clean_runs_20260508\full_mip_dense30_overlap_15min_timetable.csv `
  --progress-output clean_runs_20260508\full_mip_dense30_overlap_15min_progress.csv `
  --log-output clean_runs_20260508\full_mip_dense30_overlap_15min.log `
  --plot-output clean_runs_20260508\full_mip_dense30_overlap_15min_bounds.png `
  --dense-cluster-summary clean_runs_20260508\full_mip_dense30_overlap_15min_clusters.csv `
  --nb-days 34 --time-limit 900 --objective-mode formal `
  --enforce-subject-exam-order --symmetry 2 --mip-focus 3 --cuts 2 `
  --dense-cluster-cuts 30 --dense-cluster-size 8 `
  --dense-cluster-overlap --dense-cluster-min-new-pair-share 0.05 `
  --dense-cluster-time-limit 5
```

```powershell
py -B run_toy_antony_verbatim.py `
  --output clean_runs_20260508\toy_antony_verbatim_mip.csv `
  --progress-output clean_runs_20260508\toy_antony_verbatim_progress.csv `
  --log-output clean_runs_20260508\toy_antony_verbatim.log `
  --plot-output clean_runs_20260508\toy_antony_verbatim_bounds.png
```

After logs and histories exist:

```powershell
py -B summarize_clean_runs.py
```

## Clean Rerun Highlights

- Appendix-style toy model: objective 25,190 proved in 1,443.14 seconds.
- Reusable toy baseline: objective 25,190 proved in 1,245.53 seconds.
- Best clean full-instance guarded LNS timetable: objective 15,855,200.
- Best clean full-MILP lower-bound plateau in short proof runs: about 10,740,292.
- Callback and dense-cluster experiments did not improve the best 5-minute
  incumbent or final bound, but the dense-cluster cut lifted the raw root
  relaxation from about 0.94 million to 4.62 million before Gurobi's own cuts.
- Overlapping dense-cluster cuts lifted the full-instance raw root relaxation
  to about 4.69 million, but did not materially change the final bound plateau.
  On the toy instance, dense cuts slowed the best paper-order proof recipe, so
  the six-hour full-instance follow-up was not launched.

The toy optimum differs from Antony Furlong's reported thesis value of 25,910.
With the currently available CSV files and Gurobi 13.0.2, both the verbatim
Appendix-style script and the reusable model solve the toy instance to 25,190.
