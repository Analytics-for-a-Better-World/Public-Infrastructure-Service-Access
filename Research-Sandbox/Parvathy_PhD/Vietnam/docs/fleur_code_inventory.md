# Fleur Notebook Inventory

Fleur's notebooks remain useful for algorithmic reconstruction, but their local data files are not used by the Vietnam replication scripts.

Recovered ideas:

- greedy construction;
- randomized restricted-candidate construction;
- sample greedy construction;
- random-plus-greedy construction;
- first-improvement local search;
- best-improvement local search;
- elite pool management;
- forward and backward path relinking.
- baseline coverage summaries;
- greedy marginal-gain plots;
- local-search improvement traces;
- repeated randomized experiment tables;
- solver-comparison columns for lower bound, upper bound, gap, status, and runtime.

Implementation choice in this folder:

- data comes from fresh PISA parquets;
- instances are converted to CSR `.npz` files;
- heuristics run against `approximated_tradeoff`-style `MaxCoverInstance` objects;
- outputs report baseline, incremental, and total covered population.
- `run_vietnam_fleur_style_analysis.py` produces the notebook-style analysis tables and plots from fresh instances.

Main read-only reference files inspected:

- `sort out\Master Thesis\Construct instances.ipynb`
- `sort out\Master Thesis\Heuristic model_nieuw.ipynb`
- `sort out\Fleur Last\Greedy and local search.ipynb`
- `sort out\Fleur Last\Best Improvement Local Search.ipynb`
- `sort out\Fleur Last\JG Experiments based on paper.ipynb`
- `sort out\Fleur Last\Dashbord\JG Experiments Max Covering.ipynb`
- `sort out\Fleur Last\Dashbord\maxcovering.py`
- `sort out\Master Thesis\optimization_model.py`
