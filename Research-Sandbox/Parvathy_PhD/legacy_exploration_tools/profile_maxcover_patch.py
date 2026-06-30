from __future__ import annotations

import importlib.util
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np


REPO_MAXCOVER = Path(r"C:\github\Public-Infrastructure-Service-Access\optimization\maxcover.py")
PATCH_MAXCOVER = Path(r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis\work_patch\optimization\maxcover.py")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_random_rows(seed: int = 123):
    rng = np.random.default_rng(seed)
    n_demand = 8_000
    n_facilities = 2_500
    weights = rng.integers(1, 20, size=n_demand, dtype=np.int64)
    facility_rows = []
    for _ in range(n_facilities):
        size = int(rng.integers(15, 80))
        facility_rows.append(np.unique(rng.integers(0, n_demand, size=size, dtype=np.int32)))

    demand_rows = [[] for _ in range(n_demand)]
    for facility, row in enumerate(facility_rows):
        for demand in row:
            demand_rows[int(demand)].append(facility)
    demand_rows = [np.asarray(row, dtype=np.int32) for row in demand_rows]
    return weights, demand_rows, facility_rows


def bench(label: str, fn, repeats: int = 3):
    times = []
    peaks = []
    objective = None
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peaks.append(peak)
        objective = result.objective
    print(
        f"{label}: obj={objective} "
        f"median_s={float(np.median(times)):.4f} "
        f"min_s={min(times):.4f} "
        f"peak_mb={max(peaks) / 1e6:.2f}"
    )


def main() -> None:
    old = load_module(REPO_MAXCOVER, "old_maxcover")
    new = load_module(PATCH_MAXCOVER, "new_maxcover")
    weights, demand_rows, facility_rows = build_random_rows()

    inst_old = old.build_instance(weights, demand_rows, facility_rows, validate_consistency=False)
    inst_new = new.build_instance(weights, demand_rows, facility_rows, validate_consistency=False)
    base = new.budgeted_construct(inst_new, 120, constructor="greedy")
    seed = base.solution[:70]

    bench(
        "old select_by_marginal_gain refill",
        lambda: old.select_by_marginal_gain(inst_old, 120, initial_solution=seed),
    )
    bench(
        "new select_by_marginal_gain refill",
        lambda: new.select_by_marginal_gain(inst_new, 120, initial_solution=seed),
    )
    bench(
        "old deployment sequence",
        lambda: old.greedy_deployment_sequence(inst_old, base.solution, budgets=[20, 60, 120]).results[-1],
    )
    bench(
        "new deployment sequence",
        lambda: new.greedy_deployment_sequence(inst_new, base.solution, budgets=[20, 60, 120]).results[-1],
    )

    cfg_old = old.HeuristicConfig(
        constructors=("greedy", "randomized"),
        randomized_repeats=1,
        seed=7,
        use_path_relinking=False,
    )
    cfg_new = new.HeuristicConfig(
        constructors=("greedy", "compact", "regreedy", "randomized"),
        randomized_repeats=1,
        seed=7,
        use_path_relinking=False,
    )
    bench(
        "old approximate all",
        lambda: old.approximate_pareto_curve(inst_old, [40, 80], config=cfg_old, select_best=False).results[-1],
        repeats=2,
    )
    bench(
        "new approximate all",
        lambda: new.approximate_pareto_curve(inst_new, [40, 80], config=cfg_new, select_best=False).results[-1],
        repeats=2,
    )

    if "line_profiler" in sys.modules:
        return
    try:
        from line_profiler import LineProfiler
    except Exception as exc:
        print(f"line_profiler unavailable: {exc}")
        return

    profiler = LineProfiler()
    profiler.add_function(new.budgeted_construct)
    profiler.add_function(new.select_by_marginal_gain)
    profiler.add_function(new.SparseSwapLocalSearch.improve)

    def profile_case():
        cfg = new.HeuristicConfig(
            constructors=("greedy", "compact", "regreedy", "randomized"),
            randomized_repeats=1,
            seed=11,
            use_path_relinking=False,
        )
        return new.approximate_pareto_curve(inst_new, [60], config=cfg, select_best=False)

    profiler(profile_case)()
    profiler.print_stats()


if __name__ == "__main__":
    main()
