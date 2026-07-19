from __future__ import annotations

import itertools
import json
import subprocess
import sys

import numpy as np

import abw_maxcover as mc
from abw_maxcover.io import load_instance_npz, save_instance_npz


def random_instance(seed: int, n_demand: int = 18, n_facilities: int = 9) -> mc.MaxCoverInstance:
    rng = np.random.default_rng(seed)
    weights = rng.integers(0, 25, size=n_demand, dtype=np.int64)
    ij: list[list[int]] = [[] for _ in range(n_demand)]
    ji: list[list[int]] = [[] for _ in range(n_facilities)]
    for facility in range(n_facilities):
        demand = np.flatnonzero(rng.random(n_demand) < 0.3).astype(np.int32)
        ji[facility] = demand.tolist()
        for item in demand:
            ij[int(item)].append(facility)
    return mc.build_instance(weights, ij, ji, name=f"random_{seed}", validate_consistency=True)


def brute_force_objective(instance: mc.MaxCoverInstance, budget: int) -> int:
    best = 0
    for size in range(min(budget, instance.n_facilities) + 1):
        for solution in itertools.combinations(range(instance.n_facilities), size):
            _, objective = mc.compute_coverage_and_objective(instance, list(solution))
            best = max(best, objective)
    return best


def test_incremental_deltas_equal_full_recomputation() -> None:
    for seed in range(8):
        instance = random_instance(seed)
        selected = [0, 2, 4]
        coverage, objective = mc.compute_coverage_and_objective(instance, selected)
        for facility in range(instance.n_facilities):
            if facility not in selected:
                _, new_objective = mc.compute_coverage_and_objective(
                    instance, selected + [facility]
                )
                assert mc.add_delta(instance, coverage, facility) == new_objective - objective
        for facility in selected:
            reduced = [item for item in selected if item != facility]
            _, new_objective = mc.compute_coverage_and_objective(instance, reduced)
            assert mc.drop_delta(instance, coverage, facility) == new_objective - objective
            for replacement in range(instance.n_facilities):
                if replacement in selected:
                    continue
                swapped = reduced + [replacement]
                _, swap_objective = mc.compute_coverage_and_objective(instance, swapped)
                assert (
                    mc.swap_delta(instance, coverage, facility, replacement)
                    == swap_objective - objective
                )


def test_complete_greedy_envelope_satisfies_classical_bound() -> None:
    config = mc.HeuristicConfig(
        constructors=("greedy", "compact", "regreedy"),
        randomized_repeats=0,
        local_search="none",
        use_path_relinking=False,
    )
    for seed in range(10):
        instance = random_instance(seed)
        budgets = [1, 2, 3, 4]
        curve = mc.approximate_pareto_curve(instance, budgets, config=config)
        for result in curve.results:
            optimum = brute_force_objective(instance, result.budget)
            bound = (1.0 - (1.0 - 1.0 / result.budget) ** result.budget) * optimum
            assert result.objective is not None
            assert result.objective + 1e-9 >= bound


def test_zero_loss_compaction_preserves_saturated_objective() -> None:
    for seed in range(10):
        instance = random_instance(seed)
        greedy = mc.greedy_construct(instance)
        compact_curve = mc.approximate_pareto_curve(
            instance,
            [instance.n_facilities],
            config=mc.HeuristicConfig(
                constructors=("compact",),
                randomized_repeats=0,
                local_search="none",
                use_path_relinking=False,
            ),
        )
        assert compact_curve.results[0].objective == greedy.objective


def test_npz_round_trip_preserves_instance(tmp_path) -> None:
    instance = random_instance(42)
    path = tmp_path / "instance.npz"
    save_instance_npz(instance, path)
    restored = load_instance_npz(path)
    assert restored.name == instance.name
    assert np.array_equal(restored.weights, instance.weights)
    assert np.array_equal(restored.ij_indptr, instance.ij_indptr)
    assert np.array_equal(restored.ij_indices, instance.ij_indices)
    assert np.array_equal(restored.ji_indptr, instance.ji_indptr)
    assert np.array_equal(restored.ji_indices, instance.ji_indices)


def test_core_import_does_not_load_optional_solver_or_reporting_packages() -> None:
    code = (
        "import json,sys,abw_maxcover; "
        "print(json.dumps([name for name in "
        "['gurobipy','pyomo','pandas','scipy'] if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []
