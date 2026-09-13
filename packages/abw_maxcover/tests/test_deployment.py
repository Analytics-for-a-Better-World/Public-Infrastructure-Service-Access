"""Regression tests for greedy deployment sequencing."""

from __future__ import annotations

import numpy as np

import abw_maxcover as mc


def toy_instance() -> mc.MaxCoverInstance:
    weights = np.array([10, 7, 5, 4, 3], dtype=np.int64)
    ij = [[0, 1], [0, 2], [1, 2], [2, 3], [3]]
    ji = [[0, 1], [0, 2], [1, 2, 3], [3, 4]]
    return mc.build_instance(weights, ij, ji, name="toy", validate_consistency=True)


def test_deployment_keeps_zero_gain_pool_facilities() -> None:
    # Facilities 0, 1, and 3 already cover every demand point, so facility 2
    # adds nothing. It must still appear in the roll-out, last and with a
    # zero marginal gain, so the final step equals the source pool.
    instance = toy_instance()
    pool = [0, 1, 3, 2]

    curve = mc.greedy_deployment_sequence(instance, pool, budgets=[1, 2, 3, 4])

    assert curve.budgets() == [1, 2, 3, 4]
    final = curve.results[-1]
    assert sorted(final.solution) == sorted(pool)
    assert final.selected_count == len(pool)
    assert final.objective == instance.total_weight

    gains = [result.metadata["marginal_gain"] for result in curve.results]
    assert gains[-1] == 0
    assert sum(gains) == final.objective

    previous = 0
    for result in curve.results:
        _, recomputed = mc.compute_coverage_and_objective(instance, result.solution)
        assert result.objective == recomputed
        assert result.selected_count == result.budget
        assert result.metadata["marginal_gain"] == result.objective - previous
        previous = result.objective


def test_deployment_marginal_gains_sum_to_objective_on_random_pools() -> None:
    rng = np.random.default_rng(5)
    for seed in range(6):
        gen = np.random.default_rng(seed)
        n_demand, n_facilities = 25, 10
        weights = gen.integers(1, 20, size=n_demand, dtype=np.int64)
        ij: list[list[int]] = [[] for _ in range(n_demand)]
        ji: list[list[int]] = [[] for _ in range(n_facilities)]
        for facility in range(n_facilities):
            demand = np.flatnonzero(gen.random(n_demand) < 0.4)
            ji[facility] = demand.tolist()
            for item in demand:
                ij[int(item)].append(facility)
        instance = mc.build_instance(weights, ij, ji, validate_consistency=True)
        pool = [int(f) for f in rng.permutation(n_facilities)[: rng.integers(3, n_facilities + 1)]]

        curve = mc.greedy_deployment_sequence(instance, pool)

        assert curve.budgets() == list(range(1, len(pool) + 1))
        final = curve.results[-1]
        assert sorted(final.solution) == sorted(pool)
        _, pool_objective = mc.compute_coverage_and_objective(instance, pool)
        assert final.objective == pool_objective
        assert sum(r.metadata["marginal_gain"] for r in curve.results) == pool_objective
        objectives = [r.objective for r in curve.results]
        assert objectives == sorted(objectives)


def test_deployment_budget_zero_and_clipping() -> None:
    instance = toy_instance()
    curve = mc.greedy_deployment_sequence(instance, [0, 2, 3], budgets=[0, 5, 3])
    assert curve.budgets() == [0, 3]
    assert curve.results[0].objective == 0
    assert curve.results[0].solution == []
    assert curve.results[0].metadata["marginal_gain"] == 0
    assert curve.results[1].objective == 29
