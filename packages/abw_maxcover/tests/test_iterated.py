import numpy as np
import pytest

from abw_maxcover import IteratedSearchConfig, build_instance_from_facility_map, run_iterated_search
from abw_maxcover._incremental_core import compute_coverage_and_objective
from abw_maxcover.iterated import _elite_insert
from abw_maxcover.results import HeuristicResult


@pytest.mark.parametrize("perturb,adaptive", [(False, False), (True, False), (True, True)])
def test_iterated_fixed_work_seed_and_feasibility(perturb, adaptive):
    rng = np.random.default_rng(391)
    for trial in range(20):
        rows = {j: np.flatnonzero(rng.random(30) < 0.12) for j in range(20)}
        instance = build_instance_from_facility_map(
            rows, rng.integers(0, 100, size=30, dtype=np.int64), n_facilities=20
        )
        cfg = IteratedSearchConfig(
            time_limit_seconds=3600,
            max_iterations=8,
            seed=trial,
            perturb=perturb,
            adaptive=adaptive,
            max_pool=3,
        )
        trace = []
        a = run_iterated_search(instance, 3, config=cfg, callback=trace.append)
        b = run_iterated_search(instance, 3, config=cfg)
        assert (a.solution, a.objective) == (b.solution, b.objective)
        cov, obj = compute_coverage_and_objective(instance, a.solution)
        assert obj == a.objective
        np.testing.assert_array_equal(cov, a.coverage)
        assert len(set(a.solution)) == len(a.solution) <= 3
        assert all(t["pool_size"] <= 3 for t in trace)
        assert all(
            x["best_objective"] <= y["best_objective"]
            for x, y in zip(trace, trace[1:], strict=False)
        )


def test_elite_capacity_duplicates_and_best_preserved():
    def item(sites, score):
        return HeuristicResult(sites, score, np.zeros(4, dtype=np.int32), [score], [0.0], 0.0)

    best = item([0, 1], 10)
    pool = [best]
    _elite_insert(pool, item([1, 0], 10), 2)
    assert len(pool) == 1
    for sites, score in [([0, 2], 8), ([2, 3], 9), ([1, 2], 7)]:
        _elite_insert(pool, item(sites, score), 2)
        assert len(pool) <= 2
        assert any(p is best for p in pool)


def test_iterated_zero_budget_and_invalid_configuration():
    instance = build_instance_from_facility_map(
        {0: [], 1: []}, np.array([0, 0], dtype=np.int64), n_facilities=2
    )
    assert run_iterated_search(instance, 0).objective == 0
    positive = build_instance_from_facility_map(
        {0: [0], 1: [1]}, np.array([3, 5], dtype=np.int64), n_facilities=2
    )
    assert run_iterated_search(positive, 0).metadata["iterations"] == 0
    for cfg in [
        IteratedSearchConfig(max_pool=0),
        IteratedSearchConfig(time_limit_seconds=-1),
        IteratedSearchConfig(max_iterations=-1),
        IteratedSearchConfig(acceptance_window=0),
        IteratedSearchConfig(perturbation_fraction=0),
    ]:
        with pytest.raises(ValueError):
            run_iterated_search(instance, 1, config=cfg)
    for value in [float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            run_iterated_search(instance, 1, config=IteratedSearchConfig(time_limit_seconds=value))


def test_adaptation_and_late_acceptance_controlled_trace(monkeypatch):
    import abw_maxcover.iterated as module

    instance = build_instance_from_facility_map(
        {j: [j] for j in range(20)}, np.ones(20, dtype=np.int64), n_facilities=20
    )

    def item(sites, score):
        return HeuristicResult(sites, score, np.zeros(20, dtype=np.int32), [score], [0.0], 0.0)

    # Deliberately controlled local-optimum outputs isolate controller decisions.
    sequence = iter(
        [
            item(list(range(10)), 10),
            item(list(range(10)), 12),
            item(list(range(10, 20)), 11),
            item(list(range(10, 20)), 5),
            item(list(range(5, 15)), 13),
        ]
    )

    class Engine:
        def improve(self, *args, **kwargs):
            return next(sequence)

    monkeypatch.setattr(module.SparseSwapLocalSearch, "from_instance", lambda instance: Engine())
    monkeypatch.setattr(
        module, "budgeted_construct", lambda *args, **kwargs: item(list(range(10)), 0)
    )
    trace = []
    cfg = IteratedSearchConfig(
        time_limit_seconds=3600,
        max_iterations=4,
        path_relinking=False,
        restart_interval=100,
        acceptance_window=2,
    )
    result = run_iterated_search(instance, 10, config=cfg, callback=trace.append)
    assert [t["accepted"] for t in trace] == [True, True, False, True]
    assert trace[0]["strength"] == 2
    assert trace[1]["strength"] == 1
    assert [t["best_objective"] for t in trace] == [12, 12, 12, 13]
    assert result.objective == 13
    assert result.metadata["config"]["acceptance_window"] == 2
