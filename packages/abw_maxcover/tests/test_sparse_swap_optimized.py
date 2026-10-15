from __future__ import annotations

import numpy as np
import pytest
from _reference_sparse_swap import reference_improve

from abw_maxcover import build_instance_from_facility_map
from abw_maxcover._incremental_core import SparseSwapLocalSearch, compute_coverage_and_objective


def signature(x):
    return x.solution, x.objective, x.coverage.tolist(), x.objectives


def test_sparse_swap_matches_original_fixed_work():
    rng = np.random.default_rng(945012)
    for iteration in range(200):
        n, d = int(rng.integers(2, 35)), int(rng.integers(2, 45))
        rows = {j: np.flatnonzero(rng.random(d) < rng.uniform(0.02, 0.8)) for j in range(n)}
        weights = rng.integers(0, 1000, size=d, dtype=np.int64)
        instance = build_instance_from_facility_map(rows, weights, n_facilities=n)
        solution = rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False).tolist()
        engine = SparseSwapLocalSearch.from_instance(instance)
        kwargs = {"max_moves": [None, 0, 1][iteration % 3]}
        if iteration % 2:
            coverage, objective = compute_coverage_and_objective(instance, solution)
            original_coverage = coverage.copy()
            kwargs.update(coverage=coverage, objective=objective)
        assert signature(engine.improve(solution, **kwargs)) == signature(
            reference_improve(engine, solution, **kwargs)
        )
        if iteration % 2:
            np.testing.assert_array_equal(coverage, original_coverage)


@pytest.mark.parametrize(
    "rows,weights,solution",
    [
        ({0: [], 1: [], 2: []}, [0, 0], [0]),
        ({0: [0], 1: [], 2: [0, 1], 3: [], 4: []}, [3, 5], [0]),
        ({0: [0], 1: [1], 2: [0, 1]}, [0, 0], [0]),
        ({0: [0], 1: [1], 2: [1]}, [1, 5], [0]),
    ],
)
def test_sparse_swap_edges(rows, weights, solution):
    instance = build_instance_from_facility_map(
        rows, np.asarray(weights, dtype=np.int64), n_facilities=len(rows)
    )
    engine = SparseSwapLocalSearch.from_instance(instance)
    for kwargs in [{}, {"max_moves": 0}, {"max_moves": 1}, {"time_limit_seconds": 0}]:
        assert signature(engine.improve(solution, **kwargs)) == signature(
            reference_improve(engine, solution, **kwargs)
        )
    for kwargs in [{"max_moves": -1}, {"time_limit_seconds": -1}]:
        with pytest.raises(ValueError):
            engine.improve(solution, **kwargs)
