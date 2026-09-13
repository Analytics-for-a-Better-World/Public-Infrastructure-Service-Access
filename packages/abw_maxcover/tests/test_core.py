"""Tests for the shared incremental core helpers."""

from __future__ import annotations

import numpy as np

import abw_maxcover as mc
from abw_maxcover._incremental_core import budgeted_construct, prefix_result


def toy_instance() -> mc.MaxCoverInstance:
    weights = np.array([10, 7, 5, 4, 3], dtype=np.int64)
    ij = [[0, 1], [0, 2], [1, 2], [2, 3], [3]]
    ji = [[0, 1], [0, 2], [1, 2, 3], [3, 4]]
    return mc.build_instance(weights, ij, ji, name="toy", validate_consistency=True)


def test_prefix_result_trace_ends_at_the_prefix_objective() -> None:
    instance = toy_instance()
    full = mc.greedy_construct(instance)
    for budget in range(len(full.solution) + 1):
        prefix = prefix_result(instance, full, budget)
        assert len(prefix.solution) == budget
        assert len(prefix.objectives) == budget + 1
        assert prefix.objectives[-1] == prefix.objective
        assert prefix.total_time == prefix.times[-1]


def test_prefix_result_accounts_for_an_initial_solution() -> None:
    instance = toy_instance()
    constructed = budgeted_construct(instance, 3, initial_solution=[3])
    assert constructed.solution[0] == 3
    # One initial facility plus two greedy steps: three entries in the trace.
    assert len(constructed.objectives) == 3

    for budget in (1, 2, 3):
        prefix = prefix_result(instance, constructed, budget)
        assert prefix.solution == constructed.solution[:budget]
        assert prefix.objectives[-1] == prefix.objective
        assert len(prefix.objectives) == budget
        assert prefix.total_time == prefix.times[-1]

    # A prefix shorter than the initial solution has no trace of its own and
    # falls back to the recorded starting state.
    empty = prefix_result(instance, constructed, 0)
    assert empty.solution == []
    assert empty.objective == 0
    assert empty.objectives == constructed.objectives[:1]
