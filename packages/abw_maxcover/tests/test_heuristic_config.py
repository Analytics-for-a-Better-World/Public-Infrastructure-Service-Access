"""Tests for HeuristicConfig knobs on the heuristic portfolio."""

from __future__ import annotations

import numpy as np

import abw_maxcover as mc


def toy_instance() -> mc.MaxCoverInstance:
    weights = np.array([10, 7, 5, 4, 3], dtype=np.int64)
    ij = [[0, 1], [0, 2], [1, 2], [2, 3], [3]]
    ji = [[0, 1], [0, 2], [1, 2, 3], [3, 4]]
    return mc.build_instance(weights, ij, ji, name="toy", validate_consistency=True)


def _randomized_curve(repeats: int) -> mc.MaxCoverCurve:
    config = mc.HeuristicConfig(
        constructors=("randomized", "sample"),
        randomized_repeats=repeats,
        local_search="none",
        use_path_relinking=False,
    )
    return mc.approximate_pareto_curve(toy_instance(), [2], config=config, select_best=False)


def test_zero_randomized_repeats_runs_no_randomized_constructor() -> None:
    assert _randomized_curve(0).results == []
    # Negative values are treated as zero rather than raising or wrapping.
    assert _randomized_curve(-3).results == []


def test_randomized_repeats_controls_run_count_per_constructor() -> None:
    assert len(_randomized_curve(1).results) == 2
    assert len(_randomized_curve(3).results) == 6


def test_budget_callback_fires_once_per_execution_budget_in_ascending_order() -> None:
    calls: list[tuple[int, int]] = []
    config = mc.HeuristicConfig(
        constructors=("greedy", "compact", "randomized"),
        randomized_repeats=2,
        local_search="none",
        use_path_relinking=False,
    )
    curve = mc.run_heuristics(
        toy_instance(),
        [3, 1, 3, 2, -1],
        config=config,
        budget_callback=lambda budget, results: calls.append((budget, len(results))),
    )
    assert [budget for budget, _ in calls] == [1, 2, 3]
    # greedy, greedy_none, greedy_none_compact plus two randomized repeats.
    assert all(count == 5 for _, count in calls)
    assert curve.metadata["requested_budgets"] == [3, 1, 2]


def test_zero_repeats_with_deterministic_constructors_still_reports_them() -> None:
    config = mc.HeuristicConfig(
        constructors=("greedy", "randomized"),
        randomized_repeats=0,
        local_search="none",
        use_path_relinking=False,
    )
    curve = mc.approximate_pareto_curve(toy_instance(), [2], config=config, select_best=False)
    assert [r.method for r in curve.results] == ["greedy", "greedy_none"]
    best = mc.approximate_pareto_curve(toy_instance(), [2], config=config)
    assert best.budgets() == [2]
