"""Regression tests for reported statistics on heuristic results."""

from __future__ import annotations

import numpy as np

import abw_maxcover as mc


def instance_with_redundant_greedy_pick() -> mc.MaxCoverInstance:
    """Greedy picks facility 0 first, then 1 and 2, which together cover 0's demand.

    Facility 0 covers demand {0, 1} (weight 6), facility 1 covers {0, 2} and
    facility 2 covers {1, 3} (weight 5 each). After all three are open,
    facility 0 is redundant, so zero-loss compaction drops it.
    """
    weights = np.array([3, 3, 2, 2], dtype=np.int64)
    ij = [[0, 1], [0, 2], [1], [2]]
    ji = [[0, 1], [0, 2], [1, 3]]
    return mc.build_instance(weights, ij, ji, name="redundant_pick", validate_consistency=True)


def random_instance(seed: int, n_demand: int = 60, n_facilities: int = 16) -> mc.MaxCoverInstance:
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 50, size=n_demand, dtype=np.int64)
    ij: list[list[int]] = [[] for _ in range(n_demand)]
    ji: list[list[int]] = [[] for _ in range(n_facilities)]
    for facility in range(n_facilities):
        demand = np.flatnonzero(rng.random(n_demand) < 0.2)
        ji[facility] = demand.tolist()
        for item in demand:
            ij[int(item)].append(facility)
    return mc.build_instance(weights, ij, ji, name=f"random_{seed}", validate_consistency=True)


def _by_method(curve: mc.MaxCoverCurve, budget: int) -> dict[str, mc.MaxCoverResult]:
    return {r.method: r for r in curve.results if r.budget == budget}


def test_local_search_moves_is_zero_when_no_local_search_ran() -> None:
    instance = instance_with_redundant_greedy_pick()
    config = mc.HeuristicConfig(
        constructors=("greedy", "compact", "regreedy"),
        randomized_repeats=0,
        local_search="none",
        use_path_relinking=False,
    )
    curve = mc.approximate_pareto_curve(instance, [3], config=config, select_best=False)
    records = _by_method(curve, 3)

    assert records["greedy"].selected_count == 3
    # Compaction dropped the redundant facility: that drop is not a move.
    assert records["greedy_none_compact"].selected_count == 2
    assert records["greedy_none_compact"].metadata["compacted_selected_count"] == 2
    for method, record in records.items():
        assert record.local_search_moves == 0, method


def test_local_search_moves_counts_swaps_consistently_across_methods() -> None:
    config = mc.HeuristicConfig(
        constructors=("greedy", "compact", "regreedy"),
        randomized_repeats=0,
        local_search="first_sparse",
        use_path_relinking=False,
    )
    swapped_somewhere = False
    for seed in range(8):
        instance = random_instance(seed)
        budget = 4
        curve = mc.approximate_pareto_curve(instance, [budget], config=config, select_best=False)
        records = _by_method(curve, budget)

        greedy = records["greedy"]
        improved = records["greedy_first_sparse"]
        compact = records["greedy_first_sparse_compact"]
        regreedy = records["greedy_first_sparse_compact_regreedy"]

        # The plain greedy prefix never ran local search, whatever its size.
        assert greedy.local_search_moves == 0
        # Compaction reports the swaps of the local search it compacted.
        assert compact.local_search_moves == improved.local_search_moves
        # Regreedy adds its own local-search phase on top of the first one.
        assert regreedy.local_search_moves >= improved.local_search_moves
        # Every accepted swap strictly improves the objective, so the swap
        # count is bounded by the objective gain over the construction.
        if improved.local_search_moves:
            assert improved.objective > improved.construction_objective
            swapped_somewhere = True
    assert swapped_somewhere, "expected local search to accept at least one swap"


def test_randomized_records_report_local_search_moves_before_relinking() -> None:
    instance = random_instance(3)
    config = mc.HeuristicConfig(
        constructors=("randomized",),
        randomized_repeats=3,
        local_search="first_sparse",
        use_path_relinking=True,
        seed=1,
    )
    curve = mc.approximate_pareto_curve(instance, [4], config=config, select_best=False)
    for record in curve.results:
        assert record.local_search_moves is not None
        assert record.local_search_moves >= 0
        assert record.construction_objective is not None
        if record.local_search_moves:
            assert record.objective > record.construction_objective
