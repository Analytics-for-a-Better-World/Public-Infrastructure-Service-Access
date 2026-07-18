import numpy as np

import abw_maxcover as mc


def toy_instance() -> mc.MaxCoverInstance:
    weights = np.array([10, 7, 5, 4, 3], dtype=np.int64)
    ij = [[0, 1], [0, 2], [1, 2], [2, 3], [3]]
    ji = [[0, 1], [0, 2], [1, 2, 3], [3, 4]]
    return mc.build_instance(
        weights,
        ij,
        ji,
        name="toy",
        validate_consistency=True,
    )


def test_import_and_build_instance() -> None:
    instance = toy_instance()
    assert instance.n_demand == 5
    assert instance.n_facilities == 4
    assert instance.total_weight == 29
    assert instance.facilities_of(0).tolist() == [0, 1]
    assert instance.demand_of(2).tolist() == [1, 2, 3]


def test_approximate_curve_preserves_requested_budget_order() -> None:
    instance = toy_instance()
    checkpointed_budgets: list[int] = []
    config = mc.HeuristicConfig(
        constructors=("greedy", "compact", "regreedy", "randomized"),
        randomized_repeats=1,
        local_search="first",
        seed=11,
    )
    curve = mc.approximate_pareto_curve(
        instance,
        [3, 1, 2],
        config=config,
        result_callback=lambda result: checkpointed_budgets.append(result.budget),
    )
    assert curve.budgets() == [3, 1, 2]
    assert checkpointed_budgets == [1, 2, 3]
    assert [result.selected_count <= result.budget for result in curve.results]
    assert all(result.objective is not None for result in curve.results)


def test_shared_incremental_refill_matches_greedy_prefix() -> None:
    instance = toy_instance()
    full_greedy = mc.greedy_construct(instance)
    budgeted_greedy = mc.select_by_marginal_gain(instance, 3)
    assert budgeted_greedy.objective == full_greedy.objectives[3]
    assert budgeted_greedy.solution == full_greedy.solution[:3]


def test_delta_helpers_and_deployment_sequence() -> None:
    instance = toy_instance()
    coverage, objective = mc.compute_coverage_and_objective(instance, [0, 2])
    assert objective == 26
    assert mc.add_delta(instance, coverage, 3) == 3
    assert mc.drop_delta(instance, coverage, 0) == -10
    assert mc.swap_delta(instance, coverage, 0, 3) == -7

    deployment = mc.greedy_deployment_sequence(instance, [0, 2, 3], budgets=[0, 1, 3])
    assert deployment.budgets() == [0, 1, 3]
    assert deployment.results[-1].objective == 29


def test_bounded_path_relinking_keeps_a_consistent_best_solution() -> None:
    instance = toy_instance()
    start = mc.select_by_marginal_gain(instance, 2)
    guide = [facility for facility in range(instance.n_facilities) if facility not in start.solution]

    relinked = mc.path_relink_fast(
        instance,
        start,
        guide,
        max_steps=1,
        candidate_width=1,
        refresh_interval=1,
    )

    coverage, objective = mc.compute_coverage_and_objective(instance, relinked.solution)
    assert relinked.objective >= start.objective
    assert relinked.objective == objective
    assert np.array_equal(relinked.coverage, coverage)
    assert len(relinked.solution) == len(start.solution)
    assert len(set(relinked.solution)) == len(relinked.solution)


def test_path_relinking_preserves_exact_swap_trace_and_best_state() -> None:
    instance = mc.build_instance_from_facility_map(
        {
            0: [0, 1],
            1: [1, 2],
            2: [2, 3],
            3: [3, 4],
        },
        [10, 5, 7, 3, 4],
        n_facilities=4,
    )
    coverage, objective = mc.compute_coverage_and_objective(instance, [0, 2])
    start = mc.HeuristicResult(
        solution=[0, 2],
        objective=objective,
        coverage=coverage,
        objectives=[objective],
        times=[0.0],
        total_time=0.0,
    )

    relinked = mc.path_relink_fast(
        instance, start, [1, 3], max_steps=None, candidate_width=None, refresh_interval=1
    )

    assert relinked.objectives == [25, 22, 19]
    assert relinked.objective == 25
    assert relinked.solution == [0, 2]
    assert np.array_equal(relinked.coverage, coverage)


def test_bounded_path_relinking_validates_search_limits() -> None:
    instance = toy_instance()
    start = mc.select_by_marginal_gain(instance, 2)
    with np.testing.assert_raises(ValueError):
        mc.path_relink_fast(instance, start, [1, 3], candidate_width=0)


def test_relinked_result_reports_all_search_stages() -> None:
    instance = toy_instance()
    config = mc.HeuristicConfig(
        constructors=("randomized",),
        randomized_repeats=2,
        local_search="first",
        use_path_relinking=True,
        path_relinking_max_steps=2,
        path_relinking_candidate_width=2,
        seed=3,
    )
    curve = mc.approximate_pareto_curve(instance, [2], config=config, select_best=False)
    assert len(curve.results) == 2
    for result in curve.results:
        assert result.total_seconds >= result.solve_seconds
        assert "path_relinking_attempted" in result.metadata


def test_local_search_move_limit_is_respected() -> None:
    instance = toy_instance()
    constructed = mc.select_by_marginal_gain(instance, 2)
    search = mc.SparseSwapLocalSearch.from_instance(instance)
    assert search.household_facility_matrix is None
    improved = search.improve(
        constructed.solution,
        coverage=constructed.coverage,
        objective=constructed.objective,
        max_moves=0,
    )
    assert improved.solution == constructed.solution
    assert improved.objective == constructed.objective
    assert len(improved.objectives) == 1
