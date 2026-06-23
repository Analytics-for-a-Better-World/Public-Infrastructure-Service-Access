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
    config = mc.HeuristicConfig(
        constructors=("greedy", "compact", "regreedy", "randomized"),
        randomized_repeats=1,
        local_search="first",
        seed=11,
    )
    curve = mc.approximate_pareto_curve(instance, [3, 1, 2], config=config)
    assert curve.budgets() == [3, 1, 2]
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
