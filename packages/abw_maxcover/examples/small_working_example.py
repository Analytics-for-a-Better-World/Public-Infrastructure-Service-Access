"""Small runnable example for abw_maxcover."""

import numpy as np

from abw_maxcover import (
    HeuristicConfig,
    approximate_pareto_curve,
    build_instance,
    greedy_deployment_sequence,
)


def main() -> None:
    weights = np.array([10, 7, 5, 4, 3], dtype=np.int64)
    ij = [[0, 1], [0, 2], [1, 2], [2, 3], [3]]
    ji = [[0, 1], [0, 2], [1, 2, 3], [3, 4]]

    instance = build_instance(
        weights,
        ij,
        ji,
        name="toy",
        validate_consistency=True,
    )

    config = HeuristicConfig(
        constructors=("greedy", "compact", "regreedy", "randomized"),
        randomized_repeats=2,
        local_search="first",
        seed=7,
    )

    curve = approximate_pareto_curve(instance, [3, 1, 2], config=config)
    print("Approximate Pareto:")
    for result in curve.results:
        print(result.to_record(total_weight=instance.total_weight))

    largest = max(curve.results, key=lambda result: result.budget)
    deployment = greedy_deployment_sequence(
        instance,
        largest.solution,
        budgets=[0, 1, 2, 3],
    )
    print("\nGreedy deployment sequence:")
    for result in deployment.results:
        print(result.to_record(total_weight=instance.total_weight))


if __name__ == "__main__":
    main()
