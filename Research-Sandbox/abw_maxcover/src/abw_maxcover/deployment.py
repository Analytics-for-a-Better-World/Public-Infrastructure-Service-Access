"""Deployment sequencing from a fixed selected facility pool."""

from __future__ import annotations

from typing import Any, Callable, Literal

from ._budgets import normalise_budget_order
from .instance import MaxCoverInstance
from .results import MaxCoverCurve, MaxCoverResult
from .exact import GurobiConfig, PyomoConfig
from ._incremental_core import _deduplicate_solution, prefix_result, select_by_marginal_gain

def greedy_deployment_sequence(
    instance: MaxCoverInstance,
    solution_pool: list[int] | tuple[int, ...] | np.ndarray,
    *,
    budgets: list[int] | None = None,
    method: str = "greedy_deployment_sequence",
    metadata: dict[str, Any] | None = None,
) -> MaxCoverCurve:
    """Order a fixed pool of facilities by greedy marginal coverage gain.

    This mirrors the Lancet Nepal deployment strategy: first choose the full
    expansion set, then decide the roll-out order by repeatedly opening the
    remaining facility that adds the largest new covered population.  The final
    step reaches the same objective as the original pool; intermediate steps
    are a pragmatic deployment curve, not an independently optimal Pareto curve.
    """
    pool = [
        facility
        for facility in _deduplicate_solution(solution_pool)
        if 0 <= int(facility) < instance.n_facilities
    ]
    source_pool_size = len(pool)
    requested = (
        normalise_budget_order(budgets)[0]
        if budgets is not None
        else list(range(1, source_pool_size + 1))
    )
    clipped_requested: list[int] = []
    seen_requested: set[int] = set()
    for value in requested:
        clipped = min(int(value), source_pool_size)
        if clipped in seen_requested:
            continue
        seen_requested.add(clipped)
        clipped_requested.append(clipped)
    requested = clipped_requested

    ordered = select_by_marginal_gain(instance, source_pool_size, candidate_pool=pool)
    result_by_budget: dict[int, MaxCoverResult] = {}

    for step in requested:
        prefix = prefix_result(instance, ordered, step)
        objective = int(prefix.objective)
        idx = min(int(step), len(ordered.objectives) - 1)
        previous_objective = int(ordered.objectives[idx - 1]) if idx > 0 else 0
        marginal_gain = objective - previous_objective
        elapsed = float(ordered.times[min(step, len(ordered.times) - 1)])
        result_by_budget[step] = MaxCoverResult(
            budget=step,
            method=method,
            objective=objective,
            solution=list(prefix.solution),
            status="ok",
            coverage=prefix.coverage.copy(),
            solve_seconds=elapsed,
            total_seconds=elapsed,
            metadata={
                "source_pool_size": source_pool_size,
                "marginal_gain": int(marginal_gain),
                "deployment_step": step,
                **dict(metadata or {}),
            },
        )

    return MaxCoverCurve(
        instance_name=instance.name,
        kind="deployment_sequence",
        results=[result_by_budget[budget] for budget in requested if budget in result_by_budget],
        metadata={
            "source_pool_size": source_pool_size,
            "requested_budgets": requested,
            "execution_budgets": sorted(requested),
            **dict(metadata or {}),
        },
    )


def optimize_then_greedy_deployment(
    instance: MaxCoverInstance,
    largest_budget: int,
    *,
    deployment_budgets: list[int] | None = None,
    solver: Literal["gurobi", "pyomo"] = "gurobi",
    gurobi_config: GurobiConfig | None = None,
    pyomo_config: PyomoConfig | None = None,
    progress: Callable[[Any], Any] = lambda iterable: iterable,
) -> tuple[MaxCoverResult, MaxCoverCurve]:
    """Solve the largest budget, then greedily sequence that solution.

    The returned deployment curve answers "what if we implement the chosen
    expansion set one facility at a time?"  It should be shown separately from
    exact Pareto curves because its intermediate budgets are constrained to the
    largest-budget solution pool.
    """
    from .pareto import exact_pareto_curve

    exact_curve = exact_pareto_curve(
        instance,
        [int(largest_budget)],
        solver=solver,
        gurobi_config=gurobi_config,
        pyomo_config=pyomo_config,
        progress=progress,
    )
    exact_result = exact_curve.results[0]
    deployment_curve = greedy_deployment_sequence(
        instance,
        exact_result.solution,
        budgets=deployment_budgets,
        method=f"{exact_result.method}_greedy_deployment",
        metadata={
            "source_exact_budget": int(largest_budget),
            "source_exact_objective": exact_result.objective,
            "source_exact_status": exact_result.status,
            "source_exact_method": exact_result.method,
        },
    )
    return exact_result, deployment_curve
