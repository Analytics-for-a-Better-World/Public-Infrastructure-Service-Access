"""High-level exact and approximate Pareto curve orchestration."""

from __future__ import annotations

from typing import Any, Callable, Literal

from ._budgets import normalise_budget_order
from .instance import MaxCoverInstance
from .results import CurveComparison, MaxCoverCurve, MaxCoverResult, best_by_budget
from .exact import GurobiConfig, PyomoConfig, solve_gurobi_curve, solve_pyomo_curve
from .heuristics import HeuristicConfig, run_heuristics

def exact_pareto_curve(
    instance: MaxCoverInstance,
    budgets: list[int],
    *,
    solver: Literal["gurobi", "pyomo"] = "gurobi",
    gurobi_config: GurobiConfig | None = None,
    pyomo_config: PyomoConfig | None = None,
    progress: Callable[[Any], Any] = lambda iterable: iterable,
    result_callback: Callable[[MaxCoverResult], Any] | None = None,
) -> MaxCoverCurve:
    if solver == "gurobi":
        return solve_gurobi_curve(
            instance,
            budgets,
            config=gurobi_config,
            progress=progress,
            result_callback=result_callback,
        )
    if solver == "pyomo":
        return solve_pyomo_curve(instance, budgets, config=pyomo_config, progress=progress)
    raise ValueError(f"unsupported solver: {solver}")


def compare_curves(
    reference: MaxCoverCurve,
    challenger: MaxCoverCurve,
    *,
    label_reference: str = "reference",
    label_challenger: str = "challenger",
) -> CurveComparison:
    reference_by_budget = {int(result.budget): result for result in reference.best_by_budget()}
    challenger_by_budget = {int(result.budget): result for result in challenger.best_by_budget()}
    records: list[dict[str, Any]] = []
    for budget in sorted(set(reference_by_budget) & set(challenger_by_budget)):
        ref = reference_by_budget[budget]
        ch = challenger_by_budget[budget]
        records.append(
            {
                "budget": budget,
                "reference_method": ref.method,
                "challenger_method": ch.method,
                "reference_objective": ref.objective,
                "challenger_objective": ch.objective,
                "delta_objective": None
                if ref.objective is None or ch.objective is None
                else int(ch.objective) - int(ref.objective),
            }
        )
    return CurveComparison(
        reference_label=label_reference,
        challenger_label=label_challenger,
        records=records,
        metadata={"reference_kind": reference.kind, "challenger_kind": challenger.kind},
    )


def coverage_gain_between_instances(coarse: MaxCoverCurve, fine: MaxCoverCurve) -> CurveComparison:
    return compare_curves(coarse, fine, label_reference="coarse", label_challenger="fine")

def approximate_pareto_curve(
    instance: MaxCoverInstance,
    budgets: list[int],
    *,
    config: HeuristicConfig | None = None,
    select_best: bool = True,
    progress: Callable[[Any], Any] = lambda iterable: iterable,
    result_callback: Callable[[MaxCoverResult], Any] | None = None,
) -> MaxCoverCurve:
    """Compute a heuristic Pareto approximation with optional budget checkpoints.

    ``result_callback`` receives the best portfolio result as soon as each
    execution budget is complete. Budgets are evaluated in sorted order even
    when the returned curve follows a different requested order.
    """

    def checkpoint(_budget: int, candidates: list[MaxCoverResult]) -> None:
        if result_callback is None:
            return
        best = best_by_budget(candidates)
        if best:
            result_callback(best[0])

    curve = run_heuristics(
        instance,
        budgets,
        config=config,
        progress=progress,
        budget_callback=checkpoint if result_callback is not None else None,
    )
    if not select_best:
        curve.kind = "approximation_all"
        return curve
    requested_budgets, _ = normalise_budget_order(budgets)
    best = {int(result.budget): result for result in curve.best_by_budget()}
    return MaxCoverCurve(
        instance_name=instance.name,
        kind="approximation_best",
        results=[best[budget] for budget in requested_budgets if budget in best],
        metadata={**curve.metadata, "candidate_result_count": len(curve.results)},
    )
