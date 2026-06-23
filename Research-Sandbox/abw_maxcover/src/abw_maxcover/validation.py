"""Validation helpers for instances, solutions, and curves."""

from __future__ import annotations

from typing import Literal

from .instance import MaxCoverInstance, _validate_biadjacency_consistency
from .results import MaxCoverCurve, MaxCoverResult
from ._incremental_core import compute_coverage_and_objective


def validate_instance(instance: MaxCoverInstance) -> None:
    _validate_biadjacency_consistency(instance)


def recompute_result(instance: MaxCoverInstance, result: MaxCoverResult) -> MaxCoverResult:
    coverage, objective = compute_coverage_and_objective(instance, result.solution)
    result.coverage = coverage
    result.objective = int(objective)
    return result


def validate_solution(
    instance: MaxCoverInstance,
    result: MaxCoverResult,
    *,
    expected_objective: int | None = None,
) -> None:
    _, objective = compute_coverage_and_objective(instance, result.solution)
    if result.objective is not None and int(result.objective) != int(objective):
        raise ValueError("result objective does not match recomputed coverage")
    if expected_objective is not None and int(expected_objective) != int(objective):
        raise ValueError("solution objective does not match expected objective")


def assert_curve_monotone(
    curve: MaxCoverCurve,
    *,
    field: Literal["objective"] = "objective",
) -> None:
    previous: int | None = None
    for result in sorted(curve.results, key=lambda item: int(item.budget)):
        value = getattr(result, field)
        if value is None:
            continue
        if previous is not None and int(value) < previous:
            raise ValueError(f"curve is not monotone in {field}")
        previous = int(value)


def assert_same_objective_definition(left: MaxCoverCurve, right: MaxCoverCurve) -> None:
    if left.metadata.get("objective_definition") != right.metadata.get("objective_definition"):
        if left.metadata.get("objective_definition") or right.metadata.get("objective_definition"):
            raise ValueError("curves appear to use different objective definitions")
