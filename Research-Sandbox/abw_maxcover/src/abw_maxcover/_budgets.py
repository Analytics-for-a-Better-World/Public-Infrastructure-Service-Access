"""Private helpers for requested and execution budget order."""

from __future__ import annotations

from collections.abc import Iterable


def normalise_budget_order(budgets: Iterable[int]) -> tuple[list[int], list[int]]:
    requested: list[int] = []
    seen: set[int] = set()
    for value in budgets:
        budget = int(value)
        if budget < 0 or budget in seen:
            continue
        seen.add(budget)
        requested.append(budget)
    return requested, sorted(requested)
