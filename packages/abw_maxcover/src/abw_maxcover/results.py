"""Typed result objects and primitive record conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class HeuristicResult:
    solution: list[int]
    objective: int
    coverage: np.ndarray
    objectives: list[int]
    times: list[float]
    total_time: float


@dataclass(slots=True)
class MaxCoverResult:
    budget: int
    method: str
    objective: int | None
    solution: list[int]
    status: str = "ok"
    upper_bound: float | None = None
    mip_gap: float | None = None
    coverage: np.ndarray | None = field(default=None, repr=False)
    model_seconds: float = 0.0
    solve_seconds: float = 0.0
    total_seconds: float = 0.0
    construction_objective: int | None = None
    construction_seconds: float | None = None
    local_search_moves: int | None = None
    seed: int | None = None
    repeat: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_count(self) -> int:
        return len(self.solution)

    @property
    def covered_count(self) -> int | None:
        if self.coverage is None:
            return None
        return int(np.count_nonzero(self.coverage > 0))

    def to_record(self, *, total_weight: int | None = None) -> dict[str, Any]:
        coverage_percent = None
        if total_weight and self.objective is not None:
            coverage_percent = 100.0 * float(self.objective) / float(total_weight)
        return {
            "budget": int(self.budget),
            "method": self.method,
            "status": self.status,
            "objective": self.objective,
            "coverage_percent": coverage_percent,
            "upper_bound": self.upper_bound,
            "mip_gap": self.mip_gap,
            "selected_count": self.selected_count,
            "covered_count": self.covered_count,
            "model_seconds": self.model_seconds,
            "solve_seconds": self.solve_seconds,
            "total_seconds": self.total_seconds,
            "construction_objective": self.construction_objective,
            "construction_seconds": self.construction_seconds,
            "local_search_moves": self.local_search_moves,
            "seed": self.seed,
            "repeat": self.repeat,
            **self.metadata,
        }


@dataclass(slots=True)
class MaxCoverCurve:
    instance_name: str
    kind: str
    results: list[MaxCoverResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def best_by_budget(self) -> list[MaxCoverResult]:
        return best_by_budget(self.results)

    def budgets(self) -> list[int]:
        return [int(result.budget) for result in self.results]

    def to_records(self, *, total_weight: int | None = None) -> list[dict[str, Any]]:
        return [result.to_record(total_weight=total_weight) for result in self.results]


@dataclass(slots=True)
class CurveComparison:
    reference_label: str
    challenger_label: str
    records: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self.records]


def best_by_budget(results: list[MaxCoverResult]) -> list[MaxCoverResult]:
    grouped: dict[int, list[MaxCoverResult]] = {}
    for result in results:
        grouped.setdefault(int(result.budget), []).append(result)
    best: list[MaxCoverResult] = []
    for budget in sorted(grouped):
        candidates = grouped[budget]
        best.append(
            min(
                candidates,
                key=lambda item: (
                    -(item.objective if item.objective is not None else -1),
                    item.total_seconds,
                    item.method,
                ),
            )
        )
    return best
