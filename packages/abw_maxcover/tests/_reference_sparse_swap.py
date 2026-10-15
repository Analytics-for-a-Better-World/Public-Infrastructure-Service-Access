from __future__ import annotations

from time import perf_counter

import numpy as np

from abw_maxcover._incremental_core import _deduplicate_solution, compute_coverage_and_objective
from abw_maxcover.results import HeuristicResult


def reference_improve(
    self,
    solution: list[int],
    coverage: np.ndarray | None = None,
    objective: int | None = None,
    *,
    max_moves: int | None = None,
    time_limit_seconds: float | None = None,
) -> HeuristicResult:
    if max_moves is not None and int(max_moves) < 0:
        raise ValueError("max_moves must be nonnegative or None")
    if time_limit_seconds is not None and float(time_limit_seconds) < 0:
        raise ValueError("time_limit_seconds must be nonnegative or None")
    instance = self.instance
    weights = instance.weights
    sol = _deduplicate_solution(solution)
    if coverage is None:
        cov, obj = compute_coverage_and_objective(instance, sol)
    else:
        cov = np.asarray(coverage, dtype=np.int32).copy()
        obj = int(weights[cov > 0].sum()) if objective is None else int(objective)
    objectives = [int(obj)]
    times = [0.0]
    covered = cov > 0
    uncovered_weight = np.where(covered, 0, weights)
    open_mask = np.zeros(instance.n_facilities, dtype=bool)
    for facility in sol:
        open_mask[int(facility)] = True
    start = perf_counter()
    while True:
        if max_moves is not None and len(objectives) - 1 >= int(max_moves):
            break
        if time_limit_seconds is not None and perf_counter() - start >= float(time_limit_seconds):
            break
        modified = False
        for position, removed in enumerate(sol):
            removed_i = int(removed)
            removed_cover = self.demand_of(removed_i)
            if not removed_cover.size:
                continue
            newly_uncovered = removed_cover[cov[removed_cover] == 1]
            loss = int(weights[newly_uncovered].sum()) if newly_uncovered.size else 0
            cov[removed_cover] -= 1
            if newly_uncovered.size:
                covered[newly_uncovered] = False
                uncovered_weight[newly_uncovered] = weights[newly_uncovered]
            candidates = self.collect_candidates(newly_uncovered, open_mask=open_mask, loss=loss)
            accepted = False
            for candidate in candidates:
                candidate_i = int(candidate)
                candidate_cover = self.demand_of(candidate_i)
                if not candidate_cover.size:
                    continue
                gain_in = int(uncovered_weight[candidate_cover].sum())
                if gain_in <= loss:
                    continue
                sol[position] = candidate_i
                open_mask[removed_i] = False
                open_mask[candidate_i] = True
                cov[candidate_cover] += 1
                newly_covered = candidate_cover[~covered[candidate_cover]]
                if newly_covered.size:
                    covered[newly_covered] = True
                    uncovered_weight[newly_covered] = 0
                obj += gain_in - loss
                modified = True
                accepted = True
                times.append(perf_counter() - start)
                objectives.append(int(obj))
                break
            if accepted:
                break
            cov[removed_cover] += 1
            if newly_uncovered.size:
                covered[newly_uncovered] = True
                uncovered_weight[newly_uncovered] = 0
        if not modified:
            break
    return HeuristicResult(
        solution=sol,
        objective=int(obj),
        coverage=cov,
        objectives=objectives,
        times=times,
        total_time=float(perf_counter() - start),
    )
