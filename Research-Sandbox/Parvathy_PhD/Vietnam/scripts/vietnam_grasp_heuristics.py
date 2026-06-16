"""Scalable GRASP and path relinking for fresh PISA max-cover instances."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import random
import sys
from time import perf_counter as pc
from typing import Literal

import numpy as np

APPROX_SRC = Path(__file__).resolve().parents[3] / "approximated_tradeoff" / "src"
sys.path.insert(0, str(APPROX_SRC))

import mc_heuristics as mch

ConstructorName = Literal["greedy", "randomized", "sample", "random_plus"]
LocalSearchName = Literal["first", "first_sparse", "none"]
PathRelinkingName = Literal["fast", "original"]


@dataclass(slots=True)
class GraspRecord:
    iteration: int
    construction_objective: int
    local_search_objective: int
    path_relinking_objective: int
    best_objective: int
    construction_seconds: float
    local_search_seconds: float
    path_relinking_seconds: float
    iteration_seconds: float
    total_seconds: float
    pool_size: int
    selected_solution: list[int]


def empty_result(instance: mch.MaxCoverInstance) -> mch.HeuristicResult:
    return mch.HeuristicResult(
        solution=[],
        objective=0,
        coverage=np.zeros(instance.n_households, dtype=np.int32),
        objectives=[0],
        times=[0.0],
        total_time=0.0,
    )


def _initial_gain(instance: mch.MaxCoverInstance) -> np.ndarray:
    gain = np.zeros(instance.n_facilities, dtype=np.int64)
    counts = instance.ji_indptr[1:] - instance.ji_indptr[:-1]
    nonempty = counts > 0
    if nonempty.any():
        gain[nonempty] = np.add.reduceat(instance.w[instance.ji_indices], instance.ji_indptr[:-1][nonempty])
    return gain


def _top_rcl(gain: np.ndarray, rcl_size: int) -> np.ndarray:
    positive = np.flatnonzero(gain > 0)
    if positive.size == 0:
        return positive
    k = max(1, min(int(rcl_size), int(positive.size)))
    if k == positive.size:
        return positive[np.argsort(-gain[positive])]
    partial = positive[np.argpartition(gain[positive], -k)[-k:]]
    return partial[np.argsort(-gain[partial])]


def _choose_facility(
    gain: np.ndarray,
    *,
    constructor: ConstructorName,
    rcl_size: int,
    sample_size: int,
    step: int,
    random_plus_count: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if constructor == "greedy":
        facility = int(np.argmax(gain))
        facility_gain = int(gain[facility])
        return (facility, facility_gain) if facility_gain > 0 else (-1, 0)

    positive = np.flatnonzero(gain > 0)
    if positive.size == 0:
        return -1, 0

    if constructor == "sample":
        k = max(1, min(int(sample_size), int(positive.size)))
        sampled = rng.choice(positive, size=k, replace=False)
        facility = int(sampled[int(np.argmax(gain[sampled]))])
        return facility, int(gain[facility])

    if constructor == "random_plus" and step < random_plus_count:
        facility = int(rng.choice(positive))
        return facility, int(gain[facility])

    rcl = _top_rcl(gain, rcl_size)
    if rcl.size == 0:
        return -1, 0
    facility = int(rng.choice(rcl))
    return facility, int(gain[facility])


def budgeted_construct(
    instance: mch.MaxCoverInstance,
    budget: int,
    *,
    constructor: ConstructorName = "greedy",
    rcl_size: int = 25,
    sample_size: int = 250,
    random_plus_fraction: float = 0.15,
    seed: int | None = None,
) -> mch.HeuristicResult:
    """Budgeted greedy/RCL construction with the approximated_tradeoff update pattern."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    rng = np.random.default_rng(seed)
    gain = _initial_gain(instance)
    uncovered = np.ones(instance.n_households, dtype=bool)
    coverage = np.zeros(instance.n_households, dtype=np.int32)
    solution: list[int] = []
    objective = 0
    objectives: list[int] = [0]
    times: list[float] = [0.0]
    random_plus_count = max(0, min(int(round(random_plus_fraction * budget)), int(budget)))
    start = pc()

    for step in range(int(budget)):
        facility, facility_gain = _choose_facility(
            gain,
            constructor=constructor,
            rcl_size=rcl_size,
            sample_size=sample_size,
            step=step,
            random_plus_count=random_plus_count,
            rng=rng,
        )
        if facility < 0 or facility_gain <= 0:
            break

        solution.append(facility)
        objective += facility_gain
        gain[facility] = -1

        j0 = int(instance.ji_indptr[facility])
        j1 = int(instance.ji_indptr[facility + 1])
        if j0 != j1:
            covered = instance.ji_indices[j0:j1]
            newly_covered = covered[uncovered[covered]]
            coverage[covered] += 1

            if newly_covered.size:
                uncovered[newly_covered] = False
                starts = instance.ij_indptr[newly_covered]
                ends = instance.ij_indptr[newly_covered + 1]
                sizes = ends - starts
                total_nnz = int(sizes.sum())
                if total_nnz:
                    touched = np.empty(total_nnz, dtype=np.int32)
                    weights = np.empty(total_nnz, dtype=np.int64)
                    pos = 0
                    for household, s, e in zip(newly_covered, starts, ends):
                        m = int(e - s)
                        if not m:
                            continue
                        nxt = pos + m
                        touched[pos:nxt] = instance.ij_indices[s:e]
                        weights[pos:nxt] = instance.w[int(household)]
                        pos = nxt
                    if pos:
                        touched = touched[:pos]
                        weights = weights[:pos]
                        mask = gain[touched] >= 0
                        if mask.any():
                            np.add.at(gain, touched[mask], -weights[mask])

        objectives.append(int(objective))
        times.append(pc() - start)

    return mch.HeuristicResult(
        solution=solution,
        objective=int(objective),
        coverage=coverage,
        objectives=objectives,
        times=times,
        total_time=float(pc() - start),
    )


def improve_local_search(
    instance: mch.MaxCoverInstance,
    result: mch.HeuristicResult,
    *,
    local_search: LocalSearchName = "first",
    sparse_local_search: object | None = None,
) -> mch.HeuristicResult:
    if local_search == "none":
        return mch.HeuristicResult(
            solution=list(result.solution),
            objective=int(result.objective),
            coverage=result.coverage.copy(),
            objectives=[int(result.objective)],
            times=[0.0],
            total_time=0.0,
        )
    if local_search == "first_sparse":
        if sparse_local_search is None:
            from vietnam_sparse_local_search import SparseSwapLocalSearch

            sparse_local_search = SparseSwapLocalSearch.from_instance(instance)
        return sparse_local_search.improve(
            solution=result.solution,
            coverage=result.coverage,
            objective=result.objective,
        )
    if local_search != "first":
        raise ValueError(f"Unsupported local search: {local_search}")
    return mch.swap_first_improving(
        instance=instance,
        solution=result.solution,
        coverage=result.coverage,
        objective=result.objective,
    )


def _swap_delta(instance: mch.MaxCoverInstance, coverage: np.ndarray, swap_out: int, swap_in: int) -> int:
    out_cover = instance.households_of(int(swap_out))
    if out_cover.size:
        newly_uncovered = out_cover[coverage[out_cover] == 1]
        loss = int(instance.w[newly_uncovered].sum()) if newly_uncovered.size else 0
    else:
        newly_uncovered = np.empty(0, dtype=np.int32)
        loss = 0

    in_cover = instance.households_of(int(swap_in))
    if not in_cover.size:
        return -loss
    gain = int(instance.w[in_cover[coverage[in_cover] == 0]].sum())
    if newly_uncovered.size:
        recovered = np.intersect1d(in_cover, newly_uncovered, assume_unique=True)
        if recovered.size:
            gain += int(instance.w[recovered].sum())
    return gain - loss


def _apply_swap(instance: mch.MaxCoverInstance, coverage: np.ndarray, swap_out: int, swap_in: int) -> None:
    out_cover = instance.households_of(int(swap_out))
    if out_cover.size:
        coverage[out_cover] -= 1
    in_cover = instance.households_of(int(swap_in))
    if in_cover.size:
        coverage[in_cover] += 1


def path_relink(
    instance: mch.MaxCoverInstance,
    start_result: mch.HeuristicResult,
    guide_solution: list[int],
) -> mch.HeuristicResult:
    """Relink from start_result toward guide_solution, keeping the best path state."""
    start_time = pc()
    solution = list(start_result.solution)
    coverage = start_result.coverage.copy()
    objective = int(start_result.objective)
    guide_set = set(int(x) for x in guide_solution)
    solution_set = set(int(x) for x in solution)
    to_exit = [fac for fac in solution if fac not in guide_set]
    to_enter = [fac for fac in guide_solution if int(fac) not in solution_set]
    best_solution = list(solution)
    best_coverage = coverage.copy()
    best_objective = int(objective)
    objectives = [int(objective)]
    times = [0.0]

    while to_exit and to_enter:
        best_pair: tuple[int, int] | None = None
        best_delta: int | None = None
        for swap_out in to_exit:
            for swap_in in to_enter:
                delta = _swap_delta(instance, coverage, int(swap_out), int(swap_in))
                if best_delta is None or delta > best_delta:
                    best_delta = int(delta)
                    best_pair = (int(swap_out), int(swap_in))
        if best_pair is None or best_delta is None:
            break
        swap_out, swap_in = best_pair
        solution.remove(swap_out)
        solution.append(swap_in)
        _apply_swap(instance, coverage, swap_out, swap_in)
        objective += int(best_delta)
        to_exit.remove(swap_out)
        to_enter.remove(swap_in)
        objectives.append(int(objective))
        times.append(pc() - start_time)
        if objective > best_objective:
            best_objective = int(objective)
            best_solution = list(solution)
            best_coverage = coverage.copy()

    return mch.HeuristicResult(
        solution=best_solution,
        objective=int(best_objective),
        coverage=best_coverage,
        objectives=objectives,
        times=times,
        total_time=float(pc() - start_time),
    )


def path_relink_fast(
    instance: mch.MaxCoverInstance,
    start_result: mch.HeuristicResult,
    guide_solution: list[int],
) -> mch.HeuristicResult:
    """Relink with the same best-pair rule using cached removal-side work."""
    start_time = pc()
    solution = list(start_result.solution)
    coverage = start_result.coverage.copy()
    objective = int(start_result.objective)
    guide_set = set(int(x) for x in guide_solution)
    solution_set = set(int(x) for x in solution)
    to_exit = [fac for fac in solution if fac not in guide_set]
    to_enter = [fac for fac in guide_solution if int(fac) not in solution_set]
    best_solution = list(solution)
    best_coverage = coverage.copy()
    best_objective = int(objective)
    objectives = [int(objective)]
    times = [0.0]
    w = instance.w
    facility_households = tuple(
        instance.ji_indices[int(instance.ji_indptr[facility]) : int(instance.ji_indptr[facility + 1])]
        for facility in range(instance.n_facilities)
    )
    recovered_mask = np.zeros(instance.n_households, dtype=bool)

    while to_exit and to_enter:
        best_pair: tuple[int, int] | None = None
        best_delta: int | None = None

        for swap_out in to_exit:
            out_cover = facility_households[int(swap_out)]
            if out_cover.size:
                newly_uncovered = out_cover[coverage[out_cover] == 1]
                loss = int(w[newly_uncovered].sum()) if newly_uncovered.size else 0
            else:
                newly_uncovered = np.empty(0, dtype=np.int32)
                loss = 0

            if newly_uncovered.size:
                recovered_mask[newly_uncovered] = True

            for swap_in in to_enter:
                in_cover = facility_households[int(swap_in)]
                if not in_cover.size:
                    delta = -loss
                else:
                    gain_mask = coverage[in_cover] == 0
                    if newly_uncovered.size:
                        gain_mask = gain_mask | recovered_mask[in_cover]
                    gain = int(w[in_cover[gain_mask]].sum()) if gain_mask.any() else 0
                    delta = gain - loss

                if best_delta is None or delta > best_delta:
                    best_delta = int(delta)
                    best_pair = (int(swap_out), int(swap_in))

            if newly_uncovered.size:
                recovered_mask[newly_uncovered] = False

        if best_pair is None or best_delta is None:
            break

        swap_out, swap_in = best_pair
        solution.remove(swap_out)
        solution.append(swap_in)
        out_cover = facility_households[swap_out]
        if out_cover.size:
            coverage[out_cover] -= 1
        in_cover = facility_households[swap_in]
        if in_cover.size:
            coverage[in_cover] += 1
        objective += int(best_delta)
        to_exit.remove(swap_out)
        to_enter.remove(swap_in)
        objectives.append(int(objective))
        times.append(pc() - start_time)
        if objective > best_objective:
            best_objective = int(objective)
            best_solution = list(solution)
            best_coverage = coverage.copy()

    return mch.HeuristicResult(
        solution=best_solution,
        objective=int(best_objective),
        coverage=best_coverage,
        objectives=objectives,
        times=times,
        total_time=float(pc() - start_time),
    )


def update_pool(
    pool: list[mch.HeuristicResult],
    candidate: mch.HeuristicResult,
    *,
    max_pool: int = 8,
    min_diversity_fraction: float = 0.2,
) -> list[mch.HeuristicResult]:
    candidate = copy.deepcopy(candidate)
    if len(pool) < max_pool:
        pool.append(candidate)
        pool.sort(key=lambda item: item.objective, reverse=True)
        return pool
    objectives = np.asarray([item.objective for item in pool], dtype=np.int64)
    if candidate.objective > int(objectives.min()):
        size = max(1, len(candidate.solution))
        diffs = np.asarray([
            len(set(candidate.solution).symmetric_difference(set(item.solution))) / size
            for item in pool
        ])
        if candidate.objective > int(objectives.max()) or diffs.min(initial=1.0) >= min_diversity_fraction:
            pool[int(np.argmin(objectives))] = candidate
        else:
            similar = np.where(diffs < min_diversity_fraction)[0]
            if similar.size:
                worst_similar = similar[int(np.argmin(objectives[similar]))]
                if candidate.objective >= pool[int(worst_similar)].objective:
                    pool[int(worst_similar)] = candidate
    pool.sort(key=lambda item: item.objective, reverse=True)
    return pool[:max_pool]


def run_grasp(
    instance: mch.MaxCoverInstance,
    budget: int,
    *,
    time_limit_seconds: float = 300.0,
    max_iterations: int | None = None,
    constructor: ConstructorName = "randomized",
    rcl_size: int = 25,
    sample_size: int = 250,
    random_plus_fraction: float = 0.15,
    local_search: LocalSearchName = "first_sparse",
    path_relinking: bool = True,
    path_relinking_method: PathRelinkingName = "fast",
    seed: int = 42,
    max_pool: int = 8,
) -> tuple[mch.HeuristicResult, list[GraspRecord]]:
    rng = random.Random(seed)
    start = pc()
    best = empty_result(instance)
    pool: list[mch.HeuristicResult] = []
    records: list[GraspRecord] = []
    iteration = 0
    sparse_local_search = None
    if local_search == "first_sparse":
        from vietnam_sparse_local_search import SparseSwapLocalSearch

        sparse_local_search = SparseSwapLocalSearch.from_instance(instance)

    while True:
        if iteration > 0 and pc() - start >= time_limit_seconds:
            break
        if max_iterations is not None and iteration >= max_iterations:
            break
        iteration += 1
        iter_start = pc()

        constructed = budgeted_construct(
            instance,
            budget,
            constructor=constructor,
            rcl_size=rcl_size,
            sample_size=sample_size,
            random_plus_fraction=random_plus_fraction,
            seed=seed + iteration,
        )
        improved = improve_local_search(
            instance,
            constructed,
            local_search=local_search,
            sparse_local_search=sparse_local_search,
        )
        pr_result = improved
        pr_seconds = 0.0

        if path_relinking and pool:
            guide = rng.choice(pool)
            relink = path_relink_fast if path_relinking_method == "fast" else path_relink
            forward = relink(instance, improved, guide.solution)
            backward = relink(instance, guide, improved.solution)
            pr_result = forward if forward.objective >= backward.objective else backward
            pr_seconds = forward.total_time + backward.total_time

        candidate = pr_result if pr_result.objective >= improved.objective else improved
        pool = update_pool(pool, candidate, max_pool=max_pool)
        if candidate.objective > best.objective:
            best = copy.deepcopy(candidate)

        records.append(
            GraspRecord(
                iteration=iteration,
                construction_objective=int(constructed.objective),
                local_search_objective=int(improved.objective),
                path_relinking_objective=int(pr_result.objective),
                best_objective=int(best.objective),
                construction_seconds=float(constructed.total_time),
                local_search_seconds=float(improved.total_time),
                path_relinking_seconds=float(pr_seconds),
                iteration_seconds=float(pc() - iter_start),
                total_seconds=float(pc() - start),
                pool_size=len(pool),
                selected_solution=list(candidate.solution),
            )
        )

        if constructor == "greedy" and not path_relinking:
            break

    return best, records
