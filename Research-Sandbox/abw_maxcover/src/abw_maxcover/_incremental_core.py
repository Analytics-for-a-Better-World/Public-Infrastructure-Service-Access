"""Private shared incremental coverage engine."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal, Sequence

import numpy as np

from .instance import MaxCoverInstance
from .results import HeuristicResult

ConstructorName = Literal[
    "greedy",
    "compact",
    "regreedy",
    "re_greedy",
    "randomized",
    "grasp",
    "sample",
    "random_plus",
]
LocalSearchName = Literal["none", "first", "first_sparse"]
DETERMINISTIC_CONSTRUCTORS = {"greedy", "compact", "regreedy"}
RANDOMIZED_CONSTRUCTORS = {"randomized", "sample", "random_plus"}


def _deduplicate_solution(solution: list[int] | tuple[int, ...] | np.ndarray) -> list[int]:
    return list(dict.fromkeys(int(facility) for facility in solution))


def normalise_constructor(constructor: str) -> str:
    if constructor == "re_greedy":
        return "regreedy"
    if constructor == "grasp":
        return "randomized"
    return constructor


def compute_coverage_and_objective(
    instance: MaxCoverInstance,
    solution: list[int] | tuple[int, ...] | np.ndarray,
) -> tuple[np.ndarray, int]:
    sol = _deduplicate_solution(solution)
    coverage = np.zeros(instance.n_demand, dtype=np.int32)
    for facility in sol:
        demand = instance.demand_of(int(facility))
        if demand.size:
            coverage[demand] += 1
    return coverage, int(instance.weights[coverage > 0].sum())


def _initial_gain(instance: MaxCoverInstance) -> np.ndarray:
    gain = np.zeros(instance.n_facilities, dtype=np.int64)
    counts = instance.ji_indptr[1:] - instance.ji_indptr[:-1]
    nonempty = counts > 0
    if nonempty.any():
        gain[nonempty] = np.add.reduceat(
            instance.weights[instance.ji_indices],
            instance.ji_indptr[:-1][nonempty],
        )
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
    constructor = normalise_constructor(constructor)
    if constructor in {"compact", "regreedy"}:
        constructor = "greedy"
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

    if constructor not in RANDOMIZED_CONSTRUCTORS:
        raise ValueError(f"unsupported constructor: {constructor}")

    rcl = _top_rcl(gain, rcl_size)
    if rcl.size == 0:
        return -1, 0
    facility = int(rng.choice(rcl))
    return facility, int(gain[facility])


def budgeted_construct(
    instance: MaxCoverInstance,
    budget: int,
    *,
    constructor: ConstructorName = "greedy",
    rcl_size: int = 25,
    sample_size: int = 250,
    random_plus_fraction: float = 0.15,
    seed: int | None = None,
    initial_solution: Sequence[int] = (),
    candidate_pool: Sequence[int] | None = None,
) -> HeuristicResult:
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    constructor = normalise_constructor(constructor)

    start = perf_counter()
    target_size = int(budget)
    rng = np.random.default_rng(seed)
    allowed_mask = np.ones(instance.n_facilities, dtype=bool)
    if candidate_pool is not None:
        allowed_mask[:] = False
        for facility in candidate_pool:
            facility_i = int(facility)
            if 0 <= facility_i < instance.n_facilities:
                allowed_mask[facility_i] = True
    solution = [
        facility
        for facility in _deduplicate_solution(list(initial_solution))
        if 0 <= int(facility) < instance.n_facilities and allowed_mask[int(facility)]
    ][:target_size]
    selected_mask = np.zeros(instance.n_facilities, dtype=bool)
    coverage = np.zeros(instance.n_demand, dtype=np.int32)
    for facility in solution:
        facility_i = int(facility)
        if selected_mask[facility_i]:
            continue
        selected_mask[facility_i] = True
        demand = instance.demand_of(facility_i)
        if demand.size:
            coverage[demand] += 1
    uncovered = coverage == 0
    objective = int(instance.weights[~uncovered].sum())
    objectives: list[int] = [objective]
    times: list[float] = [0.0]
    remaining_budget = max(0, target_size - len(solution))
    random_plus_count = max(0, min(int(round(random_plus_fraction * remaining_budget)), remaining_budget))

    if solution or candidate_pool is not None:
        gain = np.full(instance.n_facilities, -1, dtype=np.int64)
        candidates = np.flatnonzero(allowed_mask & ~selected_mask)
        for facility in candidates:
            demand = instance.demand_of(int(facility))
            if demand.size:
                gain[int(facility)] = int(instance.weights[demand[uncovered[demand]]].sum())
            else:
                gain[int(facility)] = 0
    else:
        gain = _initial_gain(instance)

    for step in range(remaining_budget):
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
        selected_mask[facility] = True
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
                    for demand, row_start, row_end in zip(newly_covered, starts, ends):
                        width = int(row_end - row_start)
                        if not width:
                            continue
                        nxt = pos + width
                        touched[pos:nxt] = instance.ij_indices[row_start:row_end]
                        weights[pos:nxt] = instance.weights[int(demand)]
                        pos = nxt
                    if pos:
                        touched = touched[:pos]
                        weights = weights[:pos]
                        mask = gain[touched] >= 0
                        if mask.any():
                            np.add.at(gain, touched[mask], -weights[mask])

        objectives.append(int(objective))
        times.append(perf_counter() - start)

    return HeuristicResult(
        solution=solution,
        objective=int(objective),
        coverage=coverage,
        objectives=objectives,
        times=times,
        total_time=float(perf_counter() - start),
    )


def greedy_construct(instance: MaxCoverInstance) -> HeuristicResult:
    return budgeted_construct(instance, instance.n_facilities, constructor="greedy")


def prefix_result(instance: MaxCoverInstance, result: HeuristicResult, budget: int) -> HeuristicResult:
    solution = list(result.solution[: max(0, int(budget))])
    coverage, objective = compute_coverage_and_objective(instance, solution)
    idx = min(len(solution), len(result.times) - 1)
    return HeuristicResult(
        solution=solution,
        objective=int(objective),
        coverage=coverage,
        objectives=result.objectives[: idx + 1],
        times=result.times[: idx + 1],
        total_time=float(result.times[idx]) if result.times else 0.0,
    )


def swap_first_improving(
    instance: MaxCoverInstance,
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
    weights = instance.weights
    sol = _deduplicate_solution(solution)
    if coverage is None:
        cov, obj = compute_coverage_and_objective(instance, sol)
    else:
        cov = np.asarray(coverage, dtype=np.int32).copy()
        obj = int(weights[cov > 0].sum()) if objective is None else int(objective)

    objectives = [int(obj)]
    times = [0.0]
    base_gain = _initial_gain(instance)
    covered = cov > 0
    uncovered_weight = np.where(covered, 0, weights)
    open_mask = np.zeros(instance.n_facilities, dtype=bool)
    touched_mask = np.zeros(instance.n_facilities, dtype=bool)
    touched_list: list[int] = []

    def refresh_open_mask() -> None:
        open_mask.fill(False)
        for facility in sol:
            open_mask[int(facility)] = True

    def collect_candidates(newly_uncovered: np.ndarray) -> np.ndarray:
        for demand in newly_uncovered:
            for facility in instance.facilities_of(int(demand)):
                facility_i = int(facility)
                if open_mask[facility_i] or touched_mask[facility_i]:
                    continue
                touched_mask[facility_i] = True
                touched_list.append(facility_i)
        if not touched_list:
            return np.empty(0, dtype=np.int32)
        out = np.asarray(touched_list, dtype=np.int32)
        for facility in touched_list:
            touched_mask[facility] = False
        touched_list.clear()
        return out

    start = perf_counter()
    while True:
        if max_moves is not None and len(objectives) - 1 >= int(max_moves):
            break
        if time_limit_seconds is not None and perf_counter() - start >= float(time_limit_seconds):
            break
        refresh_open_mask()
        modified = False
        for position, removed in enumerate(sol):
            removed_i = int(removed)
            removed_cover = instance.demand_of(removed_i)
            if not removed_cover.size:
                continue
            newly_uncovered = removed_cover[cov[removed_cover] == 1]
            loss = int(weights[newly_uncovered].sum()) if newly_uncovered.size else 0
            cov[removed_cover] -= 1
            if newly_uncovered.size:
                covered[newly_uncovered] = False
                uncovered_weight[newly_uncovered] = weights[newly_uncovered]

            candidates = collect_candidates(newly_uncovered)
            if candidates.size:
                candidates = candidates[base_gain[candidates] > loss]
                if candidates.size:
                    candidates = candidates[np.argsort(-base_gain[candidates])]

            accepted = False
            for candidate in candidates:
                candidate_i = int(candidate)
                candidate_cover = instance.demand_of(candidate_i)
                if not candidate_cover.size:
                    continue
                gain_in = int(uncovered_weight[candidate_cover].sum())
                if gain_in <= loss:
                    continue
                sol[position] = candidate_i
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


def drop_redundant_facilities(
    instance: MaxCoverInstance,
    solution: list[int],
    coverage: np.ndarray | None = None,
    objective: int | None = None,
) -> HeuristicResult:
    weights = instance.weights
    sol = _deduplicate_solution(solution)
    if coverage is None:
        cov, obj = compute_coverage_and_objective(instance, sol)
    else:
        cov = np.asarray(coverage, dtype=np.int32).copy()
        obj = int(weights[cov > 0].sum()) if objective is None else int(objective)

    objectives = [int(obj)]
    times = [0.0]
    start = perf_counter()
    i = 0
    while i < len(sol):
        facility = int(sol[i])
        demand = instance.demand_of(facility)
        if not demand.size or np.all(cov[demand] >= 2):
            if demand.size:
                cov[demand] -= 1
            sol.pop(i)
            times.append(perf_counter() - start)
            objectives.append(int(obj))
            continue
        i += 1

    if int(weights[cov > 0].sum()) != int(obj):
        raise RuntimeError("redundant facility removal changed the objective")
    return HeuristicResult(
        solution=sol,
        objective=int(obj),
        coverage=cov,
        objectives=objectives,
        times=times,
        total_time=float(perf_counter() - start),
    )


class SparseSwapLocalSearch:
    """Raw-CSR first-improving swap local search."""

    def __init__(
        self,
        instance: MaxCoverInstance,
        household_facility_matrix: Any,
        base_gain: np.ndarray,
        facility_demand: tuple[np.ndarray, ...] | None,
    ) -> None:
        self.instance = instance
        self.household_facility_matrix = household_facility_matrix
        self.base_gain = base_gain
        self.facility_demand = facility_demand

    @classmethod
    def from_instance(
        cls,
        instance: MaxCoverInstance,
        *,
        cache_facility_demand: bool | None = None,
    ) -> "SparseSwapLocalSearch":
        if cache_facility_demand is None:
            cache_facility_demand = instance.n_facilities <= 50_000
        return cls(
            instance=instance,
            household_facility_matrix=None,
            base_gain=_initial_gain(instance),
            facility_demand=(
                tuple(
                    instance.ji_indices[
                        int(instance.ji_indptr[facility]) : int(instance.ji_indptr[facility + 1])
                    ]
                    for facility in range(instance.n_facilities)
                )
                if cache_facility_demand
                else None
            ),
        )

    def demand_of(self, facility: int) -> np.ndarray:
        facility_i = int(facility)
        if self.facility_demand is not None:
            return self.facility_demand[facility_i]
        start = int(self.instance.ji_indptr[facility_i])
        end = int(self.instance.ji_indptr[facility_i + 1])
        return self.instance.ji_indices[start:end]

    def collect_candidates(
        self,
        newly_uncovered: np.ndarray,
        *,
        open_mask: np.ndarray,
        loss: int,
    ) -> np.ndarray:
        if newly_uncovered.size == 0:
            return np.empty(0, dtype=np.int32)
        indptr = self.instance.ij_indptr
        starts = indptr[newly_uncovered]
        lengths = indptr[newly_uncovered + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, dtype=np.int32)

        output_offsets = np.cumsum(lengths, dtype=np.int64) - lengths
        source_positions = (
            np.arange(total, dtype=np.int64)
            + np.repeat(starts.astype(np.int64) - output_offsets, lengths)
        )
        candidates = np.unique(self.instance.ij_indices[source_positions])
        if candidates.size == 0:
            return candidates
        candidates = candidates[(~open_mask[candidates]) & (self.base_gain[candidates] > loss)]
        if candidates.size == 0:
            return candidates
        return candidates[np.argsort(-self.base_gain[candidates])]

    def improve(
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

                candidates = self.collect_candidates(
                    newly_uncovered,
                    open_mask=open_mask,
                    loss=loss,
                )

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


def improve_local_search(
    instance: MaxCoverInstance,
    result: HeuristicResult,
    *,
    local_search: LocalSearchName = "first_sparse",
    sparse_local_search: SparseSwapLocalSearch | None = None,
    max_moves: int | None = None,
    time_limit_seconds: float | None = None,
) -> HeuristicResult:
    if local_search == "none":
        return HeuristicResult(
            solution=list(result.solution),
            objective=int(result.objective),
            coverage=result.coverage.copy(),
            objectives=[int(result.objective)],
            times=[0.0],
            total_time=0.0,
        )
    if local_search == "first_sparse":
        if sparse_local_search is None:
            sparse_local_search = SparseSwapLocalSearch.from_instance(instance)
        return sparse_local_search.improve(
            solution=result.solution,
            coverage=result.coverage,
            objective=result.objective,
            max_moves=max_moves,
            time_limit_seconds=time_limit_seconds,
        )
    if local_search == "first":
        return swap_first_improving(
            instance,
            result.solution,
            coverage=result.coverage,
            objective=result.objective,
            max_moves=max_moves,
            time_limit_seconds=time_limit_seconds,
        )
    raise ValueError(f"unsupported local search: {local_search}")


def greedy_then_local_search(
    instance: MaxCoverInstance,
    budget: int,
    *,
    constructor: ConstructorName = "greedy",
    local_search: LocalSearchName = "first_sparse",
    rcl_size: int = 25,
    sample_size: int = 250,
    random_plus_fraction: float = 0.15,
    seed: int | None = None,
    sparse_local_search: SparseSwapLocalSearch | None = None,
) -> tuple[HeuristicResult, HeuristicResult, HeuristicResult]:
    constructed = budgeted_construct(
        instance,
        budget,
        constructor=constructor,
        rcl_size=rcl_size,
        sample_size=sample_size,
        random_plus_fraction=random_plus_fraction,
        seed=seed,
    )
    improved = improve_local_search(
        instance,
        constructed,
        local_search=local_search,
        sparse_local_search=sparse_local_search,
    )
    reduced = drop_redundant_facilities(
        instance,
        improved.solution,
        coverage=improved.coverage,
        objective=improved.objective,
    )
    return constructed, improved, reduced


def _swap_delta(instance: MaxCoverInstance, coverage: np.ndarray, swap_out: int, swap_in: int) -> int:
    out_cover = instance.demand_of(int(swap_out))
    if out_cover.size:
        newly_uncovered = out_cover[coverage[out_cover] == 1]
        loss = int(instance.weights[newly_uncovered].sum()) if newly_uncovered.size else 0
    else:
        newly_uncovered = np.empty(0, dtype=np.int32)
        loss = 0

    in_cover = instance.demand_of(int(swap_in))
    if not in_cover.size:
        return -loss
    gain = int(instance.weights[in_cover[coverage[in_cover] == 0]].sum())
    if newly_uncovered.size:
        recovered = np.intersect1d(in_cover, newly_uncovered, assume_unique=True)
        if recovered.size:
            gain += int(instance.weights[recovered].sum())
    return gain - loss


def _apply_swap(instance: MaxCoverInstance, coverage: np.ndarray, swap_out: int, swap_in: int) -> None:
    out_cover = instance.demand_of(int(swap_out))
    if out_cover.size:
        coverage[out_cover] -= 1
    in_cover = instance.demand_of(int(swap_in))
    if in_cover.size:
        coverage[in_cover] += 1


def path_relink_fast(
    instance: MaxCoverInstance,
    start_result: HeuristicResult,
    guide_solution: list[int],
    *,
    max_steps: int | None = 64,
    candidate_width: int | None = 16,
    refresh_interval: int = 8,
) -> HeuristicResult:
    """Relink toward a guide through a bounded beam of exact swap deltas.

    Candidate exits and entries are ranked by their current drop loss and add
    gain. Exact pairwise swap deltas are evaluated only inside the leading
    candidate beam, whose ranking is periodically refreshed. The default
    limits avoid the cubic worst-case growth of exhaustive path relinking for
    solutions containing thousands of facilities.

    Beam-level losses, gains, and incidence rows are computed once per path
    step. A reusable demand mask supplies the overlap correction for an exact
    swap delta without allocating and sorting a pairwise intersection.
    """
    if max_steps is not None and int(max_steps) < 0:
        raise ValueError("max_steps must be nonnegative or None")
    if candidate_width is not None and int(candidate_width) <= 0:
        raise ValueError("candidate_width must be positive or None")
    if int(refresh_interval) <= 0:
        raise ValueError("refresh_interval must be positive")
    start_time = perf_counter()
    coverage = start_result.coverage.copy()
    objective = int(start_result.objective)
    guide_set = set(int(x) for x in guide_solution)
    solution_set = set(int(x) for x in start_result.solution)
    to_exit = [facility for facility in start_result.solution if facility not in guide_set]
    to_enter = [facility for facility in guide_solution if int(facility) not in solution_set]
    best_objective = int(objective)
    best_step = 0
    applied_swaps: list[tuple[int, int]] = []
    objectives = [int(objective)]
    times = [0.0]
    recovery_mask = np.zeros(instance.n_demand, dtype=bool)

    available_steps = min(len(to_exit), len(to_enter))
    step_limit = available_steps if max_steps is None else min(available_steps, int(max_steps))
    width_limit = None if candidate_width is None else int(candidate_width)

    for step in range(step_limit):
        if not to_exit or not to_enter:
            break
        if step % int(refresh_interval) == 0:
            to_exit.sort(
                key=lambda facility: drop_delta(instance, coverage, int(facility)),
                reverse=True,
            )
            to_enter.sort(
                key=lambda facility: add_delta(instance, coverage, int(facility)),
                reverse=True,
            )

        exit_candidates = to_exit if width_limit is None else to_exit[:width_limit]
        enter_candidates = to_enter if width_limit is None else to_enter[:width_limit]
        enter_demand = [instance.demand_of(int(facility)) for facility in enter_candidates]
        enter_sizes = np.fromiter(
            (demand.size for demand in enter_demand),
            dtype=np.intp,
            count=len(enter_demand),
        )
        flat_enter_demand = (
            np.concatenate(enter_demand)
            if int(enter_sizes.sum())
            else np.empty(0, dtype=np.int32)
        )
        flat_enter_owner = np.repeat(
            np.arange(len(enter_candidates), dtype=np.intp), enter_sizes
        )
        enter_gain = np.zeros(len(enter_candidates), dtype=np.int64)
        uncovered = coverage[flat_enter_demand] == 0
        np.add.at(
            enter_gain,
            flat_enter_owner[uncovered],
            instance.weights[flat_enter_demand[uncovered]],
        )
        best_pair: tuple[int, int] | None = None
        best_delta: int | None = None
        recovered = np.zeros(len(enter_candidates), dtype=np.int64)
        for swap_out in exit_candidates:
            out_cover = instance.demand_of(int(swap_out))
            newly_uncovered = (
                out_cover[coverage[out_cover] == 1]
                if out_cover.size
                else np.empty(0, dtype=np.int32)
            )
            loss = (
                int(instance.weights[newly_uncovered].sum())
                if newly_uncovered.size
                else 0
            )
            if newly_uncovered.size:
                recovery_mask[newly_uncovered] = True
            recovered.fill(0)
            overlap = recovery_mask[flat_enter_demand]
            np.add.at(
                recovered,
                flat_enter_owner[overlap],
                instance.weights[flat_enter_demand[overlap]],
            )
            deltas = enter_gain - loss + recovered
            enter_index = int(np.argmax(deltas))
            delta = int(deltas[enter_index])
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_pair = (
                    int(swap_out),
                    int(enter_candidates[enter_index]),
                )
            if newly_uncovered.size:
                recovery_mask[newly_uncovered] = False
        if best_pair is None or best_delta is None:
            break
        swap_out, swap_in = best_pair
        _apply_swap(instance, coverage, swap_out, swap_in)
        objective += int(best_delta)
        to_exit.remove(swap_out)
        to_enter.remove(swap_in)
        applied_swaps.append((swap_out, swap_in))
        objectives.append(int(objective))
        times.append(perf_counter() - start_time)
        if objective > best_objective:
            best_objective = int(objective)
            best_step = len(applied_swaps)

    best_solution = list(start_result.solution)
    best_coverage = start_result.coverage.copy()
    for swap_out, swap_in in applied_swaps[:best_step]:
        best_solution.remove(swap_out)
        best_solution.append(swap_in)
        _apply_swap(instance, best_coverage, swap_out, swap_in)

    return HeuristicResult(
        solution=best_solution,
        objective=int(best_objective),
        coverage=best_coverage,
        objectives=objectives,
        times=times,
        total_time=float(perf_counter() - start_time),
    )


def add_delta(instance: MaxCoverInstance, coverage: np.ndarray, facility: int) -> int:
    demand = instance.demand_of(int(facility))
    if not demand.size:
        return 0
    return int(instance.weights[demand[coverage[demand] == 0]].sum())


def drop_delta(instance: MaxCoverInstance, coverage: np.ndarray, facility: int) -> int:
    demand = instance.demand_of(int(facility))
    if not demand.size:
        return 0
    return -int(instance.weights[demand[coverage[demand] == 1]].sum())


def swap_delta(
    instance: MaxCoverInstance,
    coverage: np.ndarray,
    drop_facility: int,
    add_facility: int,
) -> int:
    return _swap_delta(instance, coverage, int(drop_facility), int(add_facility))


def select_by_marginal_gain(
    instance: MaxCoverInstance,
    budget: int,
    *,
    initial_solution: Sequence[int] = (),
    candidate_pool: Sequence[int] | None = None,
) -> HeuristicResult:
    return budgeted_construct(
        instance,
        max(0, int(budget)),
        constructor="greedy",
        initial_solution=initial_solution,
        candidate_pool=candidate_pool,
    )
