from dataclasses import dataclass
from time import perf_counter as pc

import numpy as np


@dataclass(slots=True)
class MaxCoverInstance:
    '''
    Maximum covering instance in compressed sparse row style.

    Conventions
    -----------
    Households are indexed as 0 .. n_households - 1.
    Facilities are indexed as 0 .. n_facilities - 1.

    Household to facilities adjacency is stored as:
        ij_indptr, ij_indices

    Facility to households adjacency is stored as:
        ji_indptr, ji_indices
    '''

    w: np.ndarray
    ij_indptr: np.ndarray
    ij_indices: np.ndarray
    ji_indptr: np.ndarray
    ji_indices: np.ndarray

    def __post_init__(self) -> None:
        '''
        Validate shapes, ranges, and normalise dtypes.
        '''
        self.w = np.asarray(self.w, dtype=np.int64)
        self.ij_indptr = np.asarray(self.ij_indptr, dtype=np.int32)
        self.ij_indices = np.asarray(self.ij_indices, dtype=np.int32)
        self.ji_indptr = np.asarray(self.ji_indptr, dtype=np.int32)
        self.ji_indices = np.asarray(self.ji_indices, dtype=np.int32)

        if self.w.ndim != 1:
            raise ValueError('w must be 1D')

        if self.ij_indptr.ndim != 1 or self.ji_indptr.ndim != 1:
            raise ValueError('indptr arrays must be 1D')

        if self.ij_indices.ndim != 1 or self.ji_indices.ndim != 1:
            raise ValueError('indices arrays must be 1D')

        if self.ij_indptr.size != self.w.size + 1:
            raise ValueError('ij_indptr must have length n_households + 1')

        if self.ji_indptr.size < 1:
            raise ValueError('ji_indptr must be nonempty')

        if self.ij_indptr[0] != 0 or self.ji_indptr[0] != 0:
            raise ValueError('indptr arrays must start at 0')

        if np.any(self.ij_indptr[1:] < self.ij_indptr[:-1]):
            raise ValueError('ij_indptr must be nondecreasing')

        if np.any(self.ji_indptr[1:] < self.ji_indptr[:-1]):
            raise ValueError('ji_indptr must be nondecreasing')

        if self.ij_indptr[-1] != self.ij_indices.size:
            raise ValueError('ij_indptr[-1] must equal ij_indices.size')

        if self.ji_indptr[-1] != self.ji_indices.size:
            raise ValueError('ji_indptr[-1] must equal ji_indices.size')

        if self.ij_indices.size:
            if self.ij_indices.min() < 0:
                raise ValueError('ij_indices must be nonnegative')
            if self.ij_indices.max() >= self.n_facilities:
                raise ValueError('ij_indices contain invalid facility ids')

        if self.ji_indices.size:
            if self.ji_indices.min() < 0:
                raise ValueError('ji_indices must be nonnegative')
            if self.ji_indices.max() >= self.n_households:
                raise ValueError('ji_indices contain invalid household ids')

    @property
    def n_households(self) -> int:
        '''
        Return the number of households.
        '''
        return int(self.w.size)

    @property
    def n_facilities(self) -> int:
        '''
        Return the number of facilities.
        '''
        return int(self.ji_indptr.size - 1)

    def facilities_of(self, household: int) -> np.ndarray:
        '''
        Return the facilities covering one household.
        '''
        if household < 0 or household >= self.n_households:
            raise IndexError('household index out of range')

        start = int(self.ij_indptr[household])
        end = int(self.ij_indptr[household + 1])
        return self.ij_indices[start:end]

    def households_of(self, facility: int) -> np.ndarray:
        '''
        Return the households covered by one facility.
        '''
        if facility < 0 or facility >= self.n_facilities:
            raise IndexError('facility index out of range')

        start = int(self.ji_indptr[facility])
        end = int(self.ji_indptr[facility + 1])
        return self.ji_indices[start:end]


def _normalise_row(
    row: np.ndarray | list[int],
    *,
    assume_unique_sorted: bool,
) -> np.ndarray:
    '''
    Convert one row to a 1D int32 array.

    Parameters
    ----------
    row
        Row values.
    assume_unique_sorted
        If True, skip deduplication and sorting.

    Returns
    -------
    np.ndarray
        Normalised row as int32.
    '''
    arr = np.asarray(row, dtype=np.int32)

    if arr.ndim != 1:
        raise ValueError('each row must be 1D')

    if arr.size and np.any(arr < 0):
        raise ValueError('row values must be nonnegative')

    if assume_unique_sorted or arr.size <= 1:
        return arr

    return np.unique(arr)


def _rows_to_arrays(
    rows: dict[int, np.ndarray | list[int]] | list[np.ndarray | list[int]],
    n_rows: int,
    *,
    assume_unique_sorted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Convert a row mapping into indptr and indices arrays.

    Parameters
    ----------
    rows
        Row to columns mapping.
    n_rows
        Number of rows.
    assume_unique_sorted
        If True, trust rows to already be sorted and duplicate free.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pair of indptr and indices arrays.
    '''
    if n_rows < 0:
        raise ValueError('n_rows must be nonnegative')

    indptr = np.empty(n_rows + 1, dtype=np.int32)
    indptr[0] = 0

    normalised_rows: list[np.ndarray] = [
        np.empty(0, dtype=np.int32) for _ in range(n_rows)
    ]

    if isinstance(rows, dict):
        invalid_rows = [row for row in rows if row < 0 or row >= n_rows]
        if invalid_rows:
            raise ValueError('row mapping contains invalid row ids')

        get_row = rows.get
        for row in range(n_rows):
            arr = _normalise_row(
                get_row(row, []),
                assume_unique_sorted=assume_unique_sorted,
            )
            normalised_rows[row] = arr
            indptr[row + 1] = indptr[row] + arr.size
    else:
        if len(rows) != n_rows:
            raise ValueError('row mapping length must match n_rows')

        for row, values in enumerate(rows):
            arr = _normalise_row(
                values,
                assume_unique_sorted=assume_unique_sorted,
            )
            normalised_rows[row] = arr
            indptr[row + 1] = indptr[row] + arr.size

    indices = np.empty(int(indptr[-1]), dtype=np.int32)

    for row, arr in enumerate(normalised_rows):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        if end > start:
            indices[start:end] = arr

    return indptr, indices


def _validate_cross_ranges(
    *,
    n_households: int,
    n_facilities: int,
    ij_indices: np.ndarray,
    ji_indices: np.ndarray,
) -> None:
    '''
    Validate that adjacency arrays reference valid opposite side ids.
    '''
    if ij_indices.size and ij_indices.max() >= n_facilities:
        raise ValueError('IJ references facility ids outside JI range')

    if ji_indices.size and ji_indices.max() >= n_households:
        raise ValueError('JI references household ids outside weight range')


def _validate_biadjacency_consistency(instance: MaxCoverInstance) -> None:
    '''
    Validate that IJ and JI encode the same bipartite adjacency.

    This is a full consistency check and can be expensive on large instances.
    '''
    facility_sets = [
        set(int(household) for household in instance.households_of(facility))
        for facility in range(instance.n_facilities)
    ]
    household_sets = [
        set(int(facility) for facility in instance.facilities_of(household))
        for household in range(instance.n_households)
    ]

    for household in range(instance.n_households):
        for facility in household_sets[household]:
            if household not in facility_sets[facility]:
                raise ValueError('IJ and JI are inconsistent')

    for facility in range(instance.n_facilities):
        for household in facility_sets[facility]:
            if facility not in household_sets[household]:
                raise ValueError('JI and IJ are inconsistent')


def build_instance(
    w: np.ndarray | list[int],
    IJ: dict[int, np.ndarray | list[int]] | list[np.ndarray | list[int]],
    JI: dict[int, np.ndarray | list[int]] | list[np.ndarray | list[int]],
    *,
    assume_unique_sorted: bool = False,
    validate_consistency: bool = False,
) -> MaxCoverInstance:
    '''
    Build a canonical instance from household to facility and facility to
    household mappings.

    Parameters
    ----------
    w
        Household weights.
    IJ
        Household to facilities mapping.
    JI
        Facility to households mapping.
    assume_unique_sorted
        If True, trust IJ and JI rows to already be sorted and duplicate free.
    validate_consistency
        If True, verify that IJ and JI describe the same bipartite graph.

    Returns
    -------
    MaxCoverInstance
        Canonical positional instance.
    '''
    w_arr = np.asarray(w, dtype=np.int64)
    if w_arr.ndim != 1:
        raise ValueError('w must be 1D')

    n_households = int(w_arr.size)

    ij_indptr, ij_indices = _rows_to_arrays(
        IJ,
        n_households,
        assume_unique_sorted=assume_unique_sorted,
    )

    if isinstance(JI, dict):
        if JI:
            invalid_facilities = [facility for facility in JI if facility < 0]
            if invalid_facilities:
                raise ValueError('JI contains negative facility ids')
        n_facilities = max((int(facility) for facility in JI), default=-1) + 1
    else:
        n_facilities = len(JI)

    ji_indptr, ji_indices = _rows_to_arrays(
        JI,
        n_facilities,
        assume_unique_sorted=assume_unique_sorted,
    )

    _validate_cross_ranges(
        n_households=n_households,
        n_facilities=n_facilities,
        ij_indices=ij_indices,
        ji_indices=ji_indices,
    )

    instance = MaxCoverInstance(
        w=w_arr,
        ij_indptr=ij_indptr,
        ij_indices=ij_indices,
        ji_indptr=ji_indptr,
        ji_indices=ji_indices,
    )

    if validate_consistency:
        _validate_biadjacency_consistency(instance)

    return instance


@dataclass(slots=True)
class RestrictedInstance:
    '''
    Restricted instance together with facility id mappings.

    Attributes
    ----------
    instance
        Restricted instance with facilities reindexed as 0 .. k - 1.
    old_to_new
        Mapping from original facility id to restricted facility id.
    new_to_old
        Array such that new_to_old[j] is the original facility id of
        restricted facility j.
    '''

    instance: MaxCoverInstance
    old_to_new: dict[int, int]
    new_to_old: np.ndarray


@dataclass(slots=True)
class HeuristicResult:
    '''
    Result of a constructive or local search heuristic.
    '''

    solution: list[int]
    objective: int
    coverage: np.ndarray
    objectives: list[int]
    times: list[float]
    total_time: float


def _deduplicate_solution(solution: list[int]) -> list[int]:
    '''
    Return a duplicate free solution preserving first occurrence order.
    '''
    return list(dict.fromkeys(int(facility) for facility in solution))


def compute_coverage_and_objective(
    instance: MaxCoverInstance,
    solution: list[int],
) -> tuple[np.ndarray, int]:
    '''
    Compute coverage counts and objective value for a solution.

    Parameters
    ----------
    instance
        Maximum covering instance.
    solution
        Open facility positions.

    Returns
    -------
    tuple[np.ndarray, int]
        Coverage vector and weighted covered population.
    '''
    sol = _deduplicate_solution(solution)
    coverage = np.zeros(instance.n_households, dtype=np.int32)

    for facility in sol:
        covered = instance.households_of(int(facility))
        if covered.size:
            coverage[covered] += 1

    objective = int(instance.w[coverage > 0].sum())
    return coverage, objective


def restrict_instance(
    instance: MaxCoverInstance,
    solution: list[int],
) -> RestrictedInstance:
    '''
    Restrict an instance to the facilities in a solution.

    The restricted instance preserves the household universe and weights, but
    only keeps the selected facilities. Facilities are reindexed as
    0 .. len(solution) - 1, preserving first occurrence order from solution.

    Parameters
    ----------
    instance
        Original instance.
    solution
        Facility ids in the original instance.

    Returns
    -------
    RestrictedInstance
        Restricted instance and mappings between restricted and original
        facility ids.
    '''
    new_to_old = np.asarray(_deduplicate_solution(solution), dtype=np.int32)
    n_selected = int(new_to_old.size)
    n_households = instance.n_households

    old_to_new = {
        int(old_facility): int(new_facility)
        for new_facility, old_facility in enumerate(new_to_old)
    }

    ji_indptr = np.empty(n_selected + 1, dtype=np.int32)
    ji_indptr[0] = 0
    ji_chunks: list[np.ndarray] = []

    ij_lists: list[list[int]] = [[] for _ in range(n_households)]

    for new_facility, old_facility in enumerate(new_to_old):
        covered = instance.households_of(int(old_facility))
        ji_chunks.append(covered)
        ji_indptr[new_facility + 1] = ji_indptr[new_facility] + covered.size

        for household in covered:
            ij_lists[int(household)].append(int(new_facility))

    ji_indices = (
        np.concatenate(ji_chunks)
        if ji_chunks and ji_indptr[-1] > 0
        else np.empty(0, dtype=np.int32)
    )

    ij_indptr = np.empty(n_households + 1, dtype=np.int32)
    ij_indptr[0] = 0
    ij_chunks: list[np.ndarray] = []

    for household, facilities in enumerate(ij_lists):
        arr = np.asarray(facilities, dtype=np.int32)
        ij_chunks.append(arr)
        ij_indptr[household + 1] = ij_indptr[household] + arr.size

    ij_indices = (
        np.concatenate(ij_chunks)
        if ij_chunks and ij_indptr[-1] > 0
        else np.empty(0, dtype=np.int32)
    )

    restricted_instance = MaxCoverInstance(
        w=instance.w.copy(),
        ij_indptr=ij_indptr,
        ij_indices=ij_indices,
        ji_indptr=ji_indptr,
        ji_indices=ji_indices,
    )

    return RestrictedInstance(
        instance=restricted_instance,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
    )


def lift_solution(
    restricted_solution: list[int],
    new_to_old: np.ndarray,
) -> list[int]:
    '''
    Map a restricted solution back to original facility ids.

    Parameters
    ----------
    restricted_solution
        Facility ids in the restricted instance.
    new_to_old
        Array mapping restricted facility ids to original facility ids.

    Returns
    -------
    list[int]
        Facility ids in the original instance.
    '''
    return [int(new_to_old[int(facility)]) for facility in restricted_solution]


def greedy_construct(instance: MaxCoverInstance) -> HeuristicResult:
    '''
    Greedy construction until no positive marginal gain remains.

    Parameters
    ----------
    instance
        Maximum covering instance.

    Returns
    -------
    HeuristicResult
        Greedy solution and trace information.
    '''
    w = instance.w
    ij_indptr = instance.ij_indptr
    ij_indices = instance.ij_indices
    ji_indptr = instance.ji_indptr
    ji_indices = instance.ji_indices

    uncovered = np.ones(instance.n_households, dtype=bool)
    coverage = np.zeros(instance.n_households, dtype=np.int32)

    gain = np.zeros(instance.n_facilities, dtype=np.int64)
    counts = ji_indptr[1:] - ji_indptr[:-1]
    nonempty = counts > 0
    if nonempty.any():
        gain[nonempty] = np.add.reduceat(w[ji_indices], ji_indptr[:-1][nonempty])

    solution: list[int] = []
    objective = 0
    objectives: list[int] = [0]
    times: list[float] = [0.0]

    start = pc()

    while True:
        best_facility = int(np.argmax(gain))
        best_gain = int(gain[best_facility])

        if best_gain <= 0:
            break

        solution.append(best_facility)
        objective += best_gain
        gain[best_facility] = -1

        j0 = int(ji_indptr[best_facility])
        j1 = int(ji_indptr[best_facility + 1])

        if j0 == j1:
            times.append(pc() - start)
            objectives.append(objective)
            continue

        covered = ji_indices[j0:j1]
        newly_covered = covered[uncovered[covered]]
        coverage[covered] += 1

        if newly_covered.size:
            uncovered[newly_covered] = False

            starts = ij_indptr[newly_covered]
            ends = ij_indptr[newly_covered + 1]
            sizes = ends - starts
            total_nnz = int(sizes.sum())

            if total_nnz:
                touched = np.empty(total_nnz, dtype=np.int32)
                weights = np.empty(total_nnz, dtype=np.int64)

                pos = 0
                for household, s, e in zip(newly_covered, starts, ends):
                    m = int(e - s)
                    if m:
                        nxt = pos + m
                        touched[pos:nxt] = ij_indices[s:e]
                        weights[pos:nxt] = w[int(household)]
                        pos = nxt

                mask = gain[touched] >= 0
                if mask.any():
                    np.add.at(gain, touched[mask], -weights[mask])

        times.append(pc() - start)
        objectives.append(objective)

    total_time = float(pc() - start)

    return HeuristicResult(
        solution=solution,
        objective=int(objective),
        coverage=coverage,
        objectives=objectives,
        times=times,
        total_time=total_time,
    )


def swap_first_improving(
    instance: MaxCoverInstance,
    solution: list[int],
    coverage: np.ndarray | None = None,
    objective: int | None = None,
) -> HeuristicResult:
    '''
    First improving local search using only strict 1 for 1 swaps.

    A move removes one open facility and adds one closed facility if and only
    if the weighted covered population strictly increases.

    Parameters
    ----------
    instance
        Maximum covering instance.
    solution
        Current open facility positions.
    coverage
        Coverage counts per household. If None, computed from solution.
    objective
        Current weighted covered population. If None, computed from coverage.

    Returns
    -------
    HeuristicResult
        Improved solution and trace information.
    '''
    w = instance.w
    sol = _deduplicate_solution(solution)

    if coverage is None:
        cov, obj = compute_coverage_and_objective(instance, sol)
    else:
        cov = np.asarray(coverage, dtype=np.int32).copy()
        obj = int(w[cov > 0].sum()) if objective is None else int(objective)

    objectives: list[int] = [obj]
    times: list[float] = [0.0]

    base_gain = np.empty(instance.n_facilities, dtype=np.int64)
    for facility in range(instance.n_facilities):
        covered = instance.households_of(facility)
        base_gain[facility] = int(w[covered].sum()) if covered.size else 0

    covered = cov > 0
    uncovered_weight = np.where(covered, 0, w)

    open_mask = np.zeros(instance.n_facilities, dtype=bool)
    touched_mask = np.zeros(instance.n_facilities, dtype=bool)
    touched_list: list[int] = []

    def refresh_open_mask() -> None:
        '''
        Refresh the open facility marker array.
        '''
        open_mask.fill(False)
        for facility in sol:
            open_mask[int(facility)] = True

    def collect_candidates(newly_uncovered: np.ndarray) -> np.ndarray:
        '''
        Collect closed facilities touching at least one newly uncovered
        household.
        '''
        for household in newly_uncovered:
            for facility in instance.facilities_of(int(household)):
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

    start = pc()

    while True:
        refresh_open_mask()
        modified = False

        for position, removed in enumerate(sol):
            removed_i = int(removed)
            removed_cover = instance.households_of(removed_i)

            if not removed_cover.size:
                continue

            newly_uncovered = removed_cover[cov[removed_cover] == 1]
            loss = int(w[newly_uncovered].sum()) if newly_uncovered.size else 0

            cov[removed_cover] -= 1
            if newly_uncovered.size:
                covered[newly_uncovered] = False
                uncovered_weight[newly_uncovered] = w[newly_uncovered]

            candidates = collect_candidates(newly_uncovered)
            if candidates.size:
                # base_gain is a static upper bound on attainable gain under the
                # current coverage state. It is used only for pruning and
                # ordering, never as the true move value.
                candidates = candidates[base_gain[candidates] > loss]
                if candidates.size:
                    candidates = candidates[np.argsort(-base_gain[candidates])]

            accepted = False

            for candidate in candidates:
                candidate_i = int(candidate)
                candidate_cover = instance.households_of(candidate_i)

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
                times.append(pc() - start)
                objectives.append(obj)
                break

            if accepted:
                break

            cov[removed_cover] += 1
            if newly_uncovered.size:
                covered[newly_uncovered] = True
                uncovered_weight[newly_uncovered] = 0

        if not modified:
            break

    total_time = float(pc() - start)

    return HeuristicResult(
        solution=sol,
        objective=int(obj),
        coverage=cov,
        objectives=objectives,
        times=times,
        total_time=total_time,
    )


def drop_redundant_facilities(
    instance: MaxCoverInstance,
    solution: list[int],
    coverage: np.ndarray | None = None,
    objective: int | None = None,
) -> HeuristicResult:
    '''
    Remove facilities whose removal does not reduce the objective.

    A facility is dropped if it covers no uniquely covered household.

    Parameters
    ----------
    instance
        Maximum covering instance.
    solution
        Current open facility positions.
    coverage
        Coverage counts per household. If None, computed from solution.
    objective
        Current weighted covered population. If None, computed from coverage.

    Returns
    -------
    HeuristicResult
        Pruned solution and trace information.
    '''
    w = instance.w
    sol = _deduplicate_solution(solution)

    if coverage is None:
        cov, obj = compute_coverage_and_objective(instance, sol)
    else:
        cov = np.asarray(coverage, dtype=np.int32).copy()
        obj = int(w[cov > 0].sum()) if objective is None else int(objective)

    objectives: list[int] = [obj]
    times: list[float] = [0.0]

    start = pc()
    i = 0

    while i < len(sol):
        facility = int(sol[i])
        covered = instance.households_of(facility)

        if not covered.size or np.all(cov[covered] >= 2):
            if covered.size:
                cov[covered] -= 1
            sol.pop(i)
            times.append(pc() - start)
            objectives.append(obj)
            continue

        i += 1

    if int(w[cov > 0].sum()) != obj:
        raise RuntimeError('redundant facility removal changed the objective')

    total_time = float(pc() - start)

    return HeuristicResult(
        solution=sol,
        objective=int(obj),
        coverage=cov,
        objectives=objectives,
        times=times,
        total_time=total_time,
    )


def greedy_then_local_search(
    instance: MaxCoverInstance,
    *,
    alternate_until_stable: bool = False,
) -> tuple[HeuristicResult, HeuristicResult, HeuristicResult | None]:
    '''
    Run greedy construction, then swap local search, then drop redundant
    facilities.

    Parameters
    ----------
    instance
        Maximum covering instance.
    alternate_until_stable
        If True, alternate swap and drop until the solution stops changing.
        If False, run greedy once, swap once, and drop once.

    Returns
    -------
    tuple[HeuristicResult, HeuristicResult, HeuristicResult | None]
        Greedy result, final swap result, and final drop result.
    '''
    greedy_result = greedy_construct(instance)

    if not alternate_until_stable:
        swap_result = swap_first_improving(
            instance=instance,
            solution=greedy_result.solution,
            coverage=greedy_result.coverage,
            objective=greedy_result.objective,
        )
        drop_result = drop_redundant_facilities(
            instance=instance,
            solution=swap_result.solution,
            coverage=swap_result.coverage,
            objective=swap_result.objective,
        )
        return greedy_result, swap_result, drop_result

    current_solution = greedy_result.solution
    current_coverage = greedy_result.coverage
    current_objective = greedy_result.objective

    swap_result = HeuristicResult(
        solution=current_solution.copy(),
        objective=int(current_objective),
        coverage=current_coverage.copy(),
        objectives=[int(current_objective)],
        times=[0.0],
        total_time=0.0,
    )
    drop_result: HeuristicResult | None = None

    while True:
        swap_result = swap_first_improving(
            instance=instance,
            solution=current_solution,
            coverage=current_coverage,
            objective=current_objective,
        )
        drop_result = drop_redundant_facilities(
            instance=instance,
            solution=swap_result.solution,
            coverage=swap_result.coverage,
            objective=swap_result.objective,
        )

        if (
            drop_result.objective == current_objective
            and drop_result.solution == current_solution
        ):
            break

        current_solution = drop_result.solution
        current_coverage = drop_result.coverage
        current_objective = drop_result.objective

    return greedy_result, swap_result, drop_result


def greedy_drop_greedy(
    instance: MaxCoverInstance,
) -> tuple[HeuristicResult, HeuristicResult, RestrictedInstance, HeuristicResult]:
    '''
    Run greedy on the original instance, drop redundant facilities, restrict to
    the reduced solution, then run greedy again on the restricted instance.

    The final result is lifted back to original facility ids and evaluated on
    the original instance.

    Parameters
    ----------
    instance
        Maximum covering instance.

    Returns
    -------
    tuple[HeuristicResult, HeuristicResult, RestrictedInstance, HeuristicResult]
        Original greedy result,
        reduced result after dropping redundant facilities,
        restricted instance information,
        greedy result on the restricted instance lifted to the original instance.
    '''
    t = pc()
    greedy_result = greedy_construct(instance)

    reduced_result = drop_redundant_facilities(
        instance=instance,
        solution=greedy_result.solution,
        coverage=greedy_result.coverage,
        objective=greedy_result.objective,
    )

    restricted = restrict_instance(
        instance=instance,
        solution=reduced_result.solution,
    )

    restricted_greedy = greedy_construct(restricted.instance)
    lifted_solution = lift_solution(
        restricted_solution=restricted_greedy.solution,
        new_to_old=restricted.new_to_old,
    )
    lifted_coverage, lifted_objective = compute_coverage_and_objective(
        instance=instance,
        solution=lifted_solution,
    )

    lifted_result = HeuristicResult(
        solution=lifted_solution,
        objective=lifted_objective,
        coverage=lifted_coverage,
        objectives=restricted_greedy.objectives.copy(),
        times=restricted_greedy.times.copy(),
        total_time=float(pc() - t),
    )

    return greedy_result, reduced_result, restricted, lifted_result


def greedy_drop_greedy_solution(instance: MaxCoverInstance) -> HeuristicResult:
    '''
    Run greedy, drop redundant facilities, restrict to the reduced solution,
    run greedy again on the restricted instance, and return the final result in
    original facility ids.

    Parameters
    ----------
    instance
        Maximum covering instance.

    Returns
    -------
    HeuristicResult
        Final greedy result expressed in the original instance.
    '''
    _, _, _, lifted_result = greedy_drop_greedy(instance)
    return lifted_result
    

import pandas as pd

def greedy_result_to_dataframe(result: HeuristicResult) -> pd.DataFrame:
    '''
    Convert a greedy HeuristicResult into a trace DataFrame.

    Parameters
    ----------
    result
        Greedy heuristic result. Its objective trace is assumed to start at 0
        and each accepted move is assumed to append one facility to the
        solution.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns solution, value, and cputime.
    '''
    if len(result.objectives) != len(result.times):
        raise ValueError('result.objectives and result.times must have the same length')

    if len(result.objectives) != len(result.solution) + 1:
        raise ValueError(
            'for greedy traces, len(result.objectives) must equal len(result.solution) + 1'
        )

    solutions: list[tuple[int, ...]] = [()]
    current: tuple[int, ...] = ()

    for facility in result.solution:
        current = current + (int(facility),)
        solutions.append(current)

    return pd.DataFrame(
        {
            'solution': solutions,
            'value': result.objectives,
            'cputime': result.times,
        }
    )

def greedy_result_to_dataframe_recompute(
    instance: MaxCoverInstance,
    result: HeuristicResult,
) -> pd.DataFrame:
    '''
    Convert a greedy HeuristicResult into a trace DataFrame with
    reconstructed coverage values.

    Parameters
    ----------
    instance
        Maximum covering instance.
    result
        Greedy heuristic result.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            solution
            value
            cputime
            covered_households
    '''
    if len(result.objectives) != len(result.times):
        raise ValueError('result.objectives and result.times must have the same length')

    if len(result.objectives) != len(result.solution) + 1:
        raise ValueError(
            'for greedy traces, len(result.objectives) must equal len(result.solution) + 1'
        )

    w = instance.w
    coverage = np.zeros(instance.n_households, dtype=np.int32)

    solutions: list[tuple[int, ...]] = [()]
    values: list[int] = [0]
    covered_counts: list[int] = [0]

    current_solution: list[int] = []

    for facility in result.solution:

        facility = int(facility)
        current_solution.append(facility)

        households = instance.households_of(facility)

        if households.size:
            coverage[households] += 1

        covered_mask = coverage > 0
        value = int(w[covered_mask].sum())
        covered = int(covered_mask.sum())

        solutions.append(tuple(current_solution))
        values.append(value)
        covered_counts.append(covered)

    return pd.DataFrame(
        {
            'solution': solutions,
            'value': values,
            'covered_households': covered_counts,
            'cputime': result.times,
        }
    )