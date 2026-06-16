"""Vietnam-local sparse-matrix alternative to first-swap local search.

This module deliberately does not modify ``approximated_tradeoff``.  It keeps
the same first-improvement swap semantics, but accelerates the heavy candidate
collection step with a SciPy CSR row-sum over newly uncovered households.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter as pc

import numpy as np
from scipy import sparse

import mc_heuristics as mch


def _deduplicate_solution(solution: list[int]) -> list[int]:
    return list(dict.fromkeys(int(facility) for facility in solution))


@dataclass(slots=True)
class SparseSwapLocalSearch:
    instance: mch.MaxCoverInstance
    household_facility_matrix: sparse.csr_matrix
    base_gain: np.ndarray
    facility_households: tuple[np.ndarray, ...]

    @classmethod
    def from_instance(cls, instance: mch.MaxCoverInstance) -> "SparseSwapLocalSearch":
        data = np.ones(instance.ij_indices.size, dtype=np.uint8)
        household_facility_matrix = sparse.csr_matrix(
            (data, instance.ij_indices, instance.ij_indptr),
            shape=(instance.n_households, instance.n_facilities),
        )
        base_gain = np.zeros(instance.n_facilities, dtype=np.int64)
        counts = instance.ji_indptr[1:] - instance.ji_indptr[:-1]
        nonempty = counts > 0
        if nonempty.any():
            base_gain[nonempty] = np.add.reduceat(
                instance.w[instance.ji_indices],
                instance.ji_indptr[:-1][nonempty],
            )
        return cls(
            instance=instance,
            household_facility_matrix=household_facility_matrix,
            base_gain=base_gain,
            facility_households=tuple(
                instance.ji_indices[int(instance.ji_indptr[facility]) : int(instance.ji_indptr[facility + 1])]
                for facility in range(instance.n_facilities)
            ),
        )

    def collect_candidates(
        self,
        newly_uncovered: np.ndarray,
        *,
        open_mask: np.ndarray,
        loss: int,
    ) -> np.ndarray:
        if newly_uncovered.size == 0:
            return np.empty(0, dtype=np.int32)
        touched_counts = self.household_facility_matrix[newly_uncovered].sum(axis=0)
        candidates = np.asarray(touched_counts).ravel().nonzero()[0].astype(np.int32, copy=False)
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
    ) -> mch.HeuristicResult:
        instance = self.instance
        w = instance.w
        sol = _deduplicate_solution(solution)

        if coverage is None:
            cov, obj = mch.compute_coverage_and_objective(instance, sol)
        else:
            cov = np.asarray(coverage, dtype=np.int32).copy()
            obj = int(w[cov > 0].sum()) if objective is None else int(objective)

        objectives: list[int] = [obj]
        times: list[float] = [0.0]
        covered = cov > 0
        uncovered_weight = np.where(covered, 0, w)
        open_mask = np.zeros(instance.n_facilities, dtype=bool)
        for facility in sol:
            open_mask[int(facility)] = True
        start = pc()

        while True:
            modified = False

            for position, removed in enumerate(sol):
                removed_i = int(removed)
                removed_cover = self.facility_households[removed_i]
                if not removed_cover.size:
                    continue

                newly_uncovered = removed_cover[cov[removed_cover] == 1]
                loss = int(w[newly_uncovered].sum()) if newly_uncovered.size else 0

                cov[removed_cover] -= 1
                if newly_uncovered.size:
                    covered[newly_uncovered] = False
                    uncovered_weight[newly_uncovered] = w[newly_uncovered]

                candidates = self.collect_candidates(
                    newly_uncovered,
                    open_mask=open_mask,
                    loss=loss,
                )

                accepted = False
                for candidate in candidates:
                    candidate_i = int(candidate)
                    candidate_cover = self.facility_households[candidate_i]
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

        return mch.HeuristicResult(
            solution=sol,
            objective=int(obj),
            coverage=cov,
            objectives=objectives,
            times=times,
            total_time=float(pc() - start),
        )
