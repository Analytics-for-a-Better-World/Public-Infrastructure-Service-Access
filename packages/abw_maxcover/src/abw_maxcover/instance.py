"""Canonical maximum-cover instance data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_INT32_DTYPE = np.dtype("int32")


def _as_int_array(values: Any, dtype: np.dtype = _INT32_DTYPE) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _deduplicate_solution(solution: list[int] | tuple[int, ...] | np.ndarray) -> list[int]:
    return list(dict.fromkeys(int(facility) for facility in solution))


def _normalise_row(
    row: np.ndarray | list[int] | tuple[int, ...],
    *,
    assume_unique_sorted: bool,
) -> np.ndarray:
    arr = np.asarray(row, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("each row must be one-dimensional")
    if arr.size and int(arr.min()) < 0:
        raise ValueError("row values must be nonnegative")
    if assume_unique_sorted or arr.size <= 1:
        return arr
    return np.unique(arr)


def _rows_to_arrays(
    rows: dict[int, np.ndarray | list[int]] | list[np.ndarray | list[int]],
    n_rows: int,
    *,
    assume_unique_sorted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 0:
        raise ValueError("n_rows must be nonnegative")

    indptr = np.empty(n_rows + 1, dtype=np.int32)
    indptr[0] = 0
    normalised: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(n_rows)]

    if isinstance(rows, dict):
        invalid_rows = [row for row in rows if int(row) < 0 or int(row) >= n_rows]
        if invalid_rows:
            raise ValueError("row mapping contains invalid row ids")
        for row in range(n_rows):
            arr = _normalise_row(
                rows.get(row, []),
                assume_unique_sorted=assume_unique_sorted,
            )
            normalised[row] = arr
            indptr[row + 1] = indptr[row] + arr.size
    else:
        if len(rows) != n_rows:
            raise ValueError("row mapping length must match n_rows")
        for row, values in enumerate(rows):
            arr = _normalise_row(values, assume_unique_sorted=assume_unique_sorted)
            normalised[row] = arr
            indptr[row + 1] = indptr[row] + arr.size

    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for row, arr in enumerate(normalised):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        if end > start:
            indices[start:end] = arr
    return indptr, indices


@dataclass(slots=True)
class MaxCoverInstance:
    """Weighted maximum-cover instance in CSR biadjacency form.

    Demand nodes are indexed ``0..n_demand-1`` and candidate facilities are
    indexed ``0..n_facilities-1``.  ``ij_*`` stores demand-to-facility arcs;
    ``ji_*`` stores facility-to-demand arcs.
    """

    weights: np.ndarray
    ij_indptr: np.ndarray
    ij_indices: np.ndarray
    ji_indptr: np.ndarray
    ji_indices: np.ndarray
    name: str = "max_cover"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = _as_int_array(self.weights, np.dtype("int64"))
        self.ij_indptr = _as_int_array(self.ij_indptr)
        self.ij_indices = _as_int_array(self.ij_indices)
        self.ji_indptr = _as_int_array(self.ji_indptr)
        self.ji_indices = _as_int_array(self.ji_indices)

        if self.weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if self.ij_indptr.ndim != 1 or self.ji_indptr.ndim != 1:
            raise ValueError("indptr arrays must be one-dimensional")
        if self.ij_indices.ndim != 1 or self.ji_indices.ndim != 1:
            raise ValueError("indices arrays must be one-dimensional")
        if self.ij_indptr.size != self.weights.size + 1:
            raise ValueError("ij_indptr must have length n_demand + 1")
        if self.ji_indptr.size < 1:
            raise ValueError("ji_indptr must be nonempty")
        if int(self.ij_indptr[0]) != 0 or int(self.ji_indptr[0]) != 0:
            raise ValueError("indptr arrays must start at zero")
        if np.any(self.ij_indptr[1:] < self.ij_indptr[:-1]):
            raise ValueError("ij_indptr must be nondecreasing")
        if np.any(self.ji_indptr[1:] < self.ji_indptr[:-1]):
            raise ValueError("ji_indptr must be nondecreasing")
        if int(self.ij_indptr[-1]) != self.ij_indices.size:
            raise ValueError("ij_indptr[-1] must equal ij_indices.size")
        if int(self.ji_indptr[-1]) != self.ji_indices.size:
            raise ValueError("ji_indptr[-1] must equal ji_indices.size")
        if self.ij_indices.size:
            if int(self.ij_indices.min()) < 0:
                raise ValueError("ij_indices must be nonnegative")
            if int(self.ij_indices.max()) >= self.n_facilities:
                raise ValueError("ij_indices contain invalid facility ids")
        if self.ji_indices.size:
            if int(self.ji_indices.min()) < 0:
                raise ValueError("ji_indices must be nonnegative")
            if int(self.ji_indices.max()) >= self.n_demand:
                raise ValueError("ji_indices contain invalid demand ids")

    @property
    def w(self) -> np.ndarray:
        """Compatibility alias used by the approximated-tradeoff code."""
        return self.weights

    @property
    def n_demand(self) -> int:
        return int(self.weights.size)

    @property
    def n_households(self) -> int:
        return self.n_demand

    @property
    def n_facilities(self) -> int:
        return int(self.ji_indptr.size - 1)

    @property
    def total_weight(self) -> int:
        return int(self.weights.sum())

    def facilities_of(self, demand: int) -> np.ndarray:
        if demand < 0 or demand >= self.n_demand:
            raise IndexError("demand index out of range")
        start = int(self.ij_indptr[demand])
        end = int(self.ij_indptr[demand + 1])
        return self.ij_indices[start:end]

    def households_of(self, facility: int) -> np.ndarray:
        return self.demand_of(facility)

    def demand_of(self, facility: int) -> np.ndarray:
        if facility < 0 or facility >= self.n_facilities:
            raise IndexError("facility index out of range")
        start = int(self.ji_indptr[facility])
        end = int(self.ji_indptr[facility + 1])
        return self.ji_indices[start:end]

    def demand_with_candidates(self) -> np.ndarray:
        return np.flatnonzero(self.ij_indptr[1:] > self.ij_indptr[:-1]).astype(np.int32)


def build_instance(
    weights: np.ndarray | list[int],
    ij: dict[int, np.ndarray | list[int]] | list[np.ndarray | list[int]],
    ji: dict[int, np.ndarray | list[int]] | list[np.ndarray | list[int]],
    *,
    name: str = "max_cover",
    n_facilities: int | None = None,
    assume_unique_sorted: bool = False,
    validate_consistency: bool = False,
    metadata: dict[str, Any] | None = None,
) -> MaxCoverInstance:
    """Build a canonical instance from demand-to-facility and facility-to-demand rows."""
    weights_arr = np.asarray(weights, dtype=np.int64)
    if weights_arr.ndim != 1:
        raise ValueError("weights must be one-dimensional")

    ij_indptr, ij_indices = _rows_to_arrays(
        ij,
        int(weights_arr.size),
        assume_unique_sorted=assume_unique_sorted,
    )

    if isinstance(ji, dict):
        max_key = max((int(key) for key in ji), default=-1)
        n_ji_rows = max_key + 1 if n_facilities is None else int(n_facilities)
    else:
        n_ji_rows = len(ji) if n_facilities is None else int(n_facilities)

    ji_indptr, ji_indices = _rows_to_arrays(
        ji,
        n_ji_rows,
        assume_unique_sorted=assume_unique_sorted,
    )

    instance = MaxCoverInstance(
        weights=weights_arr,
        ij_indptr=ij_indptr,
        ij_indices=ij_indices,
        ji_indptr=ji_indptr,
        ji_indices=ji_indices,
        name=name,
        metadata=dict(metadata or {}),
    )
    if validate_consistency:
        _validate_biadjacency_consistency(instance)
    return instance


def build_instance_from_facility_map(
    facility_to_demand: dict[int, np.ndarray | list[int]],
    weights: np.ndarray | list[int],
    *,
    covered: set[int] | np.ndarray | None = None,
    name: str = "max_cover",
    n_facilities: int | None = None,
    assume_unique_sorted: bool = False,
    metadata: dict[str, Any] | None = None,
) -> MaxCoverInstance:
    """Build an instance from a legacy facility-to-demand catchment mapping."""
    weights_arr = np.asarray(weights, dtype=np.int64)
    covered_arr = np.asarray(sorted([] if covered is None else covered), dtype=np.int32)
    n_facilities_final = (
        max((int(key) for key in facility_to_demand), default=-1) + 1
        if n_facilities is None
        else int(n_facilities)
    )
    ji: dict[int, np.ndarray] = {}
    ij_lists: list[list[int]] = [[] for _ in range(weights_arr.size)]
    for facility, demand_values in facility_to_demand.items():
        arr = _normalise_row(demand_values, assume_unique_sorted=assume_unique_sorted)
        if covered_arr.size:
            arr = np.setdiff1d(arr, covered_arr, assume_unique=True)
        if arr.size == 0:
            continue
        facility_i = int(facility)
        ji[facility_i] = arr
        for demand_i in arr:
            ij_lists[int(demand_i)].append(facility_i)
    return build_instance(
        weights_arr,
        ij_lists,
        ji,
        name=name,
        n_facilities=n_facilities_final,
        assume_unique_sorted=False,
        validate_consistency=False,
        metadata=metadata,
    )


def _validate_biadjacency_consistency(instance: MaxCoverInstance) -> None:
    facility_sets = [
        set(int(demand) for demand in instance.demand_of(facility))
        for facility in range(instance.n_facilities)
    ]
    demand_sets = [
        set(int(facility) for facility in instance.facilities_of(demand))
        for demand in range(instance.n_demand)
    ]
    for demand, facilities in enumerate(demand_sets):
        for facility in facilities:
            if demand not in facility_sets[facility]:
                raise ValueError("ij and ji are inconsistent")
    for facility, demand_values in enumerate(facility_sets):
        for demand in demand_values:
            if facility not in demand_sets[demand]:
                raise ValueError("ji and ij are inconsistent")
