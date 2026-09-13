"""Tests for instance construction and validation."""

from __future__ import annotations

import numpy as np
import pytest

import abw_maxcover as mc
from abw_maxcover.instance import _has_duplicate_row_entries
from abw_maxcover.validation import validate_instance


def test_assume_unique_accepts_unsorted_rows_and_matches_default_build() -> None:
    weights = [10, 7, 5, 4, 3]
    ij = [[1, 0], [2, 0], [2, 1], [3, 2], [3]]
    ji = [[1, 0], [2, 0], [3, 1, 2], [4, 3]]
    fast = mc.build_instance(weights, ij, ji, assume_unique=True, validate_consistency=True)
    slow = mc.build_instance(weights, ij, ji, validate_consistency=True)
    for budget in (1, 2, 3):
        assert (
            mc.select_by_marginal_gain(fast, budget).objective
            == mc.select_by_marginal_gain(slow, budget).objective
        )
    assert mc.compute_coverage_and_objective(fast, [0, 2])[1] == 26


def test_assume_unique_with_duplicates_is_caught_by_consistency_validation() -> None:
    weights = [10, 7]
    ij = [[0, 0], [0]]
    ji = [[0, 0, 1]]
    # Without validation the duplicate slips through and inflates the gain.
    unchecked = mc.build_instance(weights, ij, ji, assume_unique=True)
    assert mc.select_by_marginal_gain(unchecked, 1).objective == 27
    assert mc.compute_coverage_and_objective(unchecked, [0])[1] == 17
    # With validation it is an error, on either side of the relation.
    with pytest.raises(ValueError, match="duplicate"):
        mc.build_instance(weights, ij, ji, assume_unique=True, validate_consistency=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_instance(unchecked)
    # The default build deduplicates and is therefore correct.
    clean = mc.build_instance(weights, ij, ji, validate_consistency=True)
    assert mc.select_by_marginal_gain(clean, 1).objective == 17


def test_assume_unique_sorted_alias_is_deprecated_but_equivalent() -> None:
    weights = [10, 7, 5, 4, 3]
    ij = [[0, 1], [0, 2], [1, 2], [2, 3], [3]]
    ji = [[0, 1], [0, 2], [1, 2, 3], [3, 4]]
    with pytest.warns(DeprecationWarning, match="assume_unique"):
        legacy = mc.build_instance(weights, ij, ji, assume_unique_sorted=True)
    current = mc.build_instance(weights, ij, ji, assume_unique=True)
    assert np.array_equal(legacy.ij_indices, current.ij_indices)
    assert np.array_equal(legacy.ji_indices, current.ji_indices)
    with pytest.warns(DeprecationWarning, match="assume_unique"):
        mc.build_instance_from_facility_map({0: [0, 1]}, [1, 2], assume_unique_sorted=True)


def test_negative_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        mc.build_instance([-5, 10], [[0], [0]], [[0, 1]])
    with pytest.raises(ValueError, match="nonnegative"):
        mc.MaxCoverInstance(
            weights=np.array([1, -1]),
            ij_indptr=np.array([0, 1, 2]),
            ij_indices=np.array([0, 0]),
            ji_indptr=np.array([0, 2]),
            ji_indices=np.array([0, 1]),
        )
    # Zero weights remain valid: they simply never contribute coverage.
    instance = mc.build_instance([0, 10], [[0], [0]], [[0, 1]])
    assert mc.compute_coverage_and_objective(instance, [0])[1] == 10


def test_duplicate_row_entry_check_is_per_row() -> None:
    indptr = np.array([0, 2, 4], dtype=np.int32)
    # The same column in two different rows is fine.
    assert not _has_duplicate_row_entries(indptr, np.array([0, 1, 1, 0], dtype=np.int32))
    # The same column twice in one row is not.
    assert _has_duplicate_row_entries(indptr, np.array([0, 0, 1, 2], dtype=np.int32))
    assert not _has_duplicate_row_entries(np.array([0, 0], dtype=np.int32), np.array([], np.int32))
