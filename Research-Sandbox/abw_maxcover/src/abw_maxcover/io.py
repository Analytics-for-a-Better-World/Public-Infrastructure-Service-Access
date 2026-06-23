"""Persistence and primitive-record export helpers."""

from __future__ import annotations

import csv
import json
from os import PathLike
from typing import Any, Mapping

import numpy as np

from .instance import MaxCoverInstance
from .results import CurveComparison, MaxCoverCurve


def save_instance_npz(instance: MaxCoverInstance, path: str | PathLike[str]) -> None:
    np.savez_compressed(
        path,
        weights=instance.weights,
        ij_indptr=instance.ij_indptr,
        ij_indices=instance.ij_indices,
        ji_indptr=instance.ji_indptr,
        ji_indices=instance.ji_indices,
        name=np.asarray(instance.name),
        metadata=np.asarray(json.dumps(instance.metadata)),
    )


def load_instance_npz(path: str | PathLike[str]) -> MaxCoverInstance:
    data = np.load(path, allow_pickle=False)
    metadata_raw = str(data["metadata"]) if "metadata" in data else "{}"
    name = str(data["name"]) if "name" in data else "max_cover"
    return MaxCoverInstance(
        weights=data["weights"],
        ij_indptr=data["ij_indptr"],
        ij_indices=data["ij_indices"],
        ji_indptr=data["ji_indptr"],
        ji_indices=data["ji_indices"],
        name=name,
        metadata=json.loads(metadata_raw),
    )


def curve_to_records(curve: MaxCoverCurve) -> list[dict[str, Any]]:
    total_weight = curve.metadata.get("total_weight")
    return curve.to_records(total_weight=total_weight)


def comparison_to_records(comparison: CurveComparison) -> list[dict[str, Any]]:
    return comparison.to_records()


def _write_records(path: str | PathLike[str], records: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for record in records for key in record})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_curve_csv(curve: MaxCoverCurve, path: str | PathLike[str]) -> None:
    _write_records(path, curve_to_records(curve))


def write_comparison_csv(comparison: CurveComparison, path: str | PathLike[str]) -> None:
    _write_records(path, comparison_to_records(comparison))


def write_solution_csv(curve: MaxCoverCurve, path: str | PathLike[str]) -> None:
    records = [
        {"budget": result.budget, "method": result.method, "facility": facility}
        for result in curve.results
        for facility in result.solution
    ]
    _write_records(path, records)


def write_manifest(
    path: str | PathLike[str],
    *,
    instance: MaxCoverInstance,
    curves: Mapping[str, MaxCoverCurve],
    extra: Mapping[str, Any] | None = None,
) -> None:
    manifest = {
        "instance": instance.name,
        "n_demand": instance.n_demand,
        "n_facilities": instance.n_facilities,
        "curves": {name: curve.metadata for name, curve in curves.items()},
        "extra": dict(extra or {}),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
