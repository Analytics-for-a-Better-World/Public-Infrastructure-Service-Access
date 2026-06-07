from __future__ import annotations

import pandas as pd

from scalable_distances.candidates import build_candidate_grid
from scalable_distances.layers import load_point_table, normalize_layers
from scalable_distances.matrix.dense import dense_total_matrix


def test_normalize_layers_accepts_original_aliases() -> None:
    assert normalize_layers(["amenity", "grid", "pop"], default=("amenities",)) == (
        "amenities",
        "candidates",
        "population",
    )


def test_load_point_table_prepares_source_columns(tmp_path) -> None:
    path = tmp_path / "points.csv"
    pd.DataFrame({"id": ["a"], "x": [6.1], "y": [49.6]}).to_csv(path, index=False)

    points = load_point_table(
        path,
        lon_col="x",
        lat_col="y",
        id_col="id",
        role="source",
        bbox=(6.0, 49.5, 6.2, 49.7),
    )

    assert points.loc[0, "source_id"] == "a"
    assert points.loc[0, "source_type"] == "table"
    assert points.loc[0, "lon"] == 6.1
    assert points.loc[0, "lat"] == 49.6


def test_candidate_grid_is_role_aware() -> None:
    candidates = build_candidate_grid((6.0, 49.5, 6.02, 49.52), spacing_m=1000, role="target")

    assert not candidates.empty
    assert {"target_id", "target_type", "lon", "lat"} <= set(candidates.columns)
    assert set(candidates["target_type"]) == {"candidates"}


def test_dense_total_matrix_reindexes_full_contract() -> None:
    sources = pd.DataFrame({"source_id": ["s1", "s2"], "source_type": ["amenities", "table"]})
    targets = pd.DataFrame({"target_id": ["t1", "t2"], "target_type": ["population", "population"]})
    matrix = pd.DataFrame(
        [
            {"source_id": "s1", "target_id": "t1", "total_dist": 5.0},
            {"source_id": "s2", "target_id": "t2", "total_dist": 7.0},
        ]
    )

    dense = dense_total_matrix(matrix, sources=sources, targets=targets)

    assert list(dense.index) == ["t1", "t2"]
    assert list(dense.columns) == ["s1", "s2"]
    assert dense.loc["t1", "s1"] == 5.0
    assert pd.isna(dense.loc["t1", "s2"])
