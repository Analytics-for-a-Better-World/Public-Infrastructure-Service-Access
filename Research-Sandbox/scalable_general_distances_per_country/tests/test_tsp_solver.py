from __future__ import annotations

import pandas as pd

from scalable_distances.optimization import TspConfig, build_tsp_distance_table, solve_tsp_from_distance_table
from scalable_distances.routing.base import NetworkData
from scalable_distances.routing.strategies import NetworkXRouter


def build_tsp_smoke_distances() -> pd.DataFrame:
    rows = []
    costs = {
        ("facility", "school_a"): 10.0,
        ("school_a", "school_b"): 5.0,
        ("school_b", "facility"): 10.0,
        ("facility", "school_b"): 100.0,
        ("school_b", "school_a"): 5.0,
        ("school_a", "facility"): 100.0,
    }
    for (from_id, to_id), distance_m in costs.items():
        rows.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "distance_m": distance_m,
                "distance_source": "road",
            }
        )
    for stop_id in ["facility", "school_a", "school_b"]:
        rows.append(
            {
                "from_id": stop_id,
                "to_id": stop_id,
                "distance_m": 0.0,
                "distance_source": "identity",
            }
        )
    return pd.DataFrame(rows)


def test_pyomo_highs_tsp_uses_school_to_school_leg() -> None:
    result = solve_tsp_from_distance_table(
        build_tsp_smoke_distances(),
        depot_id="facility",
        config=TspConfig(solver="pyomo-highs"),
    )
    assert result.tour == ["facility", "school_a", "school_b", "facility"]
    assert result.objective_distance == 25.0


def test_tsp_distance_builder_includes_facility_and_school_pairs() -> None:
    router = NetworkXRouter()
    router.prepare(
        NetworkData(
            nodes=pd.DataFrame(
                [
                    {"node_id": 1, "lon": 0.0, "lat": 0.0},
                    {"node_id": 2, "lon": 1.0, "lat": 0.0},
                    {"node_id": 3, "lon": 2.0, "lat": 0.0},
                ]
            ),
            edges=pd.DataFrame(
                [
                    {"u": 1, "v": 2, "length_m": 10.0},
                    {"u": 2, "v": 1, "length_m": 10.0},
                    {"u": 2, "v": 3, "length_m": 15.0},
                    {"u": 3, "v": 2, "length_m": 15.0},
                ]
            ),
        ),
        {},
    )
    stops = router.snap(
        pd.DataFrame(
            [
                {"stop_id": "facility", "stop_type": "facility", "lon": 0.0, "lat": 0.0},
                {"stop_id": "school_a", "stop_type": "school", "lon": 1.0, "lat": 0.0},
                {"stop_id": "school_b", "stop_type": "school", "lon": 2.0, "lat": 0.0},
            ]
        )
    )
    distances = build_tsp_distance_table(router, stops)
    pairs = set(zip(distances["from_id"], distances["to_id"]))
    assert ("facility", "school_a") in pairs
    assert ("school_a", "school_b") in pairs
    assert distances.loc[
        distances["from_id"].eq("school_a") & distances["to_id"].eq("school_b"),
        "distance_m",
    ].iloc[0] == 15.0
