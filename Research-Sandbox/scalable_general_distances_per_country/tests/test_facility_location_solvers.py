from __future__ import annotations

import pandas as pd

from scalable_distances.optimization import FacilityLocationConfig, solve_facility_location_by_island


def build_facility_location_smoke_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = pd.DataFrame(
        [
            {
                "demand_id": 1,
                "source_id": "school_west",
                "source_name": "West school",
                "island": "Tiny",
                "regency": "Tiny",
                "students": 100.0,
                "demand_weight": 100.0,
                "demand_lat": 0.0,
                "demand_lon": 0.0,
            },
            {
                "demand_id": 2,
                "source_id": "school_east",
                "source_name": "East school",
                "island": "Tiny",
                "regency": "Tiny",
                "students": 100.0,
                "demand_weight": 100.0,
                "demand_lat": 0.0,
                "demand_lon": 1.0,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            {"candidate_id": "facility_west", "island": "Tiny", "facility_lat": 0.0, "facility_lon": 0.0},
            {"candidate_id": "facility_east", "island": "Tiny", "facility_lat": 0.0, "facility_lon": 1.0},
        ]
    )
    return demand, candidates


def smoke_test_pyomo_highs_facility_location() -> dict[str, object]:
    demand, candidates = build_facility_location_smoke_data()
    result = solve_facility_location_by_island(
        "Tiny",
        demand,
        candidates,
        FacilityLocationConfig(setup_km=1.0, solver="pyomo-highs"),
    )
    selected = sorted(result.selected["candidate_id"].tolist())
    assert selected == ["facility_east", "facility_west"]
    assert len(result.assignments) == len(demand)
    assert result.solver == "pyomo-highs"
    return {"solver": result.solver, "selected": selected, "objective": result.objective_value}


def test_pyomo_highs_facility_location() -> None:
    smoke_test_pyomo_highs_facility_location()


if __name__ == "__main__":
    print(smoke_test_pyomo_highs_facility_location())
