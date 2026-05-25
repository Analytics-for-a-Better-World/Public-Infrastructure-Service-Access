"""Golden mini data sketch for future contract tests.

The real tests should build:
- one tiny road network,
- three demand points,
- two facilities,
- two school records to geocode,
- one duplicated OSM node/way school where the node is preferred for routing.
"""

DEMAND = [
    {"origin_id": "d1", "lat": -8.55, "lon": 125.56},
    {"origin_id": "d2", "lat": -8.56, "lon": 125.57},
    {"origin_id": "d3", "lat": -8.57, "lon": 125.58},
]

FACILITIES = [
    {"facility_id": "s1", "lat": -8.55, "lon": 125.55, "facility_type": "school"},
    {"facility_id": "s2", "lat": -8.58, "lon": 125.59, "facility_type": "school"},
]
