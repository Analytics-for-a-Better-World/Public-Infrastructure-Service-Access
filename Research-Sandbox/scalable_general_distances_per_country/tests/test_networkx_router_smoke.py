from __future__ import annotations

import pandas as pd

from scalable_distances.routing.base import NetworkData
from scalable_distances.routing.strategies import NetworkXRouter


def smoke_test_networkx_router() -> dict[str, object]:
    network = NetworkData(
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
                {"u": 2, "v": 3, "length_m": 15.0},
            ]
        ),
    )
    router = NetworkXRouter()
    router.prepare(network, {})
    sources = router.snap(
        pd.DataFrame(
            [{"source_id": "s1", "source_type": "amenities", "lon": 0.0, "lat": 0.0}]
        )
    )
    targets = router.snap(
        pd.DataFrame(
            [{"target_id": "t1", "target_type": "population", "lon": 2.0, "lat": 0.0}]
        )
    )
    matrix = router.route_many(sources, targets)
    assert len(matrix) == 1
    assert matrix.loc[0, "total_dist"] == 25.0
    return {"rows": len(matrix), "distance": float(matrix.loc[0, "total_dist"])}


if __name__ == "__main__":
    print(smoke_test_networkx_router())
