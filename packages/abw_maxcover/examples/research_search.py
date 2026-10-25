"""Small solver-free example; replace incidence and geography with real inputs."""

import numpy as np

from abw_maxcover import ResearchSearchConfig, build_instance_from_facility_map, run_research_search

instance = build_instance_from_facility_map(
    {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [0, 3], 4: [4]}, [4, 3, 2, 5, 1]
)
result = run_research_search(
    instance,
    2,
    config=ResearchSearchConfig(iterations=5, fake_clock=True, time_limit_seconds=5),
    facility_coordinates=np.array([[52, 4], [52.01, 4], [52, 4.01], [52.01, 4.01], [52.02, 4]]),
    facility_neighbors=[np.arange(5) for _ in range(5)],
)
print(result.method, result.objective, result.solution)
