from __future__ import annotations

from pathlib import Path

import pandas as pd

from scalable_distances import describe_country_sources, write_distance_matrix
from scalable_distances.matrix import split_matrix_output_key, split_matrix_table


def build_smoke_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source_id": "s1", "target_id": "d1", "source_type": "amenities", "target_type": "population", "total_dist": 10.0},
            {"source_id": "s2", "target_id": "d1", "source_type": "candidates", "target_type": "population", "total_dist": 15.0},
            {"source_id": "s1", "target_id": "t1", "source_type": "amenities", "target_type": "table", "total_dist": 20.0},
        ]
    )


def smoke_test_split_matrix(output_dir: Path) -> dict[str, object]:
    matrix = build_smoke_matrix()
    parts = split_matrix_table(matrix)
    result = write_distance_matrix(
        parts,
        output_dir=output_dir,
        run_tag="smoke",
        mode="both",
    )
    row_counts = {key: len(pd.read_parquet(path)) for key, path in result.paths.items()}
    assert split_matrix_output_key("amenities", "population") in row_counts
    assert row_counts["distance_matrix"] == len(matrix)
    assert sum(count for key, count in row_counts.items() if key != "distance_matrix") == len(matrix)

    sources = describe_country_sources(
        country_slug="luxembourg",
        iso3="LUX",
        base_dir=Path("cache/luxembourg_data"),
        worldpop_dataset="global2",
        worldpop_year=2025,
        worldpop_constrained=True,
    )
    assert sources["worldpop_filename"] == "lux_pop_2025_CN_100m_R2025A_v1.tif"
    return {"paths": result.paths, "row_counts": row_counts}


if __name__ == "__main__":
    smoke_test_split_matrix(Path("diagnostics/split_matrix_smoke"))
