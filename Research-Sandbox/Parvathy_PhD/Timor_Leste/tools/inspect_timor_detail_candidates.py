from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


OUTPUTS_DIR = Path("runs/timor_network_profile_20260623/east-timor_data/outputs")
CACHE_DIR = Path("runs/timor_network_profile_20260623/east-timor_data/cache")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def case_key(manifest: dict[str, Any]) -> str:
    runtime = manifest["parameters"]["runtime_settings"]
    resolved = manifest["parameters"]["resolved"]
    profile = runtime["network_profile"]
    simplified = "simplified" if runtime["simplify_network"] else "unsimplified"
    spacing = int(float(resolved["candidate_grid_spacing_m"]))
    return f"{profile}_{simplified}_{spacing}"


def main() -> None:
    manifests = []
    for path in sorted(OUTPUTS_DIR.glob("run_manifest*.yaml")):
        manifest = read_yaml(path)
        if int(float(manifest["parameters"]["resolved"]["candidate_grid_spacing_m"])) == 1000:
            manifests.append((case_key(manifest), manifest))

    facility_rows = []
    source_by_case = {}
    for case, manifest in manifests:
        path = Path(manifest["outputs"]["existing_sources"]["path"])
        frame = pd.read_parquet(
            path,
            columns=["ID", "Longitude", "Latitude", "name", "amenity", "dist_snap_source", "component_id"],
        )
        frame["case"] = case
        source_by_case[case] = frame
        mask = frame["name"].fillna("").astype(str).str.contains(
            "sibuni|sibun|centro|saude|saúde|clinica|clinic|hospital",
            case=False,
            regex=True,
        )
        facility_rows.extend(frame.loc[mask].to_dict(orient="records"))

    merged = None
    for case, frame in source_by_case.items():
        part = frame[["ID", "Longitude", "Latitude", "name", "amenity", "dist_snap_source", "component_id"]].copy()
        part = part.rename(
            columns={
                "dist_snap_source": f"snap_{case}",
                "component_id": f"component_{case}",
            }
        )
        if merged is None:
            merged = part
        else:
            merged = merged.merge(
                part[["ID", f"snap_{case}", f"component_{case}"]],
                on="ID",
                how="outer",
            )

    discrepancy_rows = []
    if merged is not None:
        for lhs, rhs in [
            ("driving_simplified_1000", "driving_unsimplified_1000"),
            ("driving_walk_simplified_1000", "driving_walk_unsimplified_1000"),
            ("driving_simplified_1000", "driving_walk_unsimplified_1000"),
        ]:
            lhs_col = f"snap_{lhs}"
            rhs_col = f"snap_{rhs}"
            if lhs_col in merged and rhs_col in merged:
                temp = merged.copy()
                temp["snap_improvement_m"] = pd.to_numeric(temp[lhs_col], errors="coerce") - pd.to_numeric(
                    temp[rhs_col],
                    errors="coerce",
                )
                temp["comparison"] = f"{lhs}_minus_{rhs}"
                discrepancy_rows.extend(
                    temp.sort_values("snap_improvement_m", ascending=False)
                    .head(20)
                    .to_dict(orient="records")
                )

    schema = {}
    for suffix in [
        "backend_osmium",
        "backend_osmium_simplified_v2",
        "backend_osmium_driving_walk",
        "backend_osmium_driving_walk_simplified",
    ]:
        node_path = CACHE_DIR / f"east-timor-latest.osm_nodes_{suffix}.pkl"
        edge_path = CACHE_DIR / f"east-timor-latest.osm_edges_{suffix}.pkl"
        if node_path.exists() and edge_path.exists():
            nodes = pd.read_pickle(node_path)
            edges = pd.read_pickle(edge_path)
            schema[suffix] = {
                "nodes_rows": int(len(nodes)),
                "nodes_columns": list(nodes.columns),
                "nodes_crs": str(getattr(nodes, "crs", None)),
                "edges_rows": int(len(edges)),
                "edges_columns": list(edges.columns),
                "edges_crs": str(getattr(edges, "crs", None)),
            }

    print(
        json.dumps(
            {
                "facility_matches": facility_rows[:80],
                "facility_match_count": len(facility_rows),
                "largest_snap_discrepancies": discrepancy_rows[:80],
                "schemas": schema,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
