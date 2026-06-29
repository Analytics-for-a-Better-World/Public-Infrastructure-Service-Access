from __future__ import annotations

import datetime as dt
import json
import platform
import sys
from importlib import metadata
from pathlib import Path


PACKAGES = [
    "numpy",
    "pandas",
    "geopandas",
    "shapely",
    "pyproj",
    "rasterio",
    "pyrosm",
    "osmium",
    "pandana",
    "polars",
    "scipy",
    "scikit-learn",
    "gurobipy",
    "matplotlib",
    "networkx",
    "osmnx",
    "fiona",
    "pyarrow",
    "tqdm",
    "openpyxl",
    "pytest",
]


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gurobi_smoke() -> dict[str, object]:
    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("license_smoke")
        model.Params.OutputFlag = 0
        x = model.addVar(vtype=GRB.BINARY, name="x")
        y = model.addVar(vtype=GRB.BINARY, name="y")
        model.addConstr(x + y <= 1)
        model.setObjective(2 * x + y, GRB.MAXIMIZE)
        model.optimize()
        return {
            "gurobi_version": gp.gurobi.version(),
            "status": int(model.Status),
            "objective": float(model.ObjVal),
            "x": float(x.X),
            "y": float(y.X),
        }
    except Exception as exc:  # pragma: no cover - environment probe.
        return {"error": repr(exc)}


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/system")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "captured_at": dt.datetime.now().astimezone().isoformat(),
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "gurobi_smoke": gurobi_smoke(),
    }
    path = output_dir / "python_environment.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
