from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class GeospatialBackend:
    """Installed geospatial backend versions used for reproducibility manifests."""

    geopandas: str | None
    npyosmium: str | None
    numpy: str | None
    pyproj: str | None
    shapely: str | None

    @property
    def has_npyosmium(self) -> bool:
        return self.npyosmium is not None

    def as_manifest(self) -> dict[str, str | bool | None]:
        return {
            "geopandas": self.geopandas,
            "npyosmium": self.npyosmium,
            "numpy": self.numpy,
            "pyproj": self.pyproj,
            "shapely": self.shapely,
            "has_npyosmium": self.has_npyosmium,
        }


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def detect_geospatial_backend() -> GeospatialBackend:
    """Detect optional geospatial dependencies without importing heavy modules."""
    return GeospatialBackend(
        geopandas=_version("geopandas"),
        npyosmium=_version("npyosmium"),
        numpy=_version("numpy"),
        pyproj=_version("pyproj"),
        shapely=_version("shapely"),
    )
