from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CountryDataSources:
    """Source-data naming rules shared with the original country configs."""

    country_slug: str
    iso3: str
    base_dir: Path
    geofabrik_region: str = "europe"
    worldpop_year: int = 2020
    worldpop_dataset: str = "global1"
    worldpop_release: str | None = None
    worldpop_version: str = "v1"
    worldpop_resolution: str = "100m"
    worldpop_constrained: bool = False
    worldpop_suffix: str = "ppp"
    worldpop_adjustment: str | None = "UNadj"
    worldpop_filename: str | None = None
    worldpop_url: str | None = None
    worldpop_path: Path | None = None
    pbf_filename: str | None = None

    def __post_init__(self) -> None:
        if self.worldpop_dataset not in {"global1", "global2"}:
            raise ValueError("worldpop_dataset must be 'global1' or 'global2'.")

    @property
    def resolved_pbf_filename(self) -> str:
        if self.pbf_filename is not None:
            return self.pbf_filename
        return f"{self.country_slug}-latest.osm.pbf"

    @property
    def resolved_worldpop_filename(self) -> str:
        if self.worldpop_filename is not None:
            return self.worldpop_filename

        if self.worldpop_dataset == "global2":
            constrained_part = "CN" if self.worldpop_constrained else "UA"
            release_part = self.worldpop_release or "R2025A"
            return (
                f"{self.iso3.lower()}_pop_{self.worldpop_year}_"
                f"{constrained_part}_{self.worldpop_resolution}_"
                f"{release_part}_{self.worldpop_version}.tif"
            )

        parts = [self.iso3.lower(), self.worldpop_suffix, str(self.worldpop_year)]
        if self.worldpop_adjustment:
            parts.append(self.worldpop_adjustment)
        return "_".join(parts) + ".tif"

    @property
    def pbf_url(self) -> str:
        return (
            "https://download.geofabrik.de/"
            f"{self.geofabrik_region}/{self.resolved_pbf_filename}"
        )

    @property
    def worldpop_download_url(self) -> str:
        if self.worldpop_url is not None:
            return self.worldpop_url

        if self.worldpop_dataset == "global2":
            release_part = self.worldpop_release or "R2025A"
            constraint_part = "constrained" if self.worldpop_constrained else "unconstrained"
            return (
                "https://worldpop-public-data.soton.ac.uk/GIS/Population/"
                f"Global_2015_2030/{release_part}/{self.worldpop_year}/"
                f"{self.iso3}/{self.worldpop_version}/{self.worldpop_resolution}/"
                f"{constraint_part}/{self.resolved_worldpop_filename}"
            )

        return (
            "https://data.worldpop.org/GIS/Population/Global_2000_2020/"
            f"{self.worldpop_year}/{self.iso3}/{self.resolved_worldpop_filename}"
        )

    @property
    def pbf_path(self) -> Path:
        return self.base_dir / self.resolved_pbf_filename

    @property
    def resolved_worldpop_path(self) -> Path:
        if self.worldpop_path is not None:
            return self.worldpop_path
        return self.base_dir / self.resolved_worldpop_filename

    def as_manifest(self) -> dict[str, str | int | bool | None]:
        return {
            "country_slug": self.country_slug,
            "iso3": self.iso3,
            "base_dir": self.base_dir.as_posix(),
            "geofabrik_region": self.geofabrik_region,
            "pbf_filename": self.resolved_pbf_filename,
            "pbf_url": self.pbf_url,
            "pbf_path": self.pbf_path.as_posix(),
            "worldpop_dataset": self.worldpop_dataset,
            "worldpop_year": self.worldpop_year,
            "worldpop_release": self.worldpop_release,
            "worldpop_version": self.worldpop_version,
            "worldpop_resolution": self.worldpop_resolution,
            "worldpop_constrained": self.worldpop_constrained,
            "worldpop_filename": self.resolved_worldpop_filename,
            "worldpop_url": None if self.worldpop_path is not None else self.worldpop_download_url,
            "worldpop_path": self.resolved_worldpop_path.as_posix(),
        }
