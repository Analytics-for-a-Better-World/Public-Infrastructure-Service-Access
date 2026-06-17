from dataclasses import dataclass
from pathlib import Path


ConfigValue = str | int | float | Path | bool | None


@dataclass(frozen=True, slots=True)
class CountryConfig:
    '''Fully resolved country configuration.'''

    iso3: str
    iso2: str
    country_name: str
    country_slug: str
    projected_epsg: int

    base_root: Path = Path(r'C:\local') / 'Download_Depot'
    distance_threshold_km: float = 150.0
    geofabrik_region: str = 'europe'
    worldpop_year: int = 2020
    worldpop_dataset: str = 'global1'
    worldpop_release: str | None = None
    worldpop_version: str = 'v1'
    worldpop_resolution: str = '100m'
    worldpop_constrained: bool = False
    worldpop_suffix: str = 'ppp'
    worldpop_adjustment: str | None = 'UNadj'
    worldpop_filename: str | None = None
    worldpop_url: str | None = None
    worldpop_path: Path | None = None
    pbf_filename: str | None = None
    pbf_url: str | None = None
    plot_title_suffix: str = 'roads, population, and service facilities'
    boundary_source: str = 'natural_earth'
    candidate_grid_spacing_m: float | None = None
    candidate_exclude_water: bool = True
    candidate_include_boundary: bool = True
    candidate_max_snap_dist_m: float | None = None
    aggregate_factor: int | None = None

    @property
    def BASE_DIR(self) -> Path:
        '''Return the local base directory for the country.'''
        return self.base_root / f'{self.country_slug}_data'

    @property
    def resolved_pbf_filename(self) -> str:
        '''Return the configured PBF filename.'''
        if self.pbf_filename is not None:
            return self.pbf_filename
        return f'{self.country_slug}-latest.osm.pbf'

    @property
    def resolved_worldpop_filename(self) -> str:
        '''Return the configured WorldPop filename.'''
        if self.worldpop_filename is not None:
            return self.worldpop_filename

        if self.worldpop_dataset == 'global2':
            constrained_part = 'CN' if self.worldpop_constrained else 'UA'
            release_part = self.worldpop_release or 'R2025A'
            return (
                f'{self.iso3.lower()}_pop_{self.worldpop_year}_'
                f'{constrained_part}_{self.worldpop_resolution}_'
                f'{release_part}_{self.worldpop_version}.tif'
            )

        parts: list[str] = [
            self.iso3.lower(),
            self.worldpop_suffix,
            str(self.worldpop_year),
        ]
        if self.worldpop_adjustment:
            parts.append(self.worldpop_adjustment)
        return '_'.join(parts) + '.tif'

    @property
    def PBF_URL(self) -> str:
        '''Return the OSM PBF download URL.'''
        if self.pbf_url is not None:
            return self.pbf_url

        return (
            f'https://download.geofabrik.de/'
            f'{self.geofabrik_region}/{self.resolved_pbf_filename}'
        )

    @property
    def WORLDPOP_URL(self) -> str:
        '''Return the WorldPop raster download URL.'''
        if self.worldpop_url is not None:
            return self.worldpop_url

        if self.worldpop_dataset == 'global2':
            release_part = self.worldpop_release or 'R2025A'
            constraint_part = (
                'constrained' if self.worldpop_constrained else 'unconstrained'
            )
            return (
                'https://worldpop-public-data.soton.ac.uk/GIS/Population/'
                f'Global_2015_2030/{release_part}/{self.worldpop_year}/'
                f'{self.iso3}/{self.worldpop_version}/{self.worldpop_resolution}/'
                f'{constraint_part}/{self.resolved_worldpop_filename}'
            )

        return (
            'https://data.worldpop.org/GIS/Population/Global_2000_2020/'
            f'{self.worldpop_year}/{self.iso3}/{self.resolved_worldpop_filename}'
        )

    @property
    def PBF_PATH(self) -> Path:
        '''Return the local OSM PBF path.'''
        return self.BASE_DIR / self.resolved_pbf_filename

    @property
    def WORLDPOP_PATH(self) -> Path:
        '''Return the local WorldPop raster path.'''
        if self.worldpop_path is not None:
            return self.worldpop_path
        return self.BASE_DIR / self.resolved_worldpop_filename

    @property
    def COUNTRY_NAME(self) -> str:
        '''Return the country display name.'''
        return self.country_name

    @property
    def DISTANCE_THRESHOLD_KM(self) -> float:
        '''Return the network distance threshold.'''
        return self.distance_threshold_km

    @property
    def PROJECTED_EPSG(self) -> int:
        '''Return the projected CRS EPSG code.'''
        return self.projected_epsg

    @property
    def PLOT_TITLE(self) -> str:
        '''Return the plot title.'''
        return f'{self.country_name}, {self.plot_title_suffix}'


DEFAULTS: dict[str, ConfigValue] = {
    'base_root': Path(r'C:\local') / 'Download_Depot',
    'distance_threshold_km': 150.0,
    'geofabrik_region': 'europe',
    'worldpop_year': 2020,
    'worldpop_dataset': 'global1',
    'worldpop_release': None,
    'worldpop_version': 'v1',
    'worldpop_resolution': '100m',
    'worldpop_constrained': False,
    'worldpop_suffix': 'ppp',
    'worldpop_adjustment': 'UNadj',
    'worldpop_filename': None,
    'worldpop_url': None,
    'worldpop_path': None,
    'pbf_filename': None,
    'pbf_url': None,
    'plot_title_suffix': 'roads, population, and service facilities',
    'boundary_source': 'natural_earth',
    'candidate_grid_spacing_m': None,
    'candidate_exclude_water': True,
    'candidate_include_boundary': True,
    'candidate_max_snap_dist_m': None,
    'aggregate_factor': None,
}


def build_config(
    overrides: dict[str, ConfigValue],
) -> CountryConfig:
    '''Build a country configuration from defaults plus overrides.'''
    merged: dict[str, ConfigValue] = {**DEFAULTS, **overrides}

    required_keys: tuple[str, ...] = (
        'iso3',
        'iso2',
        'country_name',
        'country_slug',
        'projected_epsg',
    )

    missing: list[str] = [key for key in required_keys if key not in merged]
    if missing:
        missing_str = ', '.join(missing)
        raise ValueError(f'Missing required config keys: {missing_str}')

    base_root = merged['base_root']
    if base_root is None:
        raise ValueError('base_root cannot be None.')

    boundary_source = merged['boundary_source']
    if boundary_source is None:
        raise ValueError('boundary_source cannot be None.')

    candidate_grid_spacing_m = merged['candidate_grid_spacing_m']
    candidate_max_snap_dist_m = merged['candidate_max_snap_dist_m']
    aggregate_factor = merged['aggregate_factor']
    worldpop_dataset = str(merged['worldpop_dataset'])

    if worldpop_dataset not in {'global1', 'global2'}:
        raise ValueError("worldpop_dataset must be 'global1' or 'global2'.")

    if aggregate_factor is not None and int(aggregate_factor) < 2:
        raise ValueError('aggregate_factor must be >= 2 or None.')

    return CountryConfig(
        iso3=str(merged['iso3']),
        iso2=str(merged['iso2']),
        country_name=str(merged['country_name']),
        country_slug=str(merged['country_slug']),
        projected_epsg=int(merged['projected_epsg']),
        base_root=Path(base_root),
        distance_threshold_km=float(merged['distance_threshold_km']),
        geofabrik_region=str(merged['geofabrik_region']),
        worldpop_year=int(merged['worldpop_year']),
        worldpop_dataset=worldpop_dataset,
        worldpop_release=(
            None
            if merged['worldpop_release'] is None
            else str(merged['worldpop_release'])
        ),
        worldpop_version=str(merged['worldpop_version']),
        worldpop_resolution=str(merged['worldpop_resolution']),
        worldpop_constrained=bool(merged['worldpop_constrained']),
        worldpop_suffix=str(merged['worldpop_suffix']),
        worldpop_adjustment=(
            None
            if merged['worldpop_adjustment'] is None
            else str(merged['worldpop_adjustment'])
        ),
        worldpop_filename=(
            None
            if merged['worldpop_filename'] is None
            else str(merged['worldpop_filename'])
        ),
        worldpop_url=(
            None
            if merged['worldpop_url'] is None
            else str(merged['worldpop_url'])
        ),
        worldpop_path=(
            None
            if merged['worldpop_path'] is None
            else Path(merged['worldpop_path'])
        ),
        pbf_filename=(
            None
            if merged['pbf_filename'] is None
            else str(merged['pbf_filename'])
        ),
        pbf_url=(
            None
            if merged['pbf_url'] is None
            else str(merged['pbf_url'])
        ),
        plot_title_suffix=str(merged['plot_title_suffix']),
        boundary_source=str(boundary_source),
        candidate_grid_spacing_m=(
            None if candidate_grid_spacing_m is None else float(candidate_grid_spacing_m)
        ),
        candidate_exclude_water=bool(merged['candidate_exclude_water']),
        candidate_include_boundary=bool(merged['candidate_include_boundary']),
        candidate_max_snap_dist_m=(
            None
            if candidate_max_snap_dist_m is None
            else float(candidate_max_snap_dist_m)
        ),
        aggregate_factor=None if aggregate_factor is None else int(aggregate_factor),
    )
