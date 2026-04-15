from pathlib import Path
from time import perf_counter as pc
import zipfile

import geopandas as gpd

from distance_pipeline.io import download_file


NATURAL_EARTH_COUNTRIES_URL = (
    'https://naturalearth.s3.amazonaws.com/10m_cultural/'
    'ne_10m_admin_0_countries.zip'
)


def download_natural_earth_boundaries(
    destination: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    '''Download the Natural Earth country boundaries archive.'''
    return download_file(
        NATURAL_EARTH_COUNTRIES_URL,
        destination,
        overwrite=overwrite,
        verbose=verbose,
    )


def load_country_geometry(
    iso3: str,
    cache_dir: str | Path,
    projected_epsg: int,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''Load a country boundary from Natural Earth and project it.'''
    t0 = pc()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / 'ne_10m_admin_0_countries.zip'
    extract_dir = cache_dir / 'ne_10m_admin_0_countries'

    download_natural_earth_boundaries(archive_path, overwrite=False, verbose=verbose)

    if not extract_dir.exists():
        if verbose:
            print(f'Extracting Natural Earth boundaries to {extract_dir}')
        with zipfile.ZipFile(archive_path, 'r') as zip_file:
            zip_file.extractall(extract_dir)

    shapefiles = sorted(extract_dir.glob('*.shp'))
    if not shapefiles:
        raise FileNotFoundError(f'No shapefile found in {extract_dir}')

    boundaries = gpd.read_file(shapefiles[0])

    country = boundaries.loc[boundaries['ADM0_A3'] == iso3].copy()
    if country.empty:
        raise ValueError(f'Country ISO3 code {iso3} not found in Natural Earth boundaries.')

    if country.crs is None:
        country = country.set_crs(epsg=4326)

    country = country.to_crs(epsg=projected_epsg)

    if verbose:
        print(
            f'Loaded country boundary for {iso3} in {pc() - t0:.2f} seconds, '
            f'{len(country):,} feature(s)'
        )

    return country
