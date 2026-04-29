import requests
from email.utils import parsedate_to_datetime
from datetime import datetime
import pandas as pd
import geopandas as gpd

# remote_utils.py

def get_remote_osm_pbf_timestamp(region: str) -> datetime | None:
    '''
    Fetch the Last-Modified timestamp of the remote .pbf file for the given region,
    typically used with Pyrosm or Geofabrik downloads.

    Parameters:
    - region: Name of the region (e.g., "netherlands")

    Returns:
    - A timezone-aware datetime object representing the remote file's last modified time,
      or None if it cannot be retrieved.
    '''
    base_url = 'https://download.geofabrik.de/europe'
    file_url = f'{base_url}/{region.lower()}-latest.osm.pbf'

    try:
        response = requests.head(file_url, allow_redirects=True, timeout=10)
        if 'Last-Modified' in response.headers:
            return parsedate_to_datetime(response.headers['Last-Modified'])
        else:
            print(f'Warning: No Last-Modified header found at {file_url}')
    except Exception as e:
        print(f'Warning: Failed to fetch remote metadata from {file_url}: {e}')

    return None
