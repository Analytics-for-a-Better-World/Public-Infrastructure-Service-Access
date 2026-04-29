# io_utils.py

# ─── Standard Library Imports ──────────────────────────────────────────────
import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import wraps
from time import perf_counter as pc

# ─── Third-Party Imports ───────────────────────────────────────────────────
import requests
import pandas as pd
import geopandas as gpd

# ─── Local Imports ─────────────────────────────────────────────────────────
from logging_utils import seconds2clock


def load_or_acquire(
    data_path: Path,
    file_name: str,
    acquire_func: callable,
    *args,
    force_refresh: bool = False,
    logger=None,
    verbose_args: bool = False,
    **kwargs
) -> object:
    """
    Load data from a pickle file located in `data_path` if it exists
    (unless `force_refresh` is True). Otherwise, acquire the data using
    `acquire_func`, save it as a pickle, and return it.

    Args:
        data_path: Directory where the cache file is stored or saved.
        file_name: File name (without extension).
        acquire_func: Callable to generate the data if cache is missing or forced.
        *args, **kwargs: Arguments passed to `acquire_func`.
        force_refresh: If True, re-run `acquire_func` and overwrite cache.
        logger: Optional logger with `.info()` method.
        verbose_args: If True, include full arguments in log messages.

    Returns:
        Loaded or freshly acquired object.
    """
    path = data_path.joinpath(file_name).with_suffix('.pkl')

    if path.exists() and not force_refresh:
        if logger:
            logger.info(f'[{file_name}] Loading cached data from {path}')
        start = pc()
        with path.open('rb') as f:
            data = pickle.load(f)
        duration = pc() - start
        size = path.stat().st_size
        if logger:
            logger.info(f'[{file_name}] Data loaded in {seconds2clock(duration)} from {path} ({size:,} bytes)')
        return data

    if logger:
        reason = 'forced refresh' if force_refresh else 'cache miss'
        if verbose_args:
            arg_str = ', '.join(map(repr, args))
            kwarg_str = ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
            full_args = f'{arg_str}{", " if args and kwargs else ""}{kwarg_str}'
            logger.info(f'[{file_name}] Acquiring new data due to {reason}, using {acquire_func.__name__}({full_args})')
        else:
            logger.info(f'[{file_name}] Acquiring new data due to {reason}, using {acquire_func.__name__}()')

    path.parent.mkdir(parents=True, exist_ok=True)

    start = pc()
    data = acquire_func(*args, **kwargs)
    duration = pc() - start

    with path.open('wb') as f:
        pickle.dump(data, f)
    size = path.stat().st_size

    if logger:
        logger.info(f'[{file_name}] Data acquired in {seconds2clock(duration)} and saved to {path} ({size:,} bytes)')

    return data


def get_remote_osm_pbf_timestamp(region: str) -> datetime | None:
    """
    Fetch the Last-Modified timestamp of the remote .pbf file for a region
    (e.g., from Geofabrik), typically used to detect changes.

    Args:
        region: Region name (e.g., "netherlands")

    Returns:
        Timezone-aware datetime if available, else None.
    """
    base_url = 'https://download.geofabrik.de/europe'
    file_url = f'{base_url}/{region.lower()}-latest.osm.pbf'

    try:
        response = requests.head(file_url, allow_redirects=True, timeout=10)
        if 'Last-Modified' in response.headers:
            return parsedate_to_datetime(response.headers['Last-Modified'])
        print(f'Warning: No Last-Modified header found at {file_url}')
    except Exception as e:
        print(f'Warning: Failed to fetch metadata from {file_url}: {e}')

    return None


def disk_cache(cache_dir: str = 'cache') -> callable:
    """
    A decorator that caches the result of a function call to disk.

    Uses a hash of the function name and arguments to store/load unique results.

    Args:
        cache_dir: Directory where cache files will be stored.

    Returns:
        Decorated function with persistent caching behavior.
    """
    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> object:
            os.makedirs(cache_dir, exist_ok=True)
            hash_key = hashlib.sha256()
            hash_key.update(func.__name__.encode())
            hash_key.update(pickle.dumps(args))
            hash_key.update(pickle.dumps(kwargs))
            filename = os.path.join(cache_dir, f'{hash_key.hexdigest()}.pkl')

            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(filename, 'wb') as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return decorator
