from pathlib import Path
import pickle
from time import perf_counter as pc
from logging_utils import seconds2clock  # if used externally
import pandas as pd
import geopandas as gpd

# cache_utils.py

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
    '''
    Load data from a pickle file located in `data_path` if it exists
    (unless `force_refresh` is True). Otherwise, acquire the data using
    `acquire_func`, save it as a pickle, and return it.

    Parameters:
    - data_path: Path where the cache file is located or saved.
    - file_name: Base name of the file to store/load (without extension).
    - acquire_func: Callable that returns the data when invoked.
    - *args, **kwargs: Arguments passed to `acquire_func`.
    - force_refresh: If True, bypass cache and re-acquire.
    - logger: Optional logger with `.info()` method.
    - verbose_args: If True, include full function arguments in acquisition log.

    Returns:
    - The data object, either loaded from disk or freshly acquired.
    '''
    path = data_path.joinpath(file_name).with_suffix('.pkl')

    if path.exists() and not force_refresh:
        if logger:
            logger.info(f'[{file_name}] Loading cached data from {path}')
        start = pc()
        with path.open('rb') as f:
            data = pickle.load(f)
        duration = pc() - start
        if logger:
            logger.info(f'[{file_name}] Data loaded in {seconds2clock(duration)} from {path}')
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

    if logger:
        logger.info(f'[{file_name}] Data acquired in {seconds2clock(duration)} and saved to {path}')

    return data
