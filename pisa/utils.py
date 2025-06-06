"""Utility functions and helpers for the PISA package.

This module contains utility functions that are used throughout the PISA package, including
validation functions, caching mechanisms, and other helper functions that support the core
functionality of the package.

See Also
--------
constants : Module containing constants used by these utility functions
"""

import hashlib
import logging
import os
import pickle
from functools import wraps
from typing import Callable

from pisa.constants import VALID_DISTANCE_TYPES, VALID_MODES_OF_TRANSPORT

logger = logging.getLogger(__name__)


def disk_cache(cache_dir: str = "cache") -> Callable:
    """Implement disk-based caching for function results.
    
    This decorator saves the result of a function call to a file on disk and loads it
    on subsequent calls with the same arguments, avoiding redundant computations.
    
    Parameters
    ----------
    cache_dir : str, optional
        Directory where cache files will be stored. (default: ``cache``)
    
    Returns
    -------
    callable
        A decorated function that implements caching behavior
    
    Example
    -------
    >>> @disk_cache()
    >>> def expensive_computation(x):
    >>>     # Some time-consuming calculation
    >>>     return x ** 2
    
    Notes
    -----
    - Cache files are stored as pickle files
    - Cache keys are generated using SHA-256 hash of function name and arguments
    - Logs cache hits/misses using the logging module
    - Creates cache directory if it doesn't exist
    - Cache files are named using the hash of the function name and arguments
    
    Raises
    ------
    pickle.PickleError
        If there are issues serializing/deserializing the cached data
    OSError
        If there are issues with file operations
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Create a hash key from the function name and arguments
            hash_key = hashlib.sha256()
            hash_key.update(func.__name__.encode())
            hash_key.update(pickle.dumps(args))
            hash_key.update(pickle.dumps(kwargs))
            filename = f"{cache_dir}/{hash_key.hexdigest()}.pkl"

            # add logging information
            logger.info("\n=== Cache Status ===")
            logger.info(f"Function: {func.__name__}")
            logger.debug(f"Args: {args}")
            logger.info(f"Cache file: {filename}")

            # check if the cache file exists:
            if os.path.exists(filename):
                logger.info("Cache HIT - Loading cached result")
                with open(filename, "rb") as f:
                    return pickle.load(f)
            else:
                # Call the function and cache its result
                logger.info("Cache MISS - Computing new result")
                result = func(*args, **kwargs)
                with open(filename, "wb") as f:
                    pickle.dump(result, f)
                return result

        return wrapper

    return decorator


def validate_distance_type(distance_type: str) -> str:
    """Validate and normalize distance type input.
    
    Parameters
    ----------
    distance_type : str
        The distance type to validate (``length`` or ``travel_time``)
        
    Returns
    -------
    str
        Normalized distance type (lowercase, stripped of whitespace)
        
    Raises
    ------
    ValueError
        If distance_type is not one of the valid types defined in VALID_DISTANCE_TYPES
    
    See Also
    --------
    VALID_DISTANCE_TYPES : Set of valid distance types
    """
    distance_type = distance_type.lower().strip()

    if distance_type not in VALID_DISTANCE_TYPES:
        raise ValueError(f"distance_type must be one of {VALID_DISTANCE_TYPES}")
    return distance_type


def validate_mode_of_transport(mode_of_transport: str) -> str:
    """Validate and normalize mode of transport input.
    
    Parameters
    ----------
    mode_of_transport : str
        The mode of transport to validate (e.g., ``driving``, ``walking``, ``cycling``)
        
    Returns
    -------
    str
        Normalized mode of transport (lowercase, stripped of whitespace)
        
    Raises
    ------
    ValueError
        If mode_of_transport is not one of the valid modes defined in VALID_MODES_OF_TRANSPORT
        
    Notes
    -----
    This function normalizes the input by converting to lowercase and removing
    leading/trailing whitespace before validation.
    
    See Also
    --------
    VALID_MODES_OF_TRANSPORT : Set of valid transport modes
    """
    mode_of_transport = mode_of_transport.lower().strip()

    if mode_of_transport not in VALID_MODES_OF_TRANSPORT:
        raise ValueError(f"mode_of_transport must be one of {VALID_MODES_OF_TRANSPORT}")
    return mode_of_transport


def validate_fallback_speed(
    fallback_speed: int | float | None, network_type: str
) -> int | float | None:
    """Validate that a fallback speed is within reasonable bounds for the given transport mode.
    
    Parameters
    ----------
    fallback_speed : int, float, or None
        The fallback speed to validate, in kilometers per hour.
        If None, no validation is performed.
    network_type : str
        The network type/mode of transport (``drive``, ``walk``, ``bike``)
        
    Returns
    -------
    int, float, or None
        The validated fallback speed or None if no fallback speed was provided
        
    Raises
    ------
    ValueError
        - If fallback_speed is not a number
        - If fallback_speed is not positive
        - If fallback_speed exceeds reasonable bounds for the given mode of transport:

            - For walking: speed must be <= 7 km/h
            - For cycling: speed must be <= 25 km/h
            - For driving: speed must be <= 130 km/h
        
    Notes
    -----
    This function is used to ensure that fallback speeds (used when OSM data doesn't
    provide speed information) are within reasonable bounds for the mode of transport.
    """

    if fallback_speed is not None:

        if not isinstance(fallback_speed, (int, float)):
            raise ValueError("Fallback speed must be a number")

        transport_specific_bounds = {
            "drive": (0, 200),
            "walk": (0, 20),
            "bike": (0, 50),
        }
        min_speed, max_speed = transport_specific_bounds[network_type]
        if not min_speed <= fallback_speed <= max_speed:
            raise ValueError(
                f"Fallback speed {fallback_speed} is out of bounds for {network_type}. "
                f"Valid range is {transport_specific_bounds[network_type]}"
            )

    return fallback_speed
