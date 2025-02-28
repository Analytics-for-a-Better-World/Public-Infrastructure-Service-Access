import hashlib
import logging
import os
import pickle
from functools import wraps

logger = logging.getLogger(__name__)


def disk_cache(cache_dir="cache"):
    """
    A decorator that implements disk-based caching for function results.
    This decorator saves the result of a function call to a file on disk and loads it
    on subsequent calls with the same arguments, avoiding redundant computations.
    Args:
        cache_dir (str, optional): Directory where cache files will be stored. Defaults to "cache".
    Returns:
        callable: A decorated function that implements caching behavior.
    Example:
        @disk_cache()
        def expensive_computation(x):
            # Some time-consuming calculation
            return x ** 2
    Notes:
        - Cache files are stored as pickle files
        - Cache keys are generated using SHA-256 hash of function name and arguments
        - Logs cache hits/misses using the logging module
        - Creates cache directory if it doesn't exist
        - Cache files are named using the hash of the function name and arguments
    Raises:
        pickle.PickleError: If there are issues serializing/deserializing the cached data
        OSError: If there are issues with file operations
    """

    def decorator(func):
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
