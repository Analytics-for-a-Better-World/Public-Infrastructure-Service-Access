import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import re

def numeric_sort_key(path: Path) -> any:
    """Extracts leading integer from a folder name for numeric sorting."""
    match = re.match(r'(\d+)', path.name)
    return int(match.group(1)) if match else float('inf')


def load_any(path: str | Path, *, allowed_cols: set[int] = {2, 4}) -> object:
    """Load .npy, .npz, .pkl, or numeric text.

    If the loaded object is a 2D NumPy array and one dimension is in allowed_cols,
    it is oriented to shape (n, k) with k in allowed_cols.
    """
    path = Path(path)

    if path.suffix == '.npy':
        result = np.load(path, allow_pickle=True)

    elif path.suffix == '.npz':
        data = np.load(path, allow_pickle=True)
        if len(data.files) != 1:
            raise ValueError(f'Expected one array in {path.name}, found {len(data.files)}')
        result = data[data.files[0]]

    elif path.suffix == '.pkl':
        with open(path, 'rb') as f:
            result = pickle.load(f)

    else:
        result = np.loadtxt(path)

    if isinstance(result, np.ndarray) and result.ndim == 2:
        r, c = result.shape
        r_ok = r in allowed_cols
        c_ok = c in allowed_cols

        if r_ok and not c_ok:
            result = result.T
        elif r_ok and c_ok:
            raise ValueError(f'{path.name} ambiguous shape {result.shape}, cannot infer orientation')

    return result

def build_mapping(
    df: pd.DataFrame,
    threshold: float,
    key_col: str,
    value_col: str,
) -> dict[int, np.ndarray]:
    '''
    Build a dictionary mapping key_col -> sorted numpy array of value_col.

    The mapping is constructed from rows where total_dist <= threshold.

    Ordering guarantees
    -------------------
    1) Dictionary keys are sorted in ascending order because
       pandas.groupby(..., sort=True) is used.

    2) Values associated with each key are returned as numpy arrays of dtype
       int32 and are explicitly sorted using numpy.sort.

    These guarantees make the output suitable for algorithms that assume
    sorted adjacency lists, for example when building compressed row style
    representations with assume_unique_sorted=True.

    Parameters
    ----------
    df
        DataFrame containing at least the columns key_col, value_col,
        and total_dist.
    threshold
        Maximum allowed value of total_dist.
    key_col
        Column whose values define the dictionary keys.
    value_col
        Column whose values are collected as arrays.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from key_col to sorted numpy arrays of value_col.
        Each array has dtype int32.
    '''
    filtered = df.loc[df['total_dist'] <= threshold, [key_col, value_col]]

    result: dict[int, np.ndarray] = {}

    for k, g in filtered.groupby(key_col, sort=True):
        result[int(k)] = np.sort(g[value_col].to_numpy(dtype=np.int32))

    return result

def reverse_mapping(mapping: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    '''
    Reverse a mapping from i -> sorted array[j] into j -> sorted array[i].

    Ordering guarantees
    -------------------
    The returned mapping guarantees:

    1) keys are sorted in ascending order
    2) values are sorted numpy arrays of dtype int32

    Parameters
    ----------
    mapping
        Dictionary mapping each i to a numpy array of j values.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from j to sorted numpy arrays of i.
    '''
    reversed_lists: dict[int, list[int]] = {}

    for i, js in mapping.items():
        for j in np.asarray(js, dtype=np.int32):
            reversed_lists.setdefault(int(j), []).append(int(i))

    return {
        j: np.asarray(is_, dtype=np.int32)
        for j, is_ in sorted(reversed_lists.items())
    }