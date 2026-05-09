# data_utils.py
import pandas as pd
import numpy as np


def find_dow_mismatches(
    df: pd.DataFrame,
    date_col: str = 'Date',
    dow_col: str = 'DOW'
) -> pd.DataFrame:
    """
    Return rows where the given day-of-week column does not match
    the actual weekday from the date column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    date_col : str, default 'Date'
        Name of the column with dates (string or datetime).
    dow_col : str, default 'DOW'
        Name of the column with the reported day-of-week abbreviations
        (e.g., 'Mon', 'Tue', ...).

    Returns
    -------
    pd.DataFrame
        DataFrame containing only rows where the reported DOW does not match
        the actual date's weekday. Includes an extra column 'expected_DOW'
        showing the correct abbreviation.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    df['expected_DOW'] = df[date_col].dt.strftime('%a')
    mismatches = df[df[dow_col] != df['expected_DOW']]
    return mismatches


def prepare_pairs_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and clean a symmetric pairs matrix from a wide-format column.

    Steps:
    - Copy the matrix from `data`.
    - Set the first column as the index.
    - Remove the index name for cleaner display.
    - Fill the diagonal with zeros (no self-conflict).

    Parameters
    ----------
    data : pd.DataFrame
        The source DataFrame containing the pairs matrix.
    col_name : str, default 'Exam Pairs ABW-2'
        The column name to extract the pairs matrix from.

    Returns
    -------
    pd.DataFrame
        Cleaned square matrix with zero diagonal and proper indices.
    """
    pairs = data.copy()
    pairs.set_index(pairs.columns[0], inplace=True)
    pairs.index.name = None
    # np.fill_diagonal(pairs.values, 0)
    return pairs

def prepare_exam_days(
    data: pd.DataFrame,
    slice_range: slice | tuple[int, int] | None = None
) -> pd.DataFrame:
    """
    Extract and clean an exam-days DataFrame from a wide-format column.

    Steps:
    - Copy the DataFrame from `data`.
    - Optionally slice rows (by .iloc).
    - Cast 'DOW' to categorical.
    - Parse 'Date' column into datetime (dayfirst=True).

    Parameters
    ----------
    data : pd.DataFrame
        The source DataFrame containing exam days.
    col_name : str, default 'exam_days3'
        The column name to extract the exam-days table from.
    slice_range : slice | tuple[int, int] | None, optional
        Row range to select via .iloc. If tuple (start, end), behaves like slice(start, end).
        If None, all rows are returned.

    Returns
    -------
    pd.DataFrame
        Cleaned exam-days DataFrame with proper dtypes.
    """
    days = data.copy()

    if slice_range is not None:
        if isinstance(slice_range, tuple):
            start, end = slice_range
            days = days.iloc[start:end].copy()
        else:
            days = days.iloc[slice_range].copy()

    if 'DOW' in days.columns:
        days['DOW'] = days['DOW'].astype('category')
    if 'Date' in days.columns:
        days['Date'] = pd.to_datetime(days['Date'], dayfirst=True)

    return days


def prepare_exam_list(
    data: pd.DataFrame,
    length_col: str | None = None,
    add_hm: bool = True
) -> pd.DataFrame:
    """
    Extract and clean an exam list with durations.

    Steps:
    - Copy the DataFrame from `data`.
    - Use the first column as the index (exam names).
    - Ensure the length column is integer minutes.
    - Optionally add a formatted H:MM string column.

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame containing the exam list.
    col_name : str, default 'M24 exam names and block lengths'
        Column name where the exam list table is stored.
    length_col : str or None
        Column name with exam lengths. If None, assumes it's the second column.
    add_hm : bool, default True
        Whether to add a formatted 'H:MM' column.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by exam name, with length in minutes
        and optional formatted H:MM column.
    """
    df = data.copy()

    # Set index to first column (exam names)
    df.set_index(df.columns[0], inplace=True)
    df.index.name = None

    # Determine which column stores lengths
    if length_col is None:
        length_col = df.columns[-1]  # assume last column is length

    # Convert to integer minutes
    df[length_col] = pd.to_numeric(df[length_col], errors='coerce').astype('Int64')

    # Optionally add H:MM string
    if add_hm:
        df['Length_hm'] = df[length_col].apply(
            lambda m: f"{m // 60}:{m % 60:02d}" if pd.notna(m) else None
        )

    return df

import numpy as np
import pandas as pd

def buildConflictPairs(pairs: pd.DataFrame, threshold: float):
    """
    Build conflict pairs and weights from a symmetric DataFrame of pairwise scores.

    Parameters
    ----------
    pairs : pd.DataFrame
        Square symmetric DataFrame, index/columns = exam IDs, values = conflict scores.
    threshold : float
        Minimum conflict score to consider (strictly greater than this value).

    Returns
    -------
    conflictPairs : list[tuple]
        List of (i, j) with i < j and conflict > threshold.
    conflictWeights : dict[tuple, float]
        Dictionary mapping (i, j) -> conflict weight.
    """
    mask = np.triu(np.ones(pairs.shape), k=1).astype(bool)  # upper triangle only
    conflicts = pairs.where(mask & (pairs > threshold))

    conflicts_flat = (
        conflicts.stack()
                 .reset_index()
                 .rename(columns={'level_0': 'i', 'level_1': 'j', 0: 'weight'})
    )

    conflictPairs = list(zip(conflicts_flat['i'], conflicts_flat['j']))
    conflictWeights = dict(zip(conflictPairs, conflicts_flat['weight']))

    return conflictPairs, conflictWeights
