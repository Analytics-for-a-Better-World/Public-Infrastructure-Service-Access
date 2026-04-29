import pandas as pd
import geopandas as gpd

# df_utils.py

def optimize_dataframe(
    df: pd.DataFrame,
    category_threshold: int = 50,
    category_density: float = 0.1
) -> pd.DataFrame:
    """
    Optimize a DataFrame by:
    - Converting object columns to categorical if low cardinality or high repetition
    - Downcasting float and int columns to smaller dtypes
    - Replacing monotonic integer columns with RangeIndex if applicable

    Args:
        df: Input DataFrame.
        category_threshold: Max unique values to convert to category directly.
        category_density: Max (unique / total) ratio to allow category conversion.

    Returns:
        Optimized copy of the DataFrame.
    """
    df = df.copy()

    for col in df.columns:
        col_data = df[col]

        # ── 1. Convert object columns to category ──────────────────────────────
        if col_data.dtype == 'object':
            num_unique = col_data.nunique(dropna=False)
            total = len(col_data)
            repetition_ratio = num_unique / total if total > 0 else 1.0

            if (num_unique <= category_threshold) or (
                repetition_ratio <= category_density
            ):
                df[col] = col_data.astype('category')

        # ── 2. Downcast numerics ───────────────────────────────────────────────
        elif pd.api.types.is_float_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast='float')

        elif pd.api.types.is_integer_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast='integer')

    # ── 3. Convert monotonic int columns to RangeIndex ─────────────────────────
    for col in df.select_dtypes(include='int').columns:
        values = df[col]
        if (
            values.is_monotonic_increasing
            and values.diff().dropna().eq(1).all()
        ):
            df = df.set_index(col, drop=True)
            df.index = pd.RangeIndex(
                start=values.iloc[0], stop=values.iloc[-1] + 1, step=1
            )
            break  # only one index column allowed

    return df
