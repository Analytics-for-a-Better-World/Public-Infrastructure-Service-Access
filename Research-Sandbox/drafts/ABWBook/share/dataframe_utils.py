# dataframe_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def optimize_dataframe(
    df: pd.DataFrame,
    category_threshold: int = 50,
    category_density: float = 0.1
) -> pd.DataFrame:
    """
    Optimize a DataFrame by:
    - Converting object columns to categorical if low cardinality or high repetition,
      excluding columns with unhashable types (like dicts or lists).
    - Downcasting float and int columns to smaller dtypes.
    - Replacing monotonic integer columns with RangeIndex if applicable.

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

        # -- 1. Convert object columns to category ------------------------------
        if col_data.dtype == 'object':
            try:
                num_unique = col_data.nunique(dropna=False)
                total = len(col_data)
                repetition_ratio = num_unique / total if total > 0 else 1.0

                if (num_unique <= category_threshold) or (
                    repetition_ratio <= category_density
                ):
                    df[col] = col_data.astype('category')
            except TypeError:
                # Skip columns with unhashable elements
                continue

        # -- 2. Downcast numerics ------------------------------------------------
        elif pd.api.types.is_float_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast='float')

        elif pd.api.types.is_integer_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast='integer')

    # -- 3. Convert monotonic int columns to RangeIndex -------------------------
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


def assert_equal_ignoring_null_types(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    check_dtype: bool = True,
    check_index_type: bool = True,
    **kwargs
) -> None:
    """
    Assert that two DataFrames are equal, treating None and np.nan as equivalent,
    and converting Categoricals to object to avoid strict type checks.

    Args:
        df1, df2: DataFrames to compare.
        check_dtype: Whether to enforce exact dtype matches.
        check_index_type: Whether to enforce exact index type matches.
        kwargs: Passed to pd.testing.assert_frame_equal
    """
    import numpy as np
    import pandas as pd

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.where(pd.notna(df), np.nan).copy()
        for col in df.select_dtypes(include='category').columns:
            df[col] = df[col].astype(object)
        return df

    pd.testing.assert_frame_equal(
        normalize(df1),
        normalize(df2),
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        **kwargs
    )


def plot_memory_usage_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    title: str | None = None,
    sort: bool = True,
    max_height: int = 12,
    file_name: str | None = None,
    per_column: bool = True
) -> None:
    """
    Plot a memory usage comparison chart between two DataFrames.

    Args:
        df_before: Original DataFrame before optimization.
        df_after: Optimized DataFrame.
        title: If provided, the title of the plot.
        sort: If per_column=True, sort bars by descending memory usage (default: True).
        max_height: Maximum figure height in inches for per-column chart.
        file_name: If provided, saves the plot to this path.
        per_column: If True, show per-column comparison; if False, show total usage only.
    """
    mem_before = df_before.memory_usage(deep=True)
    mem_after = df_after.memory_usage(deep=True)

    formatter = FuncFormatter(lambda x, _: f'{x:,.0f}')

    if per_column:
        mem_df = pd.DataFrame({
            'before': mem_before / 1024**2,
            'after': mem_after / 1024**2
        })
        if sort:
            mem_df = mem_df.sort_values('before', ascending=False)

        x = range(len(mem_df))
        width = 0.4
        fig_height = min(max_height, max(4, len(mem_df) * 0.35))

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.bar(x, mem_df['before'], width=width, label='Before', color='#d62728')
        ax.bar([i + width for i in x], mem_df['after'], width=width, label='After', color='#2ca02c')

        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(mem_df.index, rotation=90)
        ax.set_ylabel('Memory Usage (MB)')
    else:
        total_before = mem_before.sum() / 1024**2
        total_after = mem_after.sum() / 1024**2

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Before'], [total_before], color='#d62728', label='Before')
        ax.bar(['After'], [total_after], color='#2ca02c', label='After')
        ax.set_ylabel('Memory Usage (MB)')

        for i, val in enumerate([total_before, total_after]):
            ax.text(i, val + 0.1, f'{val:,.1f} MB', ha='center')

    if title is not None: 
        ax.set_title(title)
    ax.yaxis.set_major_formatter(formatter)
    ax.legend()
    fig.tight_layout()

    if file_name is not None:
        fig.savefig(file_name, dpi=300, bbox_inches='tight')

    plt.show()
