# ib_graphics.py
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from matplotlib.ticker import FuncFormatter
from textwrap import fill


def human_format(num: int) -> str:
    """Convert an integer to a human-readable string like 1k, 1M, etc."""
    if num < 1000:
        return str(num)
    for unit in ['k', 'M', 'B', 'T']:
        num /= 1000.0
        if abs(num) < 1000:
            s = f"{num:.1f}".rstrip('0').rstrip('.')
            return f"{s}{unit}"
    s = f"{num:.1f}".rstrip('0').rstrip('.')
    return f"{s}P"


def plot_exam_matrix(
    data: np.ndarray,
    figsize: tuple[float, float] = (20, 10),
    file_name: str | None = None
) -> None:
    """
    Plot exam matrix with human-readable annotations, larger labels,
    smaller annotation font, and a full-height, slim colorbar.

    If file_name is provided, saves the figure to that path.
    Otherwise, displays interactively.
    """
    annot_labels = np.vectorize(human_format)(data)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        data,
        cmap='YlOrRd',
        annot=annot_labels,
        fmt='',
        annot_kws={'size': 6},   # smaller font for counts
        cbar=True,
        cbar_kws={'aspect': 20, 'pad': 0.02},  # slim, full-height colorbar
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_month_calendar(
    days: pd.DataFrame,
    date_col: str = 'Date',
    first_weekday: int = 0,
    events: list[tuple[str, str, pd.Timestamp]] | None = None,
    *,
    fig_width: float = 12.0,
    fig_height: float = 6.0,
    wrap_width: int = 16,
    fontsize_day: int = 9,
    fontsize_event: int = 8,
    color_in_month: str = 'white',
    color_out_of_month: str = '#e0e0e0',
    color_missing_day: str = '#d0d0d0',
    color_free_day: str = '#ffcccc',  # <─ NEW default (light red)
    free_days: list[pd.Timestamp] | None = None,  # <─ NEW argument
    line_gap: float = 0.14,
    maxPerSlot: int | None = None,
    autoFit: bool = True,
    file_name: str | None = None
) -> list[str] | list[plt.Figure]:
    """
    Plot calendars for all (year, month) in `days`, supporting multiple events per slot.

    If `file_name` is provided, saves one file per month with prefix "{year}_{month:02d}_".
    Otherwise, returns Figures for interactive display.

    Parameters
    ----------
    days : pd.DataFrame
        Must contain a date column.
    free_days : list[pd.Timestamp], optional
        Specific days to highlight as "free days" (holidays, weekends).
    color_free_day : str, default '#ffcccc'
        Background color for free days.
    """
    s = days.copy()
    if not np.issubdtype(s[date_col].dtype, np.datetime64):
        s[date_col] = pd.to_datetime(s[date_col], dayfirst=True)

    s['__norm_date'] = s[date_col].dt.normalize()
    s['year'] = s['__norm_date'].dt.year
    s['month'] = s['__norm_date'].dt.month

    # Normalize free days list
    free_set: set[pd.Timestamp] = set()
    if free_days:
        free_set = {pd.Timestamp(d).normalize() for d in free_days}

    ev_map: dict[pd.Timestamp, dict[str, list[str]]] = {}
    if events:
        for name, slot, dt in events:
            d = pd.Timestamp(dt).normalize()
            slot = 'morning' if str(slot).lower().startswith('m') else 'afternoon'
            m = ev_map.setdefault(d, {'morning': [], 'afternoon': []})
            m[slot].append(str(name))

    cal = calendar.Calendar(firstweekday=first_weekday)
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekday_labels = weekday_labels[first_weekday:] + weekday_labels[:first_weekday]

    def wrapList(texts: list[str]) -> list[str]:
        return [fill(t, width=wrap_width) for t in texts]

    def limitBy(texts: list[str], capacity: int | None) -> list[str]:
        if capacity is None or capacity <= 0:
            return []
        if len(texts) <= capacity:
            return texts
        return texts[:capacity] + ['...']

    def drawLines(ax, texts: list[str], *, x: float, y0: float, upward: bool, color: str) -> None:
        for i, t in enumerate(texts):
            y = y0 + (i * line_gap if not upward else -i * line_gap)
            ax.text(x, y, t, fontsize=fontsize_event,
                    va='top' if not upward else 'bottom', ha='left', color=color)

    figs: list[plt.Figure] = []
    saved_files: list[str] = []

    for (year, month), g in s.groupby(['year', 'month'], sort=True):
        present = set(g['__norm_date'].unique())
        if len(present) == 0:
            continue

        weeks = cal.monthdatescalendar(year, month)
        n_rows = len(weeks)
        grid = np.array(weeks)

        fig, ax = plt.subplots()
        fig.set_size_inches(fig_width, fig_height, forward=True)
        fig.set_dpi(120)
        ax.set_title(f'{calendar.month_name[month]} {year}')
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels(weekday_labels)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([f'W{i+1}' for i in range(n_rows)])
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(n_rows - 0.5, -0.5)

        pad_top = 0.06
        pad_bot = 0.06

        for r in range(n_rows):
            for c in range(7):
                day = pd.Timestamp(grid[r, c])
                in_month = (day.month == month)
                day_norm = day.normalize()

                # Decide background color
                if day_norm in free_set:
                    face = color_free_day
                elif in_month and day_norm in present:
                    face = color_in_month
                elif in_month:
                    face = color_missing_day
                else:
                    face = color_out_of_month

                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor=face, edgecolor='black', linewidth=1))
                ax.text(c - 0.46, r - 0.36, str(day.day),
                        fontsize=fontsize_day, va='top', ha='left')

                if not in_month:
                    continue

                itemsM = ev_map.get(day_norm, {}).get('morning', [])
                itemsA = ev_map.get(day_norm, {}).get('afternoon', [])

                capM = capA = None
                if autoFit:
                    y0M = r - 0.12
                    bottom = r + 0.5 - pad_bot
                    availM = max(0.0, bottom - y0M)
                    capM = int(availM // line_gap)

                    y0A = r + 0.38
                    top = r - 0.5 + pad_top
                    availA = max(0.0, y0A - top)
                    capA = int(availA // line_gap)

                    if maxPerSlot is not None:
                        capM = min(capM, maxPerSlot)
                        capA = min(capA, maxPerSlot)
                else:
                    y0M = r - 0.12
                    y0A = r + 0.38
                    capM = capA = maxPerSlot if maxPerSlot is not None else 10**9

                drawM = limitBy(wrapList(itemsM), capM)
                drawA = limitBy(wrapList(itemsA), capA)

                if drawM:
                    drawLines(ax, drawM, x=c - 0.46, y0=y0M, upward=False, color='red')
                if drawA:
                    drawLines(ax, drawA, x=c - 0.46, y0=y0A, upward=True, color='green')

        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        if file_name:
            out_path = f"{year}_{month:02d}_{file_name}"
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            saved_files.append(out_path)
        else:
            figs.append(fig)

    if file_name:
        return saved_files
    return figs
