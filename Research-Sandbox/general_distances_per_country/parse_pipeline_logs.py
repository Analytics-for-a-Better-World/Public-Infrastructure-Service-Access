from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PATTERNS = {
    'country': re.compile(r'Running pipeline for (?P<value>.+)$'),
    'aggregate_factor': re.compile(r'Aggregate factor: (?P<value>.+)$'),
    'amenity_filter': re.compile(r'Amenity filter: (?P<value>.+)$'),
    'include_healthcare_tag': re.compile(r'Include healthcare tag: (?P<value>.+)$'),
    'population_points': re.compile(r'Population points: (?P<value>[\d,]+)'),
    'facilities': re.compile(r'Facilities: (?P<value>[\d,]+)'),
    'sources_total': re.compile(r'Sources total: (?P<value>[\d,]+)'),
    'spatial_setup': re.compile(
        r'preparing (?P<targets>[\d,]+) x (?P<sources>[\d,]+) '
        r'for spatial nearest neighbors bounded by (?P<threshold>[\d.]+) km '
        r'in (?P<time>[\d.]+) seconds'
    ),
    'spatial_pairs': re.compile(
        r'finding (?P<count>[\d,]+) pairs of spatial nearest neighbors '
        r'in (?P<time>[\d.]+) seconds'
    ),
    'unique_node_pairs': re.compile(
        r'creating (?P<count>[\d,]+) unique target source node pairs '
        r'in (?P<time>[\d.]+) seconds'
    ),
    'shortest_paths': re.compile(
        r'(?P<count>[\d,]+) shortest paths of which (?P<valid>[\d,]+) exist '
        r'found in (?P<time>[\d.]+) seconds'
    ),
    'assembled_distances': re.compile(
        r'assembling (?P<count>[\d,]+) distances of interest '
        r'in (?P<time>[\d.]+) seconds'
    ),
    'distance_matrix_size': re.compile(r'Distance matrix size: (?P<value>[\d,]+)'),
    'distance_computation_time': re.compile(
        r'Distance computation time: (?P<value>[\d.]+)'
    ),
    'total_runtime': re.compile(r'Total runtime: (?P<value>[\d.]+)s'),
}


def parse_int(value: str) -> int:
    '''Parse an integer that may contain thousands separators.'''
    return int(value.replace(',', ''))


def parse_optional_float(value: str) -> float | None:
    '''Parse a float-like value, returning None for textual missing values.'''
    if value.lower() in {'none', 'null', ''}:
        return None
    return float(value)


def parse_log(path: Path) -> dict[str, object]:
    '''Parse a single pipeline log file into a dictionary of metrics.'''
    metrics: dict[str, object] = {
        'run': path.stem,
        'log_file': str(path),
    }

    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        for key, pattern in PATTERNS.items():
            match = pattern.search(line)
            if match is None:
                continue

            groups = match.groupdict()

            if key == 'country':
                metrics['country'] = groups['value']

            elif key == 'aggregate_factor':
                metrics['aggregate_factor'] = groups['value']

            elif key == 'amenity_filter':
                metrics['amenity_filter'] = groups['value']

            elif key == 'include_healthcare_tag':
                metrics['include_healthcare_tag'] = groups['value']

            elif key in {
                'population_points',
                'facilities',
                'sources_total',
                'distance_matrix_size',
            }:
                metrics[key] = parse_int(groups['value'])

            elif key == 'spatial_setup':
                metrics['targets'] = parse_int(groups['targets'])
                metrics['sources'] = parse_int(groups['sources'])
                metrics['distance_threshold_km'] = float(groups['threshold'])
                metrics['spatial_setup_time_s'] = float(groups['time'])

            elif key == 'spatial_pairs':
                metrics['spatial_pairs'] = parse_int(groups['count'])
                metrics['spatial_pairs_time_s'] = float(groups['time'])

            elif key == 'unique_node_pairs':
                metrics['unique_node_pairs'] = parse_int(groups['count'])
                metrics['unique_node_pairs_time_s'] = float(groups['time'])

            elif key == 'shortest_paths':
                metrics['shortest_paths'] = parse_int(groups['count'])
                metrics['valid_shortest_paths'] = parse_int(groups['valid'])
                metrics['shortest_paths_time_s'] = float(groups['time'])

            elif key == 'assembled_distances':
                metrics['assembled_distances'] = parse_int(groups['count'])
                metrics['assembly_time_s'] = float(groups['time'])

            elif key == 'distance_computation_time':
                metrics['distance_computation_time_s'] = float(groups['value'])

            elif key == 'total_runtime':
                metrics['total_runtime_s'] = float(groups['value'])

    return metrics


def parse_logs(log_dir: Path) -> pd.DataFrame:
    '''Parse all log files in a directory into a DataFrame.'''
    rows = [parse_log(path) for path in sorted(log_dir.glob('*.log'))]
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f'No log files found in {log_dir}')

    if {'targets', 'sources'}.issubset(df.columns):
        df['full_cartesian_pairs'] = df['targets'] * df['sources']

    if {'spatial_pairs', 'full_cartesian_pairs'}.issubset(df.columns):
        df['spatial_pair_share'] = df['spatial_pairs'] / df['full_cartesian_pairs']

    if {'unique_node_pairs', 'spatial_pairs'}.issubset(df.columns):
        df['unique_pair_share'] = df['unique_node_pairs'] / df['spatial_pairs']
        df['node_pair_reduction_pct'] = 100 * (1 - df['unique_pair_share'])

    if {'valid_shortest_paths', 'shortest_paths'}.issubset(df.columns):
        df['valid_path_share'] = df['valid_shortest_paths'] / df['shortest_paths']

    return df


def format_int(value: object) -> str:
    '''Format integers for LaTeX tables.'''
    if pd.isna(value):
        return ''
    return f'{int(value):,}'


def format_float(value: object, decimals: int = 2) -> str:
    '''Format floats for LaTeX tables.'''
    if pd.isna(value):
        return ''
    return f'{float(value):.{decimals}f}'


def write_performance_table(df: pd.DataFrame, output_path: Path) -> None:
    '''Write a LaTeX table with one row per run.'''
    table = pd.DataFrame({
        'Run': df['run'],
        'Population': df.get('targets', df.get('population_points')).map(format_int),
        'Sources': df.get('sources', df.get('sources_total')).map(format_int),
        'Spatial pairs': df.get('spatial_pairs').map(format_int),
        'Unique node pairs': df.get('unique_node_pairs').map(format_int),
        'Valid paths': df.get('valid_shortest_paths').map(format_int),
        'Distances': df.get('distance_matrix_size').map(format_int),
        'Runtime (s)': df.get('total_runtime_s').map(format_float),
    })

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format='lrrrrrrr',
        caption='Performance summary for pipeline runs.',
        label='tab:performance_summary',
    )
    output_path.write_text(latex, encoding='utf-8')


def write_runtime_breakdown_table(df: pd.DataFrame, output_path: Path) -> None:
    '''Write a LaTeX table with timing breakdown by run.'''
    table = pd.DataFrame({
        'Run': df['run'],
        'Spatial query (s)': df.get('spatial_pairs_time_s').map(format_float),
        'Node pair reduction (s)': df.get('unique_node_pairs_time_s').map(format_float),
        'Shortest paths (s)': df.get('shortest_paths_time_s').map(format_float),
        'Assembly (s)': df.get('assembly_time_s').map(format_float),
        'Distance stage (s)': df.get('distance_computation_time_s').map(format_float),
        'Total (s)': df.get('total_runtime_s').map(format_float),
    })

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format='lrrrrrr',
        caption='Runtime breakdown for distance matrix construction.',
        label='tab:runtime_breakdown',
    )
    output_path.write_text(latex, encoding='utf-8')


def plot_runtime_breakdown(df: pd.DataFrame, output_path: Path) -> None:
    '''Create a stacked bar chart with runtime components.'''
    columns = [
        'spatial_pairs_time_s',
        'unique_node_pairs_time_s',
        'shortest_paths_time_s',
        'assembly_time_s',
    ]
    available = [col for col in columns if col in df.columns]

    if not available:
        return

    plot_df = df.set_index('run')[available].fillna(0)
    plot_df = plot_df.rename(columns={
        'spatial_pairs_time_s': 'Spatial query',
        'unique_node_pairs_time_s': 'Unique node pairs',
        'shortest_paths_time_s': 'Shortest paths',
        'assembly_time_s': 'Assembly',
    })

    ax = plot_df.plot(kind='bar', stacked=True)
    ax.set_xlabel('Run')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime breakdown by pipeline stage')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_total_runtime(df: pd.DataFrame, output_path: Path) -> None:
    '''Create a bar chart with total runtime by run.'''
    if 'total_runtime_s' not in df.columns:
        return

    ax = df.plot(x='run', y='total_runtime_s', kind='bar', legend=False)
    ax.set_xlabel('Run')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Total runtime by run')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    '''Parse logs and write LaTeX tables and figures.'''
    log_dir = Path('logs')
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)

    df = parse_logs(log_dir)
    df.to_csv(output_dir / 'log_metrics.csv', index=False)

    write_performance_table(df, output_dir / 'performance_table.tex')
    write_runtime_breakdown_table(df, output_dir / 'runtime_breakdown_table.tex')
    plot_runtime_breakdown(df, output_dir / 'runtime_breakdown.png')
    plot_total_runtime(df, output_dir / 'runtime_by_run.png')

    print(df)


if __name__ == '__main__':
    main()
