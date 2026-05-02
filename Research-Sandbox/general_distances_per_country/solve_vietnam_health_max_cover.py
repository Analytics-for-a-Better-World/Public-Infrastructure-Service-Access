from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter as pc

import numpy as np
import pandas as pd
import polars as pl


def add_mc_solver_path() -> None:
    """Add the sibling approximated_tradeoff solver module to sys.path."""
    here = Path(__file__).resolve()
    research_sandbox = here.parents[1]
    solver_src = research_sandbox / 'approximated_tradeoff' / 'src'
    sys.path.insert(0, str(solver_src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Solve the Vietnam health maximum-covering experiment from '
            'pipeline parquet outputs.'
        )
    )
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=Path(r'C:\local\Download_Depot\vietnam_data\outputs'),
    )
    parser.add_argument(
        '--run-tag',
        default=(
            'pop_1_sample_1_max_none_agg_10_maxdist_150000_amenity_all_'
            'candidates_spacing_10000_maxsnap_5000'
        ),
    )
    parser.add_argument('--threshold-m', type=float, default=1000.0)
    parser.add_argument('--additions', type=int, default=100)
    parser.add_argument('--time-limit-s', type=int, default=3600)
    parser.add_argument('--mip-gap', type=float, default=1e-6)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    return parser.parse_args()


def load_inputs(outputs_dir: Path, run_tag: str) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    matrix_path = outputs_dir / f'distance_matrix_{run_tag}.parquet'
    population_path = outputs_dir / f'population_{run_tag}.parquet'
    sources_path = outputs_dir / f'sources_{run_tag}.parquet'

    population = pd.read_parquet(population_path, columns=['ID', 'population'])
    population['ID'] = population['ID'].astype(str)

    sources = pd.read_parquet(sources_path, columns=['ID', 'source_type'])
    sources['ID'] = sources['ID'].astype(str)

    return matrix_path, population, sources


def build_solver_inputs(
    matrix_path: Path,
    population: pd.DataFrame,
    sources: pd.DataFrame,
    threshold_m: float,
) -> dict[str, object]:
    target_ids = population['ID'].tolist()
    target_pos = {target_id: idx for idx, target_id in enumerate(target_ids)}
    weights = population['population'].to_numpy(dtype=float)

    source_ids = sources['ID'].tolist()
    source_pos = {source_id: idx for idx, source_id in enumerate(source_ids)}
    existing_source_ids = set(
        sources.loc[sources['source_type'].astype(str) == 'existing', 'ID']
    )

    coverage = (
        pl.scan_parquet(matrix_path)
        .filter(pl.col('total_dist') <= threshold_m)
        .select(['target_id', 'source_id'])
        .unique()
        .collect()
        .to_pandas()
    )
    coverage['target_id'] = coverage['target_id'].astype(str)
    coverage['source_id'] = coverage['source_id'].astype(str)
    coverage = coverage[
        coverage['target_id'].isin(target_pos)
        & coverage['source_id'].isin(source_pos)
    ].copy()
    coverage['i'] = coverage['target_id'].map(target_pos).astype(np.int64)
    coverage['j'] = coverage['source_id'].map(source_pos).astype(np.int64)

    all_facs = {
        int(j): group['i'].to_numpy(dtype=np.int64)
        for j, group in coverage.groupby('j', sort=False)
    }
    existing_j = [source_pos[source_id] for source_id in existing_source_ids]

    covered_existing: set[int] = set()
    for j in existing_j:
        if j in all_facs:
            covered_existing.update(int(i) for i in all_facs[j])

    baseline_covered_population = (
        float(weights[list(covered_existing)].sum()) if covered_existing else 0.0
    )

    add_mc_solver_path()
    from mc_solvers import CreateIndexMapping

    mapping = CreateIndexMapping(all_facs, target_ids)
    existing_in_model = sorted(
        set(existing_j).intersection(set(int(j) for j in mapping.J.tolist()))
    )

    return {
        'weights': weights,
        'mapping': mapping,
        'source_ids': source_ids,
        'existing_j': set(existing_j),
        'existing_in_model': existing_in_model,
        'baseline_covered_population': baseline_covered_population,
        'coverage_pair_count': len(coverage),
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    matrix_path, population, sources = load_inputs(args.outputs_dir, args.run_tag)
    inputs = build_solver_inputs(
        matrix_path=matrix_path,
        population=population,
        sources=sources,
        threshold_m=args.threshold_m,
    )

    add_mc_solver_path()
    from mc_solvers import OptimizeWithGurobipy

    weights = inputs['weights']
    mapping = inputs['mapping']
    existing_in_model = inputs['existing_in_model']
    budget_total = len(existing_in_model) + args.additions

    t0 = pc()
    result = OptimizeWithGurobipy(
        w=weights,
        I=mapping.I,
        J=mapping.J,
        IJ=mapping.IJ,
        budget_list=[budget_total],
        maxTimeInSeconds=args.time_limit_s,
        mipGap=args.mip_gap,
        trace=args.trace,
        already_open=existing_in_model,
    )
    solve_wall_s = pc() - t0

    row = result.loc[budget_total]
    solution = row['solution'] or []
    existing_j = inputs['existing_j']
    selected_candidates = [j for j in solution if j not in existing_j]
    selected_candidate_ids = [inputs['source_ids'][j] for j in selected_candidates]

    total_population = float(np.asarray(weights, dtype=float).sum())
    baseline = float(inputs['baseline_covered_population'])
    optimized = float(row['value'])

    summary = pd.DataFrame(
        [
            {
                'country': 'Vietnam',
                'threshold_m': args.threshold_m,
                'candidate_grid_spacing_m': 10000,
                'aggregate_factor': 10,
                'matrix_max_total_dist_m': 150000,
                'population_points': len(population),
                'total_population': total_population,
                'sources_total': len(sources),
                'existing_sources': int((sources['source_type'] == 'existing').sum()),
                'candidate_sources': int((sources['source_type'] == 'candidate').sum()),
                'coverage_pairs': int(inputs['coverage_pair_count']),
                'baseline_covered_population': baseline,
                'baseline_covered_pct': baseline / total_population * 100,
                'additions': args.additions,
                'budget_total_in_model': budget_total,
                'selected_candidates': len(selected_candidate_ids),
                'optimized_covered_population': optimized,
                'optimized_covered_pct': optimized / total_population * 100,
                'incremental_covered_population': optimized - baseline,
                'incremental_covered_pct_points': (
                    (optimized - baseline) / total_population * 100
                ),
                'termination': row['termination'],
                'solver_reported_s': row['solving'],
                'solve_wall_s': solve_wall_s,
                'upper_bound': row['upper'],
            }
        ]
    )

    summary_path = args.results_dir / 'vietnam_health_100_additions_1000m_summary.csv'
    selected_path = (
        args.results_dir / 'vietnam_health_100_additions_selected_candidates.csv'
    )
    frontier_path = args.results_dir / 'vietnam_health_100_additions_frontier.csv'

    summary.to_csv(summary_path, index=False)
    pd.DataFrame({'source_id': selected_candidate_ids}).to_csv(
        selected_path,
        index=False,
    )
    result.drop(columns=['solution']).to_csv(frontier_path)

    print(summary.to_string(index=False))
    print(f'Wrote {summary_path}')
    print(f'Wrote {selected_path}')
    print(f'Wrote {frontier_path}')


if __name__ == '__main__':
    main()
