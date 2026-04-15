from __future__ import annotations

import pandas as pd

from distance_pipeline.pipeline_support import ensure_xy_columns



def ensure_id_column(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Ensure the table contains a valid string ``ID`` column."""
    result = df.copy()

    if 'ID' not in result.columns:
        result['ID'] = [f'{prefix}_{i}' for i in range(len(result))]

    result['ID'] = result['ID'].astype(str)
    return result



def ensure_id_index_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Force ``df.index`` to match ``df['ID']``."""
    if 'ID' not in df.columns:
        raise KeyError("Expected an 'ID' column.")

    result = df.copy()

    if result['ID'].isna().any():
        raise ValueError("Column 'ID' contains missing values.")

    result['ID'] = result['ID'].astype(str)

    if result['ID'].duplicated().any():
        duplicate_ids = result.loc[result['ID'].duplicated(), 'ID'].unique().tolist()
        preview = duplicate_ids[:10]
        raise ValueError(
            f"Column 'ID' must be unique. Duplicate values found, for example: {preview}"
        )

    result.index = result['ID']
    return result



def set_known_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known low cardinality string columns to categorical dtype."""
    result = df.copy()

    if 'source_type' in result.columns:
        result['source_type'] = pd.Categorical(
            result['source_type'],
            categories=['existing', 'candidate'],
        )

    if 'road_class' in result.columns:
        result['road_class'] = pd.Categorical(
            result['road_class'],
            categories=[
                'service',
                'living_street',
                'road',
                'unclassified',
                'residential',
                'tertiary',
                'secondary',
                'primary',
                'trunk',
                'motorway',
            ],
            ordered=True,
        )

    return result



def prepare_candidate_sources(candidate_sites_snapped: pd.DataFrame) -> pd.DataFrame:
    """Convert snapped candidate sites to the source schema expected downstream."""
    result = ensure_xy_columns(candidate_sites_snapped)

    if 'nearest_node' not in result.columns:
        raise KeyError("candidate_sites_snapped must contain 'nearest_node'")

    if 'candidate_dist_road_estrada' not in result.columns:
        raise KeyError(
            "candidate_sites_snapped must contain 'candidate_dist_road_estrada'"
        )

    result = ensure_id_column(result, prefix='candidate')
    result = result.rename(columns={'candidate_dist_road_estrada': 'dist_snap_source'})
    result['source_type'] = 'candidate'

    keep_cols: list[str] = [
        'ID',
        'Longitude',
        'Latitude',
        'nearest_node',
        'dist_snap_source',
        'source_type',
    ]

    if 'geometry' in result.columns:
        keep_cols.append('geometry')

    result = result[keep_cols].copy()
    result = ensure_id_index_matches(result)
    result = set_known_categories(result)
    return result



def prepare_existing_sources(facilities: pd.DataFrame) -> pd.DataFrame:
    """Normalize existing facilities to the source schema used downstream."""
    result = ensure_xy_columns(facilities).copy()
    result = ensure_id_column(result, prefix='existing')
    result['source_type'] = 'existing'

    keep_cols: list[str] = [
        'ID',
        'Longitude',
        'Latitude',
        'nearest_node',
        'dist_snap_source',
        'source_type',
    ]

    if 'geometry' in result.columns:
        keep_cols.append('geometry')

    result = result[keep_cols].copy()
    result = ensure_id_index_matches(result)
    result = set_known_categories(result)
    return result



def combine_existing_and_candidate_sources(
    facilities: pd.DataFrame,
    candidate_sites_snapped: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine existing facilities with candidate facilities when candidates exist."""
    existing_sources = prepare_existing_sources(facilities)

    if candidate_sites_snapped is None:
        return existing_sources

    candidate_sources = prepare_candidate_sources(candidate_sites_snapped)
    combined = pd.concat([existing_sources, candidate_sources], axis=0, sort=False)
    combined = ensure_id_index_matches(combined)
    combined = set_known_categories(combined)
    return combined
