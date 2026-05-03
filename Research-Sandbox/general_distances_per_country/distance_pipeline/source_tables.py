from __future__ import annotations

from pathlib import Path

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



def load_custom_points_table(
    path: str | Path,
    *,
    lon_column: str | None = None,
    lat_column: str | None = None,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Load a user-provided point table from CSV, Excel, parquet, or GeoJSON."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {'.xlsx', '.xls'}:
        result = pd.read_excel(path)
    elif suffix == '.parquet':
        result = pd.read_parquet(path)
    elif suffix in {'.geojson', '.gpkg', '.shp'}:
        import geopandas as gpd

        result = gpd.read_file(path)
    else:
        result = pd.read_csv(path)

    return normalize_custom_points(
        result,
        prefix=path.stem,
        lon_column=lon_column,
        lat_column=lat_column,
        id_column=id_column,
    )



def normalize_custom_points(
    df: pd.DataFrame,
    prefix: str = 'custom',
    *,
    lon_column: str | None = None,
    lat_column: str | None = None,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Normalize a user-provided point table to the pipeline point schema.

    Accepted coordinate column names are ``Longitude``/``Latitude``, ``lon``/``lat``,
    ``lng``/``lat``, or ``x``/``y``. If a geometry column with point geometries is
    present, coordinates are derived from geometry. Missing weights are filled with 1.
    """
    result = df.copy()

    explicit_columns = {
        lon_column: 'Longitude',
        lat_column: 'Latitude',
        id_column: 'ID',
    }
    for source, target in explicit_columns.items():
        if source is None:
            continue
        if source not in result.columns:
            raise KeyError(f"Column '{source}' was requested but is not present.")
        if source != target:
            result = result.rename(columns={source: target})

    rename_pairs = [
        ('longitude', 'Longitude'),
        ('lon', 'Longitude'),
        ('lng', 'Longitude'),
        ('x', 'Longitude'),
        ('latitude', 'Latitude'),
        ('lat', 'Latitude'),
        ('y', 'Latitude'),
    ]
    lower_to_actual = {str(col).lower(): col for col in result.columns}
    for lower, target in rename_pairs:
        actual = lower_to_actual.get(lower)
        if actual is not None and target not in result.columns:
            result = result.rename(columns={actual: target})

    result = ensure_xy_columns(result)
    result = ensure_id_column(result, prefix=prefix)

    if 'population' not in result.columns:
        for candidate in ('demand', 'weight', 'headcount'):
            if candidate in result.columns:
                result['population'] = result[candidate]
                break
    if 'population' not in result.columns:
        result['population'] = 1.0

    import geopandas as gpd

    if 'geometry' in result.columns:
        gdf = gpd.GeoDataFrame(
            result,
            geometry='geometry',
            crs=getattr(df, 'crs', None),
        )
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        return gdf

    return gpd.GeoDataFrame(
        result,
        geometry=gpd.points_from_xy(result['Longitude'], result['Latitude']),
        crs='EPSG:4326',
    )



def prepare_points_as_sources(
    points: pd.DataFrame,
    *,
    source_type: str,
    id_prefix: str,
) -> pd.DataFrame:
    """Normalize any snapped point table to the source schema."""
    result = ensure_xy_columns(points).copy()
    result = ensure_id_column(result, prefix=id_prefix)

    if 'nearest_node' not in result.columns:
        raise KeyError("source points must contain 'nearest_node'")
    if 'dist_snap_source' not in result.columns:
        if 'dist_snap_target' in result.columns:
            result['dist_snap_source'] = result['dist_snap_target']
        else:
            raise KeyError("source points must contain 'dist_snap_source'")

    result['source_type'] = source_type
    keep_cols = [
        'ID',
        'Longitude',
        'Latitude',
        'nearest_node',
        'dist_snap_source',
        'source_type',
    ]
    if 'geometry' in result.columns:
        keep_cols.append('geometry')
    for optional in ('name', 'amenity', 'address', 'population'):
        if optional in result.columns and optional not in keep_cols:
            keep_cols.append(optional)

    result = result[keep_cols].copy()
    result = ensure_id_index_matches(result)
    return set_known_categories(result)



def prepare_points_as_targets(
    points: pd.DataFrame,
    *,
    id_prefix: str,
) -> pd.DataFrame:
    """Normalize any snapped point table to the target schema."""
    result = ensure_xy_columns(points).copy()
    result = ensure_id_column(result, prefix=id_prefix)

    if 'nearest_node' not in result.columns:
        raise KeyError("target points must contain 'nearest_node'")
    if 'dist_snap_target' not in result.columns:
        if 'dist_snap_source' in result.columns:
            result['dist_snap_target'] = result['dist_snap_source']
        else:
            raise KeyError("target points must contain 'dist_snap_target'")

    if 'population' not in result.columns:
        result['population'] = 1.0

    result = ensure_id_index_matches(result)
    return result
