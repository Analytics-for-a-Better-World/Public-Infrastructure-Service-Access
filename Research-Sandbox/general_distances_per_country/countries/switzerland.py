from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'CHE',
        'iso2': 'CH',
        'country_name': 'Switzerland',
        'country_slug': 'switzerland',
        'projected_epsg': 2056,
        'distance_threshold_km': 150.0,
        'geofabrik_region': 'europe',
        'candidate_grid_spacing_m': 10000.0,
        'candidate_max_snap_dist_m': 5000.0,
    }
)
