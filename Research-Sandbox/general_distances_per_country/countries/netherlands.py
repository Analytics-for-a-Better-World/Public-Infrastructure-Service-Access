from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'NLD',
        'iso2': 'NL',
        'country_name': 'Netherlands',
        'country_slug': 'netherlands',
        'projected_epsg': 28992,
        'distance_threshold_km': 150.0,
        'candidate_grid_spacing_m': 5000.0,
        'candidate_max_snap_dist_m': 5000.0,
    }
)
