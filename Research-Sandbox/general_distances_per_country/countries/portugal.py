from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'PRT',
        'iso2': 'PT',
        'country_name': 'Portugal',
        'country_slug': 'portugal',
        'projected_epsg': 3763,
        'distance_threshold_km': 200.0,
        'candidate_grid_spacing_m': 5000.0,
        'candidate_max_snap_dist_m': 5000.0,
    }
)
