from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'POL',
        'iso2': 'PL',
        'country_name': 'Poland',
        'country_slug': 'poland',
        'projected_epsg': 2180,
        'distance_threshold_km': 150.0,
        'geofabrik_region': 'europe',
        'candidate_grid_spacing_m': 10000.0,
        'candidate_max_snap_dist_m': 5000.0,
    }
)
