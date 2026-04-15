from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'LAO',
        'iso2': 'LA',
        'country_name': 'Laos',
        'country_slug': 'laos',
        'projected_epsg': 32648,
        'distance_threshold_km': 300.0,
        'geofabrik_region': 'asia',
        'worldpop_filename': 'lao_ppp_2020.tif',
        'plot_title_suffix': 'roads by class, population points, and health facilities',
        'candidate_grid_spacing_m': 5000.0,
        'candidate_max_snap_dist_m': 5000.0,
    }
)
