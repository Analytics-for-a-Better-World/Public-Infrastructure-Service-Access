from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'VNM',
        'iso2': 'VN',
        'country_name': 'Vietnam',
        'country_slug': 'vietnam',
        'projected_epsg': 3405,
        'distance_threshold_km': 100.0,
        'geofabrik_region': 'asia',
        'worldpop_filename': 'vnm_ppp_2020.tif',
        'plot_title_suffix': 'roads by class, population points, and health facilities',
        'candidate_grid_spacing_m': 10000.0,
        'candidate_max_snap_dist_m': 5000.0,
        'candidate_exclude_water': False,
        'aggregate_factor': 10,
    }
)
