from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'LUX',
        'iso2': 'LU',
        'country_name': 'Luxembourg',
        'country_slug': 'luxembourg',
        'projected_epsg': 32632,
        'distance_threshold_km': 300.0,
        'geofabrik_region': 'europe',
        'worldpop_filename': 'lux_ppp_2020.tif',
        'plot_title_suffix': 'roads by class, population points, and health facilities',
        'candidate_grid_spacing_m': 5000.0,
        'candidate_max_snap_dist_m': 5000.0,
    }
)
