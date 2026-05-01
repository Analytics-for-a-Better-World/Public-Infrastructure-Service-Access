from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'TLS',
        'iso2': 'TL',
        'country_name': 'Timor Leste',
        'country_slug': 'east-timor',
        'projected_epsg': 32751,
        'distance_threshold_km': 300.0,
        'geofabrik_region': 'asia',
        'worldpop_filename': 'tls_ppp_2020.tif',
        'plot_title_suffix': 'roads by class, population points, and service facilities',
        'candidate_grid_spacing_m': 5000.0,
        'candidate_max_snap_dist_m': 5000.0,
    }
)
