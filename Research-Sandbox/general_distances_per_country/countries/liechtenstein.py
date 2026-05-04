from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'LIE',
        'iso2': 'LI',
        'country_name': 'Liechtenstein',
        'country_slug': 'liechtenstein',
        'projected_epsg': 32632,
        'distance_threshold_km': 25.0,
        'geofabrik_region': 'europe',
        'worldpop_filename': 'lie_ppp_2020.tif',
        'pbf_filename': 'liechtenstein-latest.osm.pbf',
        'plot_title_suffix': 'roads by class, population points, and service facilities',
        'candidate_grid_spacing_m': 500.0,
        'candidate_max_snap_dist_m': 1000.0,
    }
)
