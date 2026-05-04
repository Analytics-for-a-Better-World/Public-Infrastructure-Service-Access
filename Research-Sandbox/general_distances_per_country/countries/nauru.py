from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'NRU',
        'iso2': 'NR',
        'country_name': 'Nauru',
        'country_slug': 'nauru',
        'projected_epsg': 32758,
        'distance_threshold_km': 25.0,
        'geofabrik_region': 'australia-oceania',
        'worldpop_filename': 'nru_ppp_2020.tif',
        'pbf_filename': 'nauru-latest.osm.pbf',
        'plot_title_suffix': 'roads by class, population points, and service facilities',
        'candidate_grid_spacing_m': 500.0,
        'candidate_max_snap_dist_m': 1000.0,
    }
)
