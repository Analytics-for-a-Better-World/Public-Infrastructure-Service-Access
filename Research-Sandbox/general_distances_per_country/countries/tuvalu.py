from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'TUV',
        'iso2': 'TV',
        'country_name': 'Tuvalu',
        'country_slug': 'tuvalu',
        'projected_epsg': 32760,
        'distance_threshold_km': 25.0,
        'geofabrik_region': 'australia-oceania',
        'worldpop_filename': 'tuv_ppp_2020.tif',
        'pbf_filename': 'tuvalu-latest.osm.pbf',
        'plot_title_suffix': 'roads by class, population points, and service facilities',
        'candidate_grid_spacing_m': 500.0,
        'candidate_max_snap_dist_m': 1000.0,
    }
)
