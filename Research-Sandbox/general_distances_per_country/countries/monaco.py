from countries.base import build_config


CFG = build_config(
    {
        'iso3': 'MCO',
        'iso2': 'MC',
        'country_name': 'Monaco',
        'country_slug': 'monaco',
        'projected_epsg': 32632,
        'distance_threshold_km': 25.0,
        'geofabrik_region': 'europe',
        'worldpop_filename': 'mco_ppp_2020.tif',
        'pbf_filename': 'monaco-latest.osm.pbf',
        'plot_title_suffix': 'roads by class, population points, and service facilities',
        'candidate_grid_spacing_m': 500.0,
        'candidate_max_snap_dist_m': 1000.0,
    }
)
