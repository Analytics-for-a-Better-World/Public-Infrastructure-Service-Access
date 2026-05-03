"""distance_pipeline package."""

from distance_pipeline.config_loader import load_cfg, normalize_country_code, resolve_country_module_name
from distance_pipeline.pipeline_support import (
    build_context_map_path,
    build_map_facilities,
    ensure_xy_columns,
    format_resolution_suffix,
    resolve_candidate_grid_spacing,
    resolve_candidate_max_snap_dist,
)
from distance_pipeline.routing import (
    add_edge_speeds,
    build_networkx_graph,
    directed_costs_from_matrix,
    directed_tsp_via_gurobi_sparse,
    route_between_nodes,
    sparse_costs_from_matrix,
    symmetric_tsp_via_gurobi_sparse,
)
from distance_pipeline.source_tables import (
    combine_existing_and_candidate_sources,
    ensure_id_column,
    ensure_id_index_matches,
    prepare_candidate_sources,
    prepare_existing_sources,
    set_known_categories,
)

__all__ = [
    'add_edge_speeds',
    'build_networkx_graph',
    'build_context_map_path',
    'build_map_facilities',
    'combine_existing_and_candidate_sources',
    'directed_costs_from_matrix',
    'directed_tsp_via_gurobi_sparse',
    'ensure_id_column',
    'ensure_id_index_matches',
    'ensure_xy_columns',
    'format_resolution_suffix',
    'load_cfg',
    'normalize_country_code',
    'prepare_candidate_sources',
    'prepare_existing_sources',
    'resolve_candidate_grid_spacing',
    'resolve_candidate_max_snap_dist',
    'resolve_country_module_name',
    'route_between_nodes',
    'set_known_categories',
    'sparse_costs_from_matrix',
    'symmetric_tsp_via_gurobi_sparse',
]
