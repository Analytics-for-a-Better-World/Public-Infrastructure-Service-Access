from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    '''Runtime settings for the distance pipeline.'''

    population_threshold: float = 1.0
    sample_fraction: float = 1.0
    max_points: int | None = None
    max_total_dist: float | None = None
    candidate_grid_spacing_m: float | None = None
    candidate_max_snap_dist_m: float | None = None
    candidate_exclude_water: bool | None = None
    deduplicate_amenities: bool = True
    force_recompute: bool = False
    verbose: bool = True
    save_context_map: bool = False
    show_context_map: bool = True
    context_map_path: Path | None = None
    context_map_dpi: int = 300
    context_map_basemap: str | None = None
    context_map_basemap_alpha: float = 0.52
    context_map_roads: bool = True
    bbox: tuple[float, float, float, float] | None = None
    matrix_output_mode: str = 'combined'
    network_backend: str = 'pyrosm'
