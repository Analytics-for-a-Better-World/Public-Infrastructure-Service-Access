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
    force_recompute: bool = False
    verbose: bool = True
    save_context_map: bool = False
    show_context_map: bool = True
    context_map_path: Path | None = None
    context_map_dpi: int = 300
