from collections.abc import Callable
from pathlib import Path
import pickle
from time import perf_counter as pc

from countries.base import CountryConfig


def _none_or_number(value: float | int | None, suffix: str = '') -> str:
    """Format an optional numeric cache-key component."""
    if value is None:
        return 'none'
    return f'{value:g}{suffix}'


def _none_or_int(value: int | None) -> str:
    """Format an optional integer cache-key component."""
    if value is None:
        return 'none'
    return str(value)


def _amenity_part(amenity_values: list[str] | None) -> str:
    """Format amenity filters for cache-key filenames."""
    if amenity_values is None:
        return 'all'
    return '_'.join(sorted(amenity_values))


def _load_pickle[T](cache_path: Path) -> T:
    """Load a pickled object from disk."""
    with cache_path.open('rb') as file:
        return pickle.load(file)


def _save_pickle(cache_path: Path, obj: object) -> None:
    """Save a pickled object to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def _timed_cached_call[T](
    cache_path: Path,
    builder: Callable[[], T],
    force_recompute: bool = False,
    verbose: bool = True,
) -> T:
    """Return a cached object if available, otherwise build it and cache it."""
    t0: float = pc()

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_recompute:
        if verbose:
            print(f'Loading cache: {cache_path.name}')

        t_load_start: float = pc()
        with cache_path.open('rb') as file:
            obj: T = pickle.load(file)
        t_load_end: float = pc()

        if verbose:
            print(f'Load time: {t_load_end - t_load_start:.2f}s')
            print(f'Total time: {t_load_end - t0:.2f}s')

        return obj

    if verbose:
        action: str = 'Rebuilding' if force_recompute and cache_path.exists() else 'Building'
        print(f'{action} cache: {cache_path.name}')

    t_build_start: float = pc()
    obj = builder()
    t_build_end: float = pc()

    if verbose:
        print(f'Build time: {t_build_end - t_build_start:.2f}s')

    t_save_start: float = pc()
    with cache_path.open('wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    t_save_end: float = pc()

    if verbose:
        print(f'Save time: {t_save_end - t_save_start:.2f}s')
        print(f'Total time: {t_save_end - t0:.2f}s')

    return obj


class CacheManager:
    """Manage cache paths and cached execution for a given country configuration."""

    def __init__(
        self,
        cfg: CountryConfig,
        force_recompute: bool = False,
        verbose: bool = True,
    ) -> None:
        self.cfg: CountryConfig = cfg
        self.force_recompute: bool = force_recompute
        self.verbose: bool = verbose

    @property
    def cache_dir(self) -> Path:
        return self.cfg.BASE_DIR / 'cache'

    @property
    def figures_dir(self) -> Path:
        return self.cfg.BASE_DIR / 'figures'

    @property
    def pbf_stem(self) -> str:
        return self.cfg.PBF_PATH.stem

    @property
    def worldpop_stem(self) -> str:
        return self.cfg.WORLDPOP_PATH.stem

    def nodes_path(self) -> Path:
        return self.cache_dir / f'{self.pbf_stem}_nodes.pkl'

    def boundaries_dir(self) -> Path:
        return self.cache_dir / 'boundaries'

    def boundary_archive_path(self) -> Path:
        return self.boundaries_dir() / 'ne_10m_admin_0_countries.zip'

    def country_boundary_path(self) -> Path:
        return self.cache_dir / f'{self.cfg.iso3.lower()}_country_boundary_epsg_{self.cfg.PROJECTED_EPSG}.pkl'

    def water_bodies_path(self) -> Path:
        return self.cache_dir / f'{self.pbf_stem}_water_bodies_epsg_{self.cfg.PROJECTED_EPSG}.pkl'

    def candidate_sites_path(
        self,
        grid_spacing_m: float,
        exclude_water: bool,
        include_boundary: bool,
    ) -> Path:
        water_part = 'no_water' if exclude_water else 'water_allowed'
        boundary_part = 'include_boundary' if include_boundary else 'strict_interior'
        return (
            self.cache_dir
            / (
                f'{self.cfg.iso3.lower()}_candidate_sites_'
                f'spacing_{grid_spacing_m:g}m_'
                f'{water_part}_{boundary_part}_'
                f'epsg_{self.cfg.PROJECTED_EPSG}.pkl'
            )
        )

    def candidate_sites_snapped_path(
        self,
        grid_spacing_m: float,
        exclude_water: bool,
        include_boundary: bool,
        distance_col: str,
        max_snap_dist_m: float | None,
    ) -> Path:
        water_part = 'no_water' if exclude_water else 'water_allowed'
        boundary_part = 'include_boundary' if include_boundary else 'strict_interior'
        max_snap_part = 'none' if max_snap_dist_m is None else f'{max_snap_dist_m:g}m'
        return (
            self.cache_dir
            / (
                f'{self.cfg.iso3.lower()}_candidate_sites_snapped_'
                f'spacing_{grid_spacing_m:g}m_'
                f'{water_part}_{boundary_part}_'
                f'{distance_col}_max_snap_{max_snap_part}_'
                f'epsg_{self.cfg.PROJECTED_EPSG}.pkl'
            )
        )

    def edges_path(self) -> Path:
        return self.cache_dir / f'{self.pbf_stem}_edges.pkl'

    def roads_path(self) -> Path:
        return self.cache_dir / f'{self.pbf_stem}_roads.pkl'

    def facilities_path(
        self,
        amenity_values: list[str] | None = None,
    ) -> Path:
        amenity_part = _amenity_part(amenity_values)
        return self.cache_dir / f'{self.pbf_stem}_facilities_{amenity_part}.pkl'

    def health_facilities_path(
        self,
        amenity_values: list[str] | None = None,
        include_healthcare_tag: bool | None = None,
    ) -> Path:
        return self.facilities_path(
            amenity_values=amenity_values,
        )

    def facility_points_path(
        self,
        amenity_values: list[str] | None = None,
        deduplicate_amenities: bool = True,
    ) -> Path:
        amenity_part = _amenity_part(amenity_values)
        dedup_part = 'dedup_v1' if deduplicate_amenities else 'raw'
        return (
            self.cache_dir
            / (
                f'{self.pbf_stem}_facility_points_{amenity_part}_'
                f'epsg_{self.cfg.PROJECTED_EPSG}_{dedup_part}.pkl'
            )
        )

    def health_facilities_points_path(
        self,
        amenity_values: list[str] | None = None,
        include_healthcare_tag: bool | None = None,
    ) -> Path:
        return self.facility_points_path(
            amenity_values=amenity_values,
            deduplicate_amenities=True,
        )

    def population_points_path(
        self,
        population_threshold: float,
        sample_fraction: float,
        max_points: int | None,
        aggregate_factor: int | None,
    ) -> Path:
        max_points_str: str = 'none' if max_points is None else str(max_points)
        aggregate_factor_str: str = 'none' if aggregate_factor is None else str(aggregate_factor)
        return (
            self.cache_dir
            / (
                f'{self.worldpop_stem}_population_points_'
                f'pop_{population_threshold:g}_'
                f'sample_{sample_fraction:g}_'
                f'agg_{aggregate_factor_str}_'
                f'max_{max_points_str}.pkl'
            )
        )

    def population_snapped_path(self, distance_col: str) -> Path:
        return self.population_snapped_path_for(
            distance_col=distance_col,
            population_threshold=None,
            sample_fraction=None,
            max_points=None,
            aggregate_factor=None,
        )

    def population_snapped_path_for(
        self,
        distance_col: str,
        population_threshold: float | None,
        sample_fraction: float | None,
        max_points: int | None,
        aggregate_factor: int | None,
    ) -> Path:
        population_part = (
            f'pop_{_none_or_number(population_threshold)}_'
            f'sample_{_none_or_number(sample_fraction)}_'
            f'agg_{_none_or_int(aggregate_factor)}_'
            f'max_{_none_or_int(max_points)}'
        )
        return (
            self.cache_dir
            / (
                f'{self.worldpop_stem}_population_snapped_'
                f'{population_part}_{distance_col}_'
                f'epsg_{self.cfg.PROJECTED_EPSG}.pkl'
            )
        )

    def sources_snapped_path(self, distance_col: str) -> Path:
        return self.sources_snapped_path_for(
            distance_col=distance_col,
            amenity_values=None,
        )

    def sources_snapped_path_for(
        self,
        distance_col: str,
        amenity_values: list[str] | None,
    ) -> Path:
        amenity_part = _amenity_part(amenity_values)
        return (
            self.cache_dir
            / (
                f'{self.pbf_stem}_sources_snapped_'
                f'{amenity_part}_'
                f'{distance_col}_epsg_{self.cfg.PROJECTED_EPSG}.pkl'
            )
        )

    def hospitals_snapped_path(self, distance_col: str) -> Path:
        return self.sources_snapped_path(distance_col=distance_col)

    def hospitals_snapped_path_for(
        self,
        distance_col: str,
        amenity_values: list[str] | None,
        include_healthcare_tag: bool | None = None,
    ) -> Path:
        return self.sources_snapped_path_for(
            distance_col=distance_col,
            amenity_values=amenity_values,
        )

    def distance_matrix_path(
        self,
        distance_threshold_largest: float,
        max_total_dist: float | None = None,
    ) -> Path:
        return self.distance_matrix_path_for(
            distance_threshold_largest=distance_threshold_largest,
            max_total_dist=max_total_dist,
            population_threshold=None,
            sample_fraction=None,
            max_points=None,
            aggregate_factor=None,
            amenity_values=None,
            candidate_grid_spacing_m=None,
            candidate_max_snap_dist_m=None,
            has_candidates=False,
        )

    def distance_matrix_path_for(
        self,
        distance_threshold_largest: float,
        max_total_dist: float | None = None,
        population_threshold: float | None = None,
        sample_fraction: float | None = None,
        max_points: int | None = None,
        aggregate_factor: int | None = None,
        amenity_values: list[str] | None = None,
        candidate_grid_spacing_m: float | None = None,
        candidate_max_snap_dist_m: float | None = None,
        has_candidates: bool = False,
        include_healthcare_tag: bool | None = None,
    ) -> Path:
        max_total_dist_str = _none_or_number(max_total_dist, 'm')
        population_part = (
            f'pop_{_none_or_number(population_threshold)}_'
            f'sample_{_none_or_number(sample_fraction)}_'
            f'agg_{_none_or_int(aggregate_factor)}_'
            f'max_{_none_or_int(max_points)}'
        )
        amenity_part = _amenity_part(amenity_values)
        candidate_part = (
            'candidates_'
            f'spacing_{_none_or_number(candidate_grid_spacing_m, "m")}_'
            f'max_snap_{_none_or_number(candidate_max_snap_dist_m, "m")}'
            if has_candidates
            else 'no_candidates'
        )
        return (
            self.cache_dir
            / (
                f'{self.pbf_stem}_distance_matrix_'
                f'threshold_{distance_threshold_largest:g}km_'
                f'max_total_{max_total_dist_str}_'
                f'{population_part}_'
                f'{amenity_part}_'
                f'{candidate_part}.pkl'
            )
        )

    def context_map_path(self, suffix: str = 'context_map', ext: str = 'png') -> Path:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        return self.figures_dir / f'{self.pbf_stem}_{suffix}.{ext}'

    def run[T](
        self,
        cache_path: Path,
        builder: Callable[[], T],
    ) -> T:
        return _timed_cached_call(
            cache_path=cache_path,
            builder=builder,
            force_recompute=self.force_recompute,
            verbose=self.verbose,
        )

    def load_or_build_network_data(
        self,
        builder: Callable[[], tuple[object, object]],
    ) -> tuple[object, object]:
        """Load cached nodes and edges, or build and cache both in a single pass."""
        nodes_path = self.nodes_path()
        edges_path = self.edges_path()
        t0: float = pc()

        can_load = (
            nodes_path.exists()
            and edges_path.exists()
            and not self.force_recompute
        )

        if can_load:
            if self.verbose:
                print(f'Loading cache: {nodes_path.name}')
                print(f'Loading cache: {edges_path.name}')

            t_load_start: float = pc()
            nodes = _load_pickle(nodes_path)
            edges = _load_pickle(edges_path)
            t_load_end: float = pc()

            if self.verbose:
                print(f'Load time: {t_load_end - t_load_start:.2f}s')
                print(f'Total time: {t_load_end - t0:.2f}s')

            return nodes, edges

        if self.verbose:
            action = 'Rebuilding' if self.force_recompute and (nodes_path.exists() or edges_path.exists()) else 'Building'
            print(f'{action} cache: {nodes_path.name}, {edges_path.name}')

        t_build_start: float = pc()
        nodes, edges = builder()
        t_build_end: float = pc()

        if self.verbose:
            print(f'Build time: {t_build_end - t_build_start:.2f}s')

        t_save_start: float = pc()
        _save_pickle(nodes_path, nodes)
        _save_pickle(edges_path, edges)
        t_save_end: float = pc()

        if self.verbose:
            print(f'Save time: {t_save_end - t_save_start:.2f}s')
            print(f'Total time: {t_save_end - t0:.2f}s')

        return nodes, edges
