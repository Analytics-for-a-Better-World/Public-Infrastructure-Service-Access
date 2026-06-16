from collections.abc import Callable
import hashlib
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
    raw = '_'.join(sorted(amenity_values))
    if len(raw) <= 120:
        return raw
    digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]
    return f'{raw[:80]}_{digest}'


def _safe_part(value: object) -> str:
    """Format an arbitrary cache-key component for filenames."""
    return str(value).replace('-', 'm').replace('.', 'p').replace('+', '')


def _bbox_part(bbox: tuple[float, float, float, float] | list[float] | None) -> str:
    """Format an optional lon/lat bbox for cache-key filenames."""
    if bbox is None:
        return ''
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        f'_bbox_{_safe_part(min_lon)}_{_safe_part(min_lat)}_'
        f'{_safe_part(max_lon)}_{_safe_part(max_lat)}'
    )


def _backend_part(network_backend: str | None) -> str:
    """Format an optional OSM network backend for cache-key filenames."""
    if network_backend in (None, '', 'pyrosm'):
        return ''
    return f'_backend_{_safe_part(network_backend)}'


def _snap_components_part(snap_components: tuple[int, ...] | None) -> str:
    """Format optional allowed snapping component IDs for cache-key filenames."""
    if snap_components is None:
        return ''
    joined = '-'.join(str(component_id) for component_id in snap_components)
    return f'_snap_components_{joined}'


def _path_with_short_cache_name(cache_path: Path) -> Path:
    """Shorten very long cache filenames while preserving deterministic identity."""
    path_text = str(cache_path)
    if len(path_text) <= 240 and len(cache_path.name) <= 180:
        return cache_path

    digest = hashlib.sha1(path_text.encode('utf-8')).hexdigest()[:12]
    suffix = cache_path.suffix
    suffix_len = len(suffix)
    max_name_len = 120
    stem_limit = max_name_len - suffix_len - len(digest) - 1
    readable_stem = cache_path.stem[:max(24, stem_limit)]
    return cache_path.with_name(f'{readable_stem}_{digest}{suffix}')


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

    def nodes_path(
        self,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        network_backend: str | None = None,
    ) -> Path:
        return (
            self.cache_dir
            / f'{self.pbf_stem}_nodes{_bbox_part(bbox)}{_backend_part(network_backend)}.pkl'
        )

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
        snap_components: tuple[int, ...] | None = None,
    ) -> Path:
        water_part = 'no_water' if exclude_water else 'water_allowed'
        boundary_part = 'include_boundary' if include_boundary else 'strict_interior'
        max_snap_part = 'none' if max_snap_dist_m is None else f'{max_snap_dist_m:g}m'
        snap_part = _snap_components_part(snap_components)
        return (
            self.cache_dir
            / (
                f'{self.cfg.iso3.lower()}_candidate_sites_snapped_'
                f'spacing_{grid_spacing_m:g}m_'
                f'{water_part}_{boundary_part}_'
                f'{distance_col}_max_snap_{max_snap_part}_'
                f'epsg_{self.cfg.PROJECTED_EPSG}{snap_part}.pkl'
            )
        )

    def edges_path(
        self,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        network_backend: str | None = None,
    ) -> Path:
        return (
            self.cache_dir
            / f'{self.pbf_stem}_edges{_bbox_part(bbox)}{_backend_part(network_backend)}.pkl'
        )

    def roads_path(
        self,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        network_backend: str | None = None,
    ) -> Path:
        return (
            self.cache_dir
            / f'{self.pbf_stem}_roads{_bbox_part(bbox)}{_backend_part(network_backend)}.pkl'
        )

    def facilities_path(
        self,
        amenity_values: list[str] | None = None,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        network_backend: str | None = None,
    ) -> Path:
        amenity_part = _amenity_part(amenity_values)
        return (
            self.cache_dir
            / (
                f'{self.pbf_stem}_facilities_{amenity_part}'
                f'{_bbox_part(bbox)}{_backend_part(network_backend)}.pkl'
            )
        )

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
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        network_backend: str | None = None,
    ) -> Path:
        amenity_part = _amenity_part(amenity_values)
        dedup_part = 'dedup_v1' if deduplicate_amenities else 'raw'
        return (
            self.cache_dir
            / (
                f'{self.pbf_stem}_facility_points_{amenity_part}_'
                f'epsg_{self.cfg.PROJECTED_EPSG}_{dedup_part}'
                f'{_bbox_part(bbox)}{_backend_part(network_backend)}.pkl'
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
        random_seed: int = 42,
        aggregate_factor: int | None = None,
    ) -> Path:
        max_points_str: str = 'none' if max_points is None else str(max_points)
        aggregate_factor_str: str = 'none' if aggregate_factor is None else str(aggregate_factor)
        return (
            self.cache_dir
            / (
                f'{self.worldpop_stem}_population_points_'
                f'pop_{population_threshold:g}_'
                f'sample_{sample_fraction:g}_'
                f'seed_{random_seed}_'
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
            random_seed=None,
            aggregate_factor=None,
        )

    def population_snapped_path_for(
        self,
        distance_col: str,
        population_threshold: float | None,
        sample_fraction: float | None,
        max_points: int | None,
        random_seed: int | None = None,
        aggregate_factor: int | None = None,
        snap_components: tuple[int, ...] | None = None,
        network_backend: str | None = None,
    ) -> Path:
        population_part = (
            f'pop_{_none_or_number(population_threshold)}_'
            f'sample_{_none_or_number(sample_fraction)}_'
            f'seed_{_none_or_int(random_seed)}_'
            f'agg_{_none_or_int(aggregate_factor)}_'
            f'max_{_none_or_int(max_points)}'
        )
        snap_part = _snap_components_part(snap_components)
        backend_part = _backend_part(network_backend)
        return (
            self.cache_dir
            / (
                f'{self.worldpop_stem}_population_snapped_'
                f'{population_part}_{distance_col}_'
                f'epsg_{self.cfg.PROJECTED_EPSG}{snap_part}{backend_part}.pkl'
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
        snap_components: tuple[int, ...] | None = None,
        network_backend: str | None = None,
    ) -> Path:
        amenity_part = _amenity_part(amenity_values)
        snap_part = _snap_components_part(snap_components)
        backend_part = _backend_part(network_backend)
        return (
            self.cache_dir
            / (
                f'{self.pbf_stem}_sources_snapped_'
                f'{amenity_part}_'
                f'{distance_col}_epsg_{self.cfg.PROJECTED_EPSG}{snap_part}{backend_part}.pkl'
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
            random_seed=None,
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
        random_seed: int | None = None,
        aggregate_factor: int | None = None,
        amenity_values: list[str] | None = None,
        candidate_grid_spacing_m: float | None = None,
        candidate_max_snap_dist_m: float | None = None,
        has_candidates: bool = False,
        include_healthcare_tag: bool | None = None,
        snap_components: tuple[int, ...] | None = None,
        network_backend: str | None = None,
    ) -> Path:
        max_total_dist_str = _none_or_number(max_total_dist, 'm')
        population_part = (
            f'pop_{_none_or_number(population_threshold)}_'
            f'sample_{_none_or_number(sample_fraction)}_'
            f'seed_{_none_or_int(random_seed)}_'
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
        snap_part = _snap_components_part(snap_components)
        backend_part = _backend_part(network_backend)
        return (
            self.cache_dir
            / (
                f'{self.pbf_stem}_distance_matrix_'
                f'threshold_{distance_threshold_largest:g}km_'
                f'max_total_{max_total_dist_str}_'
                f'{population_part}_'
                f'{amenity_part}_'
                f'{candidate_part}{snap_part}{backend_part}.pkl'
            )
        )

    def node_pair_distances_dir(
        self,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        network_backend: str | None = None,
        cost_profile: str = 'length',
    ) -> Path:
        """Return the reusable road-node-pair distance cache directory."""
        network_part = (
            f'{self.pbf_stem}{_bbox_part(bbox)}{_backend_part(network_backend)}'
        )
        return (
            self.cache_dir
            / 'node_pair_distances'
            / f'{network_part}_cost_{_safe_part(cost_profile)}'
        )

    def context_map_path(self, suffix: str = 'context_map', ext: str = 'png') -> Path:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        return self.figures_dir / f'{self.pbf_stem}_{suffix}.{ext}'

    def run[T](
        self,
        cache_path: Path,
        builder: Callable[[], T],
    ) -> T:
        original_cache_path = cache_path
        cache_path = _path_with_short_cache_name(cache_path)
        if self.verbose and cache_path != original_cache_path:
            print(
                'Shortened cache filename for Windows path limit: '
                f'{original_cache_path.name} -> {cache_path.name}'
            )
        return _timed_cached_call(
            cache_path=cache_path,
            builder=builder,
            force_recompute=self.force_recompute,
            verbose=self.verbose,
        )

    def load_or_build_network_data(
        self,
        builder: Callable[[], tuple[object, object]],
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        network_backend: str | None = None,
    ) -> tuple[object, object]:
        """Load cached nodes and edges, or build and cache both in a single pass."""
        nodes_path = self.nodes_path(bbox=bbox, network_backend=network_backend)
        edges_path = self.edges_path(bbox=bbox, network_backend=network_backend)
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
