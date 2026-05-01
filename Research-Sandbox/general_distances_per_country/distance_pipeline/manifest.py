from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from countries.base import CountryConfig
from distance_pipeline.settings import PipelineSettings


def file_metadata(path: str | Path) -> dict[str, Any]:
    """Return reproducibility metadata for a local input or output file."""
    file_path = Path(path)
    metadata: dict[str, Any] = {
        'path': str(file_path),
        'exists': file_path.exists(),
    }

    if not file_path.exists():
        return metadata

    stat = file_path.stat()
    metadata.update(
        {
            'size_bytes': stat.st_size,
            'modified_utc': datetime.fromtimestamp(
                stat.st_mtime,
                tz=UTC,
            ).isoformat(),
            'sha256': sha256_file(file_path),
        }
    )
    return metadata


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA256 checksum of a file in chunks."""
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(repo_dir: str | Path) -> str | None:
    """Return the current git commit hash when available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    commit = result.stdout.strip()
    return commit or None


def country_config_metadata(cfg: CountryConfig) -> dict[str, Any]:
    """Serialize stable country configuration fields for a manifest."""
    return {
        'iso3': cfg.iso3,
        'iso2': cfg.iso2,
        'country_name': cfg.country_name,
        'country_slug': cfg.country_slug,
        'projected_epsg': cfg.projected_epsg,
        'base_root': str(cfg.base_root),
        'base_dir': str(cfg.BASE_DIR),
        'distance_threshold_km': cfg.distance_threshold_km,
        'geofabrik_region': cfg.geofabrik_region,
        'worldpop_year': cfg.worldpop_year,
        'worldpop_suffix': cfg.worldpop_suffix,
        'worldpop_adjustment': cfg.worldpop_adjustment,
        'worldpop_filename': cfg.resolved_worldpop_filename,
        'pbf_filename': cfg.resolved_pbf_filename,
        'pbf_url': cfg.PBF_URL,
        'worldpop_url': cfg.WORLDPOP_URL,
        'boundary_source': cfg.boundary_source,
        'candidate_grid_spacing_m': cfg.candidate_grid_spacing_m,
        'candidate_exclude_water': cfg.candidate_exclude_water,
        'candidate_include_boundary': cfg.candidate_include_boundary,
        'candidate_max_snap_dist_m': cfg.candidate_max_snap_dist_m,
        'aggregate_factor': cfg.aggregate_factor,
    }


def build_run_manifest(
    *,
    cfg: CountryConfig,
    settings: PipelineSettings,
    aggregate_factor: int | None,
    amenity_values: list[str] | None,
    include_healthcare_tag: bool,
    candidate_grid_spacing_m: float | None,
    candidate_max_snap_dist_m: float | None,
    has_candidates: bool,
    output_paths: dict[str, str | Path],
    repo_dir: str | Path,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for a pipeline run."""
    input_files: dict[str, dict[str, Any]] = {
        'osm_pbf': {
            'url': cfg.PBF_URL,
            **file_metadata(cfg.PBF_PATH),
        },
        'worldpop_raster': {
            'url': cfg.WORLDPOP_URL,
            **file_metadata(cfg.WORLDPOP_PATH),
        },
    }

    return {
        'schema_version': 1,
        'created_utc': datetime.now(UTC).isoformat(),
        'pipeline_git_commit': current_git_commit(repo_dir),
        'country_config': country_config_metadata(cfg),
        'runtime_settings': asdict(settings),
        'resolved_parameters': {
            'aggregate_factor': aggregate_factor,
            'amenity_values': amenity_values,
            'include_healthcare_tag': include_healthcare_tag,
            'candidate_grid_spacing_m': candidate_grid_spacing_m,
            'candidate_max_snap_dist_m': candidate_max_snap_dist_m,
            'has_candidates': has_candidates,
        },
        'input_files': input_files,
        'outputs': {
            name: file_metadata(path)
            for name, path in output_paths.items()
        },
    }


def write_run_manifest(
    manifest: dict[str, Any],
    manifest_path: str | Path,
) -> Path:
    """Write a manifest as indented JSON."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding='utf-8',
    )
    return path
