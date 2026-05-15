from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


SHARED_MANIFEST_SCHEMA_VERSION = "wfp-access-manifest/v1"


def make_yaml_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): make_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [make_yaml_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def file_metadata(path: str | Path, *, role: str | None = None, url: str | None = None) -> dict[str, Any]:
    file_path = Path(path)
    metadata: dict[str, Any] = {
        "path": str(file_path),
        "exists": file_path.exists(),
    }
    if role is not None:
        metadata["role"] = role
    if url is not None:
        metadata["url"] = url
    if file_path.exists():
        stat = file_path.stat()
        metadata.update(
            {
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            }
        )
    return metadata


def build_manifest(
    *,
    manifest_kind: str,
    implementation: dict[str, Any],
    cache: dict[str, Any],
    case: dict[str, Any] | None = None,
    code: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    intermediate_artifacts: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return make_yaml_safe(
        {
            "schema_version": SHARED_MANIFEST_SCHEMA_VERSION,
            "manifest_kind": manifest_kind,
            "created_utc": datetime.now(UTC).isoformat(),
            "implementation": implementation,
            "code": code or {},
            "case": case or {},
            "cache": cache,
            "inputs": inputs or {},
            "parameters": parameters or {},
            "intermediate_artifacts": intermediate_artifacts or {},
            "outputs": outputs or {},
            "diagnostics": diagnostics or {},
        }
    )


def validate_manifest_shape(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "manifest_kind",
        "created_utc",
        "implementation",
        "code",
        "case",
        "cache",
        "inputs",
        "parameters",
        "intermediate_artifacts",
        "outputs",
        "diagnostics",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Manifest missing required shared field(s): {', '.join(missing)}")
    if manifest["schema_version"] != SHARED_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema {manifest['schema_version']!r}; "
            f"expected {SHARED_MANIFEST_SCHEMA_VERSION!r}"
        )


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    validate_manifest_shape(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            make_yaml_safe(manifest),
            sort_keys=True,
            allow_unicode=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return output_path
