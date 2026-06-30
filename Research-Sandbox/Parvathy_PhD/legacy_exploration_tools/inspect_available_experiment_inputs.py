from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def safe_read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - diagnostic helper
        return {"_error": str(exc)}


def file_info(path: Path) -> dict:
    info = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }
    if path.exists() and path.suffix.lower() == ".parquet":
        try:
            frame = pd.read_parquet(path)
            info.update(
                {
                    "rows": int(len(frame)),
                    "columns": list(frame.columns),
                    "head": frame.head(3).to_dict(orient="records"),
                }
            )
        except Exception as exc:
            info["read_error"] = str(exc)
    return info


def resolved_params(manifest: dict) -> dict:
    return manifest.get("resolved_parameters") or manifest.get("parameters", {}).get("resolved", {})


def summarize_manifests(outputs_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(outputs_dir.glob("run_manifest*.yaml")):
        manifest = safe_read_yaml(path)
        params = resolved_params(manifest)
        runtime = manifest.get("runtime_settings") or manifest.get("parameters", {}).get("runtime_settings", {})
        outputs = manifest.get("outputs", {})
        rows.append(
            {
                "manifest": str(path),
                "has_candidates": params.get("has_candidates"),
                "candidate_grid_spacing_m": params.get("candidate_grid_spacing_m"),
                "candidate_max_snap_dist_m": params.get("candidate_max_snap_dist_m"),
                "max_total_dist": runtime.get("max_total_dist"),
                "aggregate_factor": params.get("aggregate_factor"),
                "outputs": {
                    key: str(value.get("path")) if isinstance(value, dict) else str(value)
                    for key, value in outputs.items()
                },
            }
        )
    return rows


def summarize_folder(label: str, outputs_dir: Path) -> dict:
    parquet_files = sorted(outputs_dir.glob("*.parquet"))
    interesting = [
        path
        for path in parquet_files
        if path.name.startswith(("population", "sources", "existing_sources", "distance_matrix"))
    ]
    return {
        "label": label,
        "outputs_dir": str(outputs_dir),
        "manifests": summarize_manifests(outputs_dir),
        "parquets": [file_info(path) for path in interesting],
    }


def main() -> None:
    report = {
        "timor_leste": summarize_folder(
            "Timor-Leste",
            ROOT / "runs" / "TimorLeste_20260618_220002" / "east-timor_data" / "outputs",
        ),
        "vietnam": summarize_folder(
            "Vietnam",
            ROOT / "runs" / "vietnam_20260619_0630" / "vietnam_data" / "outputs",
        ),
    }
    out_dir = ROOT / "outputs" / "article_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "available_experiment_inputs.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"written": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
