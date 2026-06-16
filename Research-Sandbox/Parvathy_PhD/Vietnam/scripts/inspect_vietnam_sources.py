from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SHARE_ROOT = Path(r"C:\Users\joaqu\OneDrive - UvA\share\Vietnam")
OUTPUT_ROOT = Path(r"C:\local\Parvathy\Vietnam")


def file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def main() -> None:
    facility = SHARE_ROOT / "stroke-facs-100-en.xlsx"
    summary = {
        "data_rule": "fresh PISA outputs only; Fleur npy/pickle files are intentionally ignored",
        "share_root": str(SHARE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "facility_table": file_info(facility),
        "fresh_pipeline_parquets": [],
        "built_instances": [],
        "experiment_outputs": [],
    }
    if facility.exists():
        df = pd.read_excel(facility)
        summary["facility_table"].update({
            "rows": int(len(df)),
            "columns": list(df.columns),
            "latitude_nonnull": int(df["latitude"].notna().sum()) if "latitude" in df else None,
            "longitude_nonnull": int(df["longitude"].notna().sum()) if "longitude" in df else None,
        })
    for path in sorted((OUTPUT_ROOT / "fresh_downloads").glob("**/*.parquet")) if (OUTPUT_ROOT / "fresh_downloads").exists() else []:
        summary["fresh_pipeline_parquets"].append(file_info(path))
    for path in sorted((OUTPUT_ROOT / "optimization").glob("*.npz")) if (OUTPUT_ROOT / "optimization").exists() else []:
        info = file_info(path)
        meta = path.with_suffix(".metadata.json")
        if meta.exists():
            try:
                info["metadata"] = json.loads(meta.read_text(encoding="utf-8"))
            except Exception as exc:
                info["metadata_error"] = str(exc)
        summary["built_instances"].append(info)
    for path in sorted((OUTPUT_ROOT / "grasp_latest").glob("*")) if (OUTPUT_ROOT / "grasp_latest").exists() else []:
        if path.is_file():
            summary["experiment_outputs"].append(file_info(path))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
