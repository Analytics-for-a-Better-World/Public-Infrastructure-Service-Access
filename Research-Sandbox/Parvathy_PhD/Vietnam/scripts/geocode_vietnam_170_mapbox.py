from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reference_cache" / "data" / "vietnam"

RAW_170 = BASE / "vietnam_stroke_centers_170_vi_vnsa_2025_09_extracted_raw.csv"
GOOGLE_PIPELINE = BASE / "vietnam_stroke_centers_170_vi_vnsa_2025_09_google_geocoded_pipeline_source.csv"

MAPBOX_COMPACT = BASE / "vietnam_stroke_centers_170_vi_vnsa_2025_09_mapbox_permanent_geocoded_compact.csv"
MAPBOX_PIPELINE = BASE / "vietnam_stroke_centers_170_vi_vnsa_2025_09_mapbox_permanent_geocoded_pipeline_source.csv"
MAPBOX_VALIDATION_CSV = BASE / "vietnam_stroke_centers_170_vi_vnsa_2025_09_mapbox_vs_google_validation.csv"
MAPBOX_SUMMARY_JSON = BASE / "vietnam_stroke_centers_170_vi_vnsa_2025_09_mapbox_permanent_geocoding_summary.json"
MAPBOX_NOTES_MD = BASE / "vietnam_stroke_centers_170_vi_vnsa_2025_09_mapbox_permanent_geocoding_notes.md"
MAPBOX_RESPONSE_CACHE = BASE / "vietnam_stroke_centers_170_vi_vnsa_2025_09_mapbox_permanent_geocoding_cache.json"


VIETNAM_BOUNDS = {
    "min_lon": 102.0,
    "max_lon": 110.0,
    "min_lat": 8.0,
    "max_lat": 24.0,
}


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def haversine_m(lat1: float | None, lon1: float | None, lat2: float | None, lon2: float | None) -> float | None:
    if None in (lat1, lon1, lat2, lon2):
        return None
    radius = 6_371_000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fnum(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except Exception:
        return None


def inside_vietnam_bounds(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    return (
        VIETNAM_BOUNDS["min_lat"] <= lat <= VIETNAM_BOUNDS["max_lat"]
        and VIETNAM_BOUNDS["min_lon"] <= lon <= VIETNAM_BOUNDS["max_lon"]
    )


def distance_flag(distance_m: float | None) -> str:
    if distance_m is None:
        return "no_comparison"
    if distance_m <= 250:
        return "agree_<=250m"
    if distance_m <= 1_000:
        return "near_250m_1km"
    if distance_m <= 5_000:
        return "review_1_5km"
    if distance_m <= 20_000:
        return "large_5_20km"
    return "major_>20km"


def build_query(row: dict[str, str]) -> str:
    parts = [
        row.get("ten_benh_vien", ""),
        row.get("dia_chi", ""),
        row.get("tinh_thanh_pho", ""),
        "Viet Nam",
    ]
    return ", ".join(part.strip() for part in parts if part and part.strip())


def classify_mapbox(row: dict[str, Any]) -> str:
    if not row.get("mapbox_status") == "ok":
        return "failed"
    lat = fnum(row.get("mapbox_lat"))
    lon = fnum(row.get("mapbox_lon"))
    relevance = fnum(row.get("mapbox_relevance"))
    if not inside_vietnam_bounds(lat, lon):
        return "poor_outside_vietnam"
    if relevance is not None and relevance < 0.65:
        return "poor_low_relevance"
    feature_type = str(row.get("mapbox_feature_type") or "")
    if feature_type and feature_type not in {"poi", "address", "place", "locality", "neighborhood"}:
        return "review_feature_type"
    if relevance is not None and relevance >= 0.85:
        return "high"
    return "medium"


def mapbox_forward(token: str, query: str, *, permanent: bool, language: str, country: str, limit: int) -> dict[str, Any]:
    params = {
        "q": query,
        "limit": str(limit),
        "language": language,
        "country": country,
        "permanent": "true" if permanent else "false",
        "access_token": token,
    }
    url = "https://api.mapbox.com/search/geocode/v6/forward?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "Parvathy-PhD-replication/2026"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten_feature(feature: dict[str, Any] | None) -> dict[str, Any]:
    if not feature:
        return {
            "mapbox_status": "no_result",
            "mapbox_lat": "",
            "mapbox_lon": "",
            "mapbox_name": "",
            "mapbox_full_address": "",
            "mapbox_feature_type": "",
            "mapbox_relevance": "",
            "mapbox_accuracy": "",
            "mapbox_match_code": "",
            "mapbox_id": "",
        }
    properties = feature.get("properties") or {}
    coordinates = properties.get("coordinates") or {}
    geometry = feature.get("geometry") or {}
    geom_coordinates = geometry.get("coordinates") or []
    lon = coordinates.get("longitude")
    lat = coordinates.get("latitude")
    if (lon is None or lat is None) and isinstance(geom_coordinates, list) and len(geom_coordinates) >= 2:
        lon, lat = geom_coordinates[0], geom_coordinates[1]
    match_code = properties.get("match_code")
    return {
        "mapbox_status": "ok",
        "mapbox_lat": lat if lat is not None else "",
        "mapbox_lon": lon if lon is not None else "",
        "mapbox_name": properties.get("name", ""),
        "mapbox_full_address": properties.get("full_address") or properties.get("place_formatted") or "",
        "mapbox_feature_type": properties.get("feature_type", ""),
        "mapbox_relevance": properties.get("relevance", ""),
        "mapbox_accuracy": coordinates.get("accuracy", ""),
        "mapbox_match_code": json.dumps(match_code, ensure_ascii=False, sort_keys=True) if match_code else "",
        "mapbox_id": properties.get("mapbox_id", ""),
    }


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def geocode_rows(
    rows: list[dict[str, str]],
    *,
    token: str,
    cache: dict[str, Any],
    sleep_seconds: float,
    force: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        tt = str(row.get("tt", "")).strip()
        query = build_query(row)
        cache_key = f"v6|permanent=true|country=vn|language=vi|{query}"
        if not force and cache_key in cache:
            response = cache[cache_key]
        else:
            try:
                response = mapbox_forward(
                    token,
                    query,
                    permanent=True,
                    language="vi",
                    country="vn",
                    limit=1,
                )
            except urllib.error.HTTPError as exc:
                payload = exc.read().decode("utf-8", errors="replace")
                response = {"error": {"status": exc.code, "message": payload}}
            except Exception as exc:
                response = {"error": {"status": "", "message": str(exc)}}
            cache[cache_key] = response
            save_cache(MAPBOX_RESPONSE_CACHE, cache)
            if sleep_seconds:
                time.sleep(sleep_seconds)
        features = response.get("features") or []
        feature = features[0] if features else None
        flat = flatten_feature(feature)
        if "error" in response:
            flat["mapbox_status"] = f"error_{response['error'].get('status', '')}"
            flat["mapbox_error"] = response["error"].get("message", "")
        else:
            flat["mapbox_error"] = ""
        result = {
            **row,
            **flat,
            "mapbox_geocode_confidence": "",
            "mapbox_geocode_query": query,
            "mapbox_permanent": "true",
            "mapbox_geocoded_utc": datetime.now(timezone.utc).isoformat(),
        }
        result["mapbox_geocode_confidence"] = classify_mapbox(result)
        out.append(result)
        print(
            f"{index:03d}/170 TT={tt} {result['mapbox_geocode_confidence']} "
            f"{result.get('mapbox_lat')},{result.get('mapbox_lon')} {result.get('mapbox_name')}",
            flush=True,
        )
    return out


def make_pipeline_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "TT": row.get("tt", ""),
                "facility_name_vi": row.get("ten_benh_vien", ""),
                "service_type_vi": row.get("loai_hinh", ""),
                "address_vi": row.get("dia_chi", ""),
                "hotline": row.get("duong_day_nong", ""),
                "thrombolysis_vi": row.get("bv_co_tieu_soi_huyet", ""),
                "intervention_vi": row.get("bv_co_can_thiep", ""),
                "province_city_vi": row.get("tinh_thanh_pho", ""),
                "latitude": row.get("mapbox_lat", ""),
                "longitude": row.get("mapbox_lon", ""),
                "geocode_provider": "mapbox",
                "geocode_confidence": row.get("mapbox_geocode_confidence", ""),
                "mapbox_feature_type": row.get("mapbox_feature_type", ""),
                "mapbox_relevance": row.get("mapbox_relevance", ""),
                "mapbox_accuracy": row.get("mapbox_accuracy", ""),
                "mapbox_name": row.get("mapbox_name", ""),
                "mapbox_full_address": row.get("mapbox_full_address", ""),
                "mapbox_id": row.get("mapbox_id", ""),
                "geocode_query": row.get("mapbox_geocode_query", ""),
                "geocode_review_flags": "",
                "source_page": row.get("source_page", ""),
                "source_file": row.get("source_file", ""),
            }
        )
    return out


def validate_against_google(mapbox_rows: list[dict[str, Any]], google_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    google_by_tt = {str(row.get("TT", "")).strip(): row for row in google_rows}
    out: list[dict[str, Any]] = []
    for row in mapbox_rows:
        tt = str(row.get("tt", "")).strip()
        google = google_by_tt.get(tt, {})
        mapbox_lat = fnum(row.get("mapbox_lat"))
        mapbox_lon = fnum(row.get("mapbox_lon"))
        google_lat = fnum(google.get("latitude"))
        google_lon = fnum(google.get("longitude"))
        distance = haversine_m(mapbox_lat, mapbox_lon, google_lat, google_lon)
        out.append(
            {
                "TT": tt,
                "facility_name_vi": row.get("ten_benh_vien", ""),
                "province_city_vi": row.get("tinh_thanh_pho", ""),
                "mapbox_lat": row.get("mapbox_lat", ""),
                "mapbox_lon": row.get("mapbox_lon", ""),
                "mapbox_confidence": row.get("mapbox_geocode_confidence", ""),
                "mapbox_relevance": row.get("mapbox_relevance", ""),
                "mapbox_feature_type": row.get("mapbox_feature_type", ""),
                "mapbox_name": row.get("mapbox_name", ""),
                "mapbox_full_address": row.get("mapbox_full_address", ""),
                "google_lat": google.get("latitude", ""),
                "google_lon": google.get("longitude", ""),
                "google_confidence": google.get("geocode_confidence", ""),
                "google_location_type": google.get("google_location_type", ""),
                "google_partial_match": google.get("google_partial_match", ""),
                "google_formatted_address": google.get("google_formatted_address", ""),
                "distance_mapbox_google_m": "" if distance is None else round(distance, 3),
                "distance_flag": distance_flag(distance),
            }
        )
    return out


def summarize(mapbox_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], started: str) -> dict[str, Any]:
    confidence_counts = Counter(row.get("mapbox_geocode_confidence", "") for row in mapbox_rows)
    flag_counts = Counter(row.get("distance_flag", "") for row in validation_rows)
    distances = [
        float(row["distance_mapbox_google_m"])
        for row in validation_rows
        if str(row.get("distance_mapbox_google_m", "")).strip()
    ]
    distances.sort()
    def percentile(p: float) -> float | None:
        if not distances:
            return None
        k = (len(distances) - 1) * p
        lo = math.floor(k)
        hi = math.ceil(k)
        if lo == hi:
            return distances[int(k)]
        return distances[lo] * (hi - k) + distances[hi] * (k - lo)

    poor_count = sum(
        1
        for row in mapbox_rows
        if str(row.get("mapbox_geocode_confidence", "")).startswith("poor")
        or str(row.get("mapbox_geocode_confidence", "")) == "failed"
    )
    major_disagreement = flag_counts.get("major_>20km", 0) + flag_counts.get("large_5_20km", 0)
    quality_decision = "use_with_manual_review"
    if poor_count > 5 or major_disagreement > 20:
        quality_decision = "do_not_use_as_primary_without_manual_correction"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started,
        "source_raw_csv": str(RAW_170),
        "google_reference_csv": str(GOOGLE_PIPELINE),
        "mapbox_compact_csv": str(MAPBOX_COMPACT),
        "mapbox_pipeline_source_csv": str(MAPBOX_PIPELINE),
        "validation_csv": str(MAPBOX_VALIDATION_CSV),
        "provider": "Mapbox Geocoding API v6 forward",
        "permanent_parameter": True,
        "country_parameter": "vn",
        "language_parameter": "vi",
        "record_count": len(mapbox_rows),
        "mapbox_confidence_counts": dict(confidence_counts),
        "mapbox_google_distance_flag_counts": dict(flag_counts),
        "distance_mapbox_google_m": {
            "count": len(distances),
            "min": distances[0] if distances else None,
            "median": percentile(0.5),
            "p75": percentile(0.75),
            "p90": percentile(0.9),
            "p95": percentile(0.95),
            "max": distances[-1] if distances else None,
        },
        "poor_or_failed_mapbox_count": poor_count,
        "large_or_major_mapbox_google_disagreement_count": major_disagreement,
        "quality_decision": quality_decision,
        "storage_note": (
            "Coordinates are stored as a local derived dataset only. Mapbox was queried "
            "with permanent=true. Account/license conditions must be checked before "
            "committing coordinates to GitHub or using them as a published reproducibility source."
        ),
    }


def write_notes(summary: dict[str, Any]) -> None:
    lines = [
        "# Vietnam 170 Mapbox Permanent Geocoding Notes",
        "",
        f"Created UTC: {summary['created_utc']}",
        "",
        "## Source and provider",
        "",
        f"- Raw 170-row source: `{summary['source_raw_csv']}`",
        f"- Provider: {summary['provider']}",
        "- Request mode: `permanent=true`, `country=vn`, `language=vi`, `limit=1`.",
        "- Google-derived coordinates are used only as a validation reference; the coordinate sets are not mixed.",
        "",
        "## Storage and publication guardrail",
        "",
        (
            "Mapbox and Google geocoding outputs are provider-derived data, not automatically open data. "
            "The Mapbox calls in this run used `permanent=true`, which is required when the returned "
            "coordinates are intended to be stored and reused. This technical setting is not by itself "
            "a publication license check: account terms and project conditions must still be reviewed "
            "before writing the coordinate CSV to GitHub or treating it as a public reproducibility input."
        ),
        "",
        "## Summary",
        "",
        f"- Records geocoded: {summary['record_count']}",
        f"- Mapbox confidence counts: `{summary['mapbox_confidence_counts']}`",
        f"- Mapbox-Google distance flags: `{summary['mapbox_google_distance_flag_counts']}`",
        f"- Distance statistics in meters: `{summary['distance_mapbox_google_m']}`",
        f"- Quality decision: `{summary['quality_decision']}`",
        "",
        "## Files",
        "",
        f"- Mapbox compact CSV: `{summary['mapbox_compact_csv']}`",
        f"- Mapbox pipeline source CSV: `{summary['mapbox_pipeline_source_csv']}`",
        f"- Mapbox vs Google validation CSV: `{summary['validation_csv']}`",
        f"- Machine-readable summary: `{MAPBOX_SUMMARY_JSON}`",
    ]
    MAPBOX_NOTES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore the local Mapbox response cache.")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()
    token = os.environ.get("MAPBOX_ACCESS_TOKEN", "").strip()
    if not token:
        print("MAPBOX_ACCESS_TOKEN is required.", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc).isoformat()
    raw_rows = read_csv(RAW_170)
    google_rows = read_csv(GOOGLE_PIPELINE)
    cache = load_cache(MAPBOX_RESPONSE_CACHE)
    mapbox_rows = geocode_rows(
        raw_rows,
        token=token,
        cache=cache,
        sleep_seconds=args.sleep_seconds,
        force=args.force,
    )
    pipeline_rows = make_pipeline_source(mapbox_rows)
    validation_rows = validate_against_google(mapbox_rows, google_rows)
    write_csv(MAPBOX_COMPACT, mapbox_rows)
    write_csv(MAPBOX_PIPELINE, pipeline_rows)
    write_csv(MAPBOX_VALIDATION_CSV, validation_rows)
    summary = summarize(mapbox_rows, validation_rows, started)
    MAPBOX_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
