from __future__ import annotations

import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode

import pandas as pd
import requests


BASE = Path(__file__).resolve().parent
BEST = BASE / "best_healthcare_csvs"
DATA = BASE / "data" / "health_service_locations"
INPUT = BEST / "portugal_permanent_care_curated_official_pages.csv"
OUTPUT_BEST = BEST / "portugal_permanent_care_curated_official_pages_geocoded.csv"
OUTPUT_DATA = DATA / "portugal_permanent_care_curated_official_pages_geocoded.csv"
CACHE_PATH = BEST / "geocoding_cache_portugal_permanent_care.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GOOGLE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
USER_AGENT = "PISA thesis Portugal permanent care geocoding"


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_queries(row: pd.Series) -> list[str]:
    name = clean(row.get("name"))
    address = clean(row.get("address"))
    municipality = clean(row.get("municipality"))

    queries = [
        ", ".join(part for part in [name, address, municipality, "Portugal"] if part),
        ", ".join(part for part in [address, municipality, "Portugal"] if part),
        ", ".join(part for part in [name, municipality, "Portugal"] if part),
    ]

    result: list[str] = []
    for query in queries:
        if query and query not in result:
            result.append(query)
    return result


def geocode_nominatim(query: str) -> dict | None:
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "pt",
        "addressdetails": 1,
    }
    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if not data:
        return None
    item = data[0]
    return {
        "latitude": float(item["lat"]),
        "longitude": float(item["lon"]),
        "geocode_status": "geocoded",
        "geocode_provider": "nominatim",
        "geocode_confidence": item.get("importance"),
        "geocode_formatted_address": item.get("display_name", ""),
        "geocode_raw_type": item.get("type", ""),
    }


def geocode_google(query: str, api_key: str) -> dict | None:
    params = {
        "address": query,
        "region": "pt",
        "components": "country:PT",
        "key": api_key,
    }
    response = requests.get(f"{GOOGLE_URL}?{urlencode(params)}", timeout=30)
    response.raise_for_status()
    data = response.json()
    status = data.get("status")
    if status != "OK" or not data.get("results"):
        return None
    item = data["results"][0]
    location = item["geometry"]["location"]
    return {
        "latitude": float(location["lat"]),
        "longitude": float(location["lng"]),
        "geocode_status": "geocoded",
        "geocode_provider": "google_maps",
        "geocode_confidence": item.get("geometry", {}).get("location_type", ""),
        "geocode_formatted_address": item.get("formatted_address", ""),
        "geocode_raw_type": "|".join(item.get("types", [])),
    }


def geocode_row(row: pd.Series, cache: dict[str, dict], google_key: str | None) -> dict:
    queries = build_queries(row)
    if not queries:
        return {
            "latitude": None,
            "longitude": None,
            "geocode_status": "no_query",
            "geocode_provider": "",
            "geocode_confidence": "",
            "geocode_query": "",
            "geocode_formatted_address": "",
            "geocode_raw_type": "",
        }

    for query in queries:
        cache_key = f"nominatim::{query}"
        if cache_key not in cache:
            try:
                cache[cache_key] = geocode_nominatim(query) or {"geocode_status": "not_found"}
            except Exception as exc:
                cache[cache_key] = {"geocode_status": f"error:{type(exc).__name__}"}
            save_cache(cache)
            time.sleep(1.1)
        cached = cache[cache_key]
        if cached.get("geocode_status") == "geocoded":
            return {**cached, "geocode_query": query}

    if google_key:
        for query in queries:
            cache_key = f"google_maps::{query}"
            if cache_key not in cache:
                try:
                    cache[cache_key] = geocode_google(query, google_key) or {
                        "geocode_status": "not_found"
                    }
                except Exception as exc:
                    cache[cache_key] = {"geocode_status": f"error:{type(exc).__name__}"}
                save_cache(cache)
                time.sleep(0.1)
            cached = cache[cache_key]
            if cached.get("geocode_status") == "geocoded":
                return {**cached, "geocode_query": query}

    return {
        "latitude": None,
        "longitude": None,
        "geocode_status": "not_found",
        "geocode_provider": "",
        "geocode_confidence": "",
        "geocode_query": queries[0],
        "geocode_formatted_address": "",
        "geocode_raw_type": "",
    }


def main() -> None:
    df = pd.read_csv(INPUT)
    cache = load_cache()
    google_key = os.environ.get("GOOGLE_MAPS_API_KEY")

    records: list[dict] = []
    for _, row in df.iterrows():
        records.append(geocode_row(row, cache, google_key))

    result = df.copy()
    for column in [
        "latitude",
        "longitude",
        "geocode_status",
        "geocode_provider",
        "geocode_confidence",
        "geocode_query",
        "geocode_formatted_address",
        "geocode_raw_type",
    ]:
        result[column] = [record.get(column) for record in records]

    result["needs_geocoding"] = result["geocode_status"] != "geocoded"
    result.to_csv(OUTPUT_BEST, index=False)
    result.to_csv(OUTPUT_DATA, index=False)

    summary = result.groupby(["geocode_status", "geocode_provider"], dropna=False).size()
    print(summary.to_string())
    print(f"Wrote {OUTPUT_BEST}")
    print(f"Wrote {OUTPUT_DATA}")


if __name__ == "__main__":
    main()
