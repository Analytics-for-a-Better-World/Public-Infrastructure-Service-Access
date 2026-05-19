from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "health_service_locations"
OUT.mkdir(parents=True, exist_ok=True)
OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "PISA thesis data gathering; Portugal health service layers"}


def overpass(query: str) -> list[dict]:
    response = requests.post(OVERPASS, data={"data": query}, headers=HEADERS, timeout=360)
    response.raise_for_status()
    return response.json().get("elements", [])


def point(element: dict) -> tuple[float | None, float | None]:
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center") or {}
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None, None


def address(tags: dict) -> str:
    return ", ".join(x for x in [
        f"{tags.get('addr:street', '')} {tags.get('addr:housenumber', '')}".strip(),
        tags.get("addr:postcode", ""),
        tags.get("addr:city") or tags.get("addr:place") or "",
    ] if x)


def rows(elements: list[dict], layer: str, source_detail: str) -> pd.DataFrame:
    out = []
    for element in elements:
        lat, lon = point(element)
        if lat is None or lon is None:
            continue
        tags = element.get("tags") or {}
        out.append({
            "facility_id": f"osm:{element.get('type')}:{element.get('id')}",
            "layer": layer,
            "name": tags.get("name") or tags.get("operator") or "",
            "latitude": lat,
            "longitude": lon,
            "address": address(tags),
            "city": tags.get("addr:city") or tags.get("addr:place") or "",
            "postcode": tags.get("addr:postcode") or "",
            "source": "OpenStreetMap via Overpass API",
            "source_detail": source_detail,
            "osm_type": element.get("type"),
            "osm_id": element.get("id"),
            "amenity": tags.get("amenity") or "",
            "healthcare": tags.get("healthcare") or "",
            "healthcare_speciality": tags.get("healthcare:speciality") or "",
            "operator": tags.get("operator") or "",
            "phone": tags.get("phone") or tags.get("contact:phone") or "",
            "website": tags.get("website") or tags.get("contact:website") or "",
        })
    return pd.DataFrame(out).drop_duplicates(subset=["facility_id"])


def write_geojson(df: pd.DataFrame, path: Path) -> None:
    features = []
    for rec in df.to_dict("records"):
        if pd.isna(rec.get("latitude")) or pd.isna(rec.get("longitude")):
            continue
        props = {k: (None if pd.isna(v) else v) for k, v in rec.items() if k not in {"latitude", "longitude"}}
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(rec["longitude"]), float(rec["latitude"])]}, "properties": props})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_primary_care() -> pd.DataFrame:
    query = '''
[out:json][timeout:240];
area["ISO3166-1"="PT"][admin_level=2]->.pt;
(
  nwr["amenity"="doctors"](area.pt);
  nwr["healthcare"="doctor"](area.pt);
  nwr["healthcare"="clinic"]["name"~"Centro de Saude|Centro de Saúde|USF|UCSP|Unidade de Saude|Unidade de Saúde",i](area.pt);
  nwr["name"~"Centro de Saude|Centro de Saúde|USF|UCSP|Unidade de Saude|Unidade de Saúde",i](area.pt);
);
out center tags;
'''
    return rows(overpass(query), "primary_care", "OSM primary-care proxy: doctors plus Centro de Saúde/USF/UCSP name patterns")


def fetch_permanent_care() -> pd.DataFrame:
    query = """
[out:json][timeout:240];
area["ISO3166-1"="PT"][admin_level=2]->.pt;
(
  nwr["name"~"Serviço de Atendimento Permanente|Servico de Atendimento Permanente|Atendimento Permanente|SAP|Atendimento Prolongado|Serviço de Atendimento Complementar|Servico de Atendimento Complementar",i](area.pt);
  nwr["healthcare"="clinic"]["name"~"Atendimento Permanente|SAP|Atendimento Prolongado|Atendimento Complementar",i](area.pt);
  nwr["amenity"~"clinic|doctors|hospital",i]["name"~"Atendimento Permanente|SAP|Atendimento Prolongado|Atendimento Complementar",i](area.pt);
);
out center tags;
"""
    return rows(
        overpass(query),
        "permanent_care",
        "OSM SAP/AP proxy: Serviço de Atendimento Permanente, Atendimento Permanente, SAP, Atendimento Prolongado, or Atendimento Complementar name patterns",
    )


def fetch_hospitals() -> pd.DataFrame:
    query = '''
[out:json][timeout:240];
area["ISO3166-1"="PT"][admin_level=2]->.pt;
(
  nwr["amenity"="hospital"](area.pt);
  nwr["healthcare"="hospital"](area.pt);
  nwr["emergency"="yes"]["amenity"~"hospital|clinic|doctors",i](area.pt);
  nwr["emergency"="yes"]["healthcare"~"hospital|clinic|doctor",i](area.pt);
);
out center tags;
'''
    return rows(overpass(query), "hospital_emergency_proxy", "OSM hospital/emergency proxy: amenity=hospital, healthcare=hospital, or emergency=yes on healthcare facility tags")


def main() -> None:
    study = pd.concat([fetch_primary_care(), fetch_permanent_care(), fetch_hospitals()], ignore_index=True).drop_duplicates(subset=["facility_id"])
    study.to_csv(OUT / "portugal_health_service_locations_study_layers.csv", index=False, encoding="utf-8-sig")
    write_geojson(study, OUT / "portugal_health_service_locations_study_layers.geojson")
    for layer, df in study.groupby("layer"):
        df.to_csv(OUT / f"portugal_{layer}.csv", index=False, encoding="utf-8-sig")
        write_geojson(df, OUT / f"portugal_{layer}.geojson")
    (OUT / "metadata.json").write_text(json.dumps({
        "counts": study["layer"].value_counts().to_dict(),
        "note": "Portugal layers are OSM-derived proxies. permanent_care captures SAP/AP/atendimento complementar name patterns and should be reviewed against official SNS/ERS facility registers if a stable downloadable endpoint is identified.",
    }, indent=2), encoding="utf-8")
    print(study["layer"].value_counts())


if __name__ == "__main__":
    main()
