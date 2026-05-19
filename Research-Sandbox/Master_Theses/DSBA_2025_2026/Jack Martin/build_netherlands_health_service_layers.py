from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
from pyproj import Transformer

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "health_service_locations"
OUT.mkdir(parents=True, exist_ok=True)
VZ_BASE = "https://cstm.rivm.nl/vzinfo/custom-vzinfo-data/maps"
OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "PISA thesis data gathering; Netherlands health service layers"}
COLS = [
    "facility_id", "layer", "name", "latitude", "longitude", "address", "city", "postcode",
    "source", "source_detail", "collaboration", "osm_type", "osm_id", "amenity", "healthcare",
    "healthcare_speciality", "operator", "phone", "website", "rd_x", "rd_y", "hospital_type",
]


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


def osm_rows(elements: list[dict], layer: str, source_detail: str) -> pd.DataFrame:
    rows = []
    for element in elements:
        lat, lon = point(element)
        if lat is None or lon is None:
            continue
        tags = element.get("tags") or {}
        rows.append({
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
    return pd.DataFrame(rows).drop_duplicates(subset=["facility_id"])


def write_geojson(df: pd.DataFrame, path: Path) -> None:
    features = []
    for rec in df.to_dict("records"):
        if pd.isna(rec.get("latitude")) or pd.isna(rec.get("longitude")):
            continue
        props = {k: (None if pd.isna(v) else v) for k, v in rec.items() if k not in {"latitude", "longitude"}}
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(rec["longitude"]), float(rec["latitude"])]}, "properties": props})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_hap() -> pd.DataFrame:
    config = requests.get(f"{VZ_BASE}/maps/locaties-huisartsenposten-2025/config.json", headers=HEADERS, timeout=60).json()
    table = pd.read_csv(f"{VZ_BASE}/maps/{config['id']}/{config['map']['csv']}", sep=";")
    geo = requests.get(f"{VZ_BASE}/{config['map']['locations'][0]['uri']}", headers=HEADERS, timeout=60).json()
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    coords = []
    for feature in geo["features"]:
        x, y = feature["geometry"]["coordinates"]
        lon, lat = transformer.transform(x, y)
        coords.append({"id": feature["properties"]["id"], "latitude": lat, "longitude": lon, "rd_x": x, "rd_y": y})
    df = table.merge(pd.DataFrame(coords), on="id", how="left")
    return pd.DataFrame({
        "facility_id": "vzinfo_hap:" + df["id"].astype(str),
        "layer": "huisartsenpost",
        "name": df["Naam"],
        "latitude": df["latitude"],
        "longitude": df["longitude"],
        "city": df["Plaats"],
        "source": "RIVM/VZinfo",
        "source_detail": "Locaties huisartsenspoedposten 2025, peildatum december 2025",
        "collaboration": df["Waarde"],
        "website": "https://www.vzinfo.nl/acute-zorg/regionaal/hap",
        "rd_x": df["rd_x"],
        "rd_y": df["rd_y"],
    })


def fetch_seh_hospitals() -> pd.DataFrame:
    name = "reistijd-naar-dichtstbijzijnde-ziekenhuis-met-seh-2025"
    config = requests.get(f"{VZ_BASE}/maps/{name}/config.json", headers=HEADERS, timeout=60).json()
    click = pd.read_csv(f"{VZ_BASE}/maps/{config['id']}/{config['map']['csv']}", sep=";")
    wide = click.pivot_table(index=["id", "Ziekenhuis"], columns="Indicator", values="Waarde", aggfunc="first").reset_index()
    geo = requests.get(f"{VZ_BASE}/{config['map']['locations'][0]['uri']}", headers=HEADERS, timeout=60).json()
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    coords = []
    for feature in geo["features"]:
        props = feature["properties"]
        x, y = feature["geometry"]["coordinates"]
        lon, lat = transformer.transform(x, y)
        coords.append({"id": int(props["ALGZKHNR"]), "latitude": lat, "longitude": lon, "postcode": props.get("PC6", ""), "rd_x": x, "rd_y": y})
    df = wide.merge(pd.DataFrame(coords), on="id", how="left")
    return pd.DataFrame({
        "facility_id": "vzinfo_seh:" + df["id"].astype(str),
        "layer": "hospital",
        "name": df["Ziekenhuis"],
        "latitude": df["latitude"],
        "longitude": df["longitude"],
        "city": df.get("Plaatsnaam", ""),
        "postcode": df["postcode"],
        "source": "RIVM/VZinfo",
        "source_detail": "Reistijd naar dichtstbijzijnde ziekenhuis met SEH 2025",
        "website": "https://www.vzinfo.nl/acute-zorg/regionaal/hap#map/reistijd-naar-dichtstbijzijnde-ziekenhuis-met-seh-2025",
        "rd_x": df["rd_x"],
        "rd_y": df["rd_y"],
        "hospital_type": df["Type SEH"],
    })


def fetch_gp() -> pd.DataFrame:
    query = '''
[out:json][timeout:240];
area["ISO3166-1"="NL"][admin_level=2]->.nl;
(
  nwr["amenity"="doctors"](area.nl);
  nwr["healthcare"="doctor"](area.nl);
  nwr["healthcare:speciality"~"general_practitioner|huisarts",i](area.nl);
  nwr["name"~"huisarts|huisartsen|huisartsenpraktijk",i](area.nl);
);
out center tags;
'''
    df = osm_rows(overpass(query), "general_practitioner", "OSM GP/huisarts tags and name match")
    return df[~df["name"].str.lower().str.contains("huisartsenpost|spoedpost|spoedzorg", na=False)]


def main() -> None:
    frames = [fetch_gp(), fetch_hap(), fetch_seh_hospitals()]
    study = pd.concat([f.reindex(columns=COLS) for f in frames], ignore_index=True)
    study.to_csv(OUT / "netherlands_health_service_locations_study_layers.csv", index=False, encoding="utf-8-sig")
    write_geojson(study, OUT / "netherlands_health_service_locations_study_layers.geojson")
    for layer, df in study.groupby("layer"):
        df.to_csv(OUT / f"netherlands_{layer}.csv", index=False, encoding="utf-8-sig")
        write_geojson(df, OUT / f"netherlands_{layer}.geojson")
    (OUT / "metadata.json").write_text(json.dumps({"counts": study["layer"].value_counts().to_dict()}, indent=2), encoding="utf-8")
    print(study["layer"].value_counts())


if __name__ == "__main__":
    main()
