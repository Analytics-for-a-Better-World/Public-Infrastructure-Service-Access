from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "health_service_locations"
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

DATASET_ID = "caracterizacao-das-valencias-de-urgencia"
API_BASE = "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets"
HEADERS = {"User-Agent": "PISA thesis Portugal SNS urgency official download"}


def fetch_records(limit: int = 100) -> list[dict]:
    records: list[dict] = []
    offset = 0
    while True:
        url = f"{API_BASE}/{DATASET_ID}/records"
        response = requests.get(
            url,
            params={"limit": limit, "offset": offset},
            headers=HEADERS,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("results", [])
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if len(records) >= payload.get("total_count", 0):
            break
    return records


def normalize_raw(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        loc = rec.get("localizacao_geografica") or {}
        row = dict(rec)
        row.pop("localizacao_geografica", None)
        row["latitude"] = loc.get("lat")
        row["longitude"] = loc.get("lon")
        rows.append(row)
    return pd.DataFrame(rows)


def deduplicate_facilities(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Facility point = one urgency service at one coordinate.
    # Other fields, such as hospital/entity labels, can vary across valency records.
    group_cols = [
        "nome_do_servico_de_urgencia",
        "latitude",
        "longitude",
    ]
    for _, group in raw_df.groupby(group_cols, dropna=False):
        first = group.iloc[0]
        valencias = sorted(set(group["nome_da_valencia"].dropna().astype(str))) if "nome_da_valencia" in group else []
        hospitals = sorted(set(group.get("unidade_hospitalar", pd.Series(dtype=str)).dropna().astype(str)))
        parent_entities = sorted(set(group.get("entidade_grupo_hospitalar", pd.Series(dtype=str)).dropna().astype(str)))
        urgency_types = sorted(set(group.get("tipo_de_urgencia", pd.Series(dtype=str)).dropna().astype(str)))
        rows.append({
            "facility_id": f"sns_urgency:{len(rows)+1}",
            "layer": "hospital_emergency_official",
            "name": first.get("nome_do_servico_de_urgencia", ""),
            "hospital": " | ".join(hospitals),
            "parent_entity": " | ".join(parent_entities),
            "urgency_type": " | ".join(urgency_types),
            "latitude": first.get("latitude"),
            "longitude": first.get("longitude"),
            "address": first.get("endereco", ""),
            "city": first.get("localidade", ""),
            "postcode": first.get("codigo_postal", ""),
            "region": first.get("regiao", ""),
            "phone": first.get("telefone", ""),
            "email": first.get("email", ""),
            "parish": first.get("freguesia", ""),
            "n_valencias": len(valencias),
            "valencias": " | ".join(valencias),
            "source": "SNS Transparência",
            "source_dataset": DATASET_ID,
            "source_url": "https://transparencia.sns.gov.pt/explore/dataset/caracterizacao-das-valencias-de-urgencia/",
        })
    return pd.DataFrame(rows).sort_values(["region", "city", "name"]).reset_index(drop=True)


def main() -> None:
    records = fetch_records()
    raw_df = normalize_raw(records)
    facilities = deduplicate_facilities(raw_df)

    raw_path = RAW / "portugal_sns_urgency_valencies_raw.csv"
    facility_path = OUT / "portugal_sns_urgency_facility_locations_official.csv"
    metadata_path = OUT / "portugal_sns_urgency_facility_locations_official_metadata.md"

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    facilities.to_csv(facility_path, index=False, encoding="utf-8-sig")
    metadata_path.write_text(
        "# Portugal SNS Urgency Facility Locations\n\n"
        "Official source: SNS Transparência dataset `caracterizacao-das-valencias-de-urgencia`.\n\n"
        "Raw records describe urgency-service valencies; the facility file deduplicates those records into point locations.\n\n"
        f"Raw records: {len(raw_df)}\n\n"
        f"Deduplicated facilities: {len(facilities)}\n\n"
        "Source URL: https://transparencia.sns.gov.pt/explore/dataset/caracterizacao-das-valencias-de-urgencia/\n",
        encoding="utf-8",
    )

    print(f"Raw records: {len(raw_df)} -> {raw_path}")
    print(f"Deduplicated facilities: {len(facilities)} -> {facility_path}")
    print(facilities["urgency_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
