from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "sns_monitoring"
OUT.mkdir(parents=True, exist_ok=True)

SNS_DATASET_ID = "monitorizacao-sazonal-csh"
SNS_URL = (
    "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets/"
    f"{SNS_DATASET_ID}/records"
)


def fetch_sns_hospital_care_seasonal_monitoring() -> pd.DataFrame:
    """Fetch SNS hospital-care seasonal monitoring records.

    This is not a facility-location dataset. It is a contextual activity dataset
    with urgency indicators by date and ARS region. Use it for demand/workload
    context, not for mapping service points.
    """
    export_url = (
        "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets/"
        f"{SNS_DATASET_ID}/exports/csv"
    )
    response = requests.get(
        export_url,
        params={
            "lang": "pt",
            "timezone": "Europe/Lisbon",
            "use_labels": "false",
            "delimiter": ",",
        },
        timeout=180,
        headers={"User-Agent": "PISA thesis Portugal SNS monitoring fetch"},
    )
    response.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(response.content.decode("utf-8-sig")))
    if not df.empty:
        sort_cols = [col for col in ["periodo", "ars", "indicador"] if col in df.columns]
        df = df.drop_duplicates()
        if sort_cols:
            df = df.sort_values(sort_cols)
        df = df.reset_index(drop=True)
    return df


def main() -> None:
    df = fetch_sns_hospital_care_seasonal_monitoring()
    csv_path = OUT / "portugal_sns_hospital_care_seasonal_monitoring.csv"
    meta_path = OUT / "portugal_sns_hospital_care_seasonal_monitoring_metadata.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(
        "# Portugal SNS Hospital-Care Seasonal Monitoring\n\n"
        "Source: SNS Transparência dataset `monitorizacao-sazonal-csh`.\n\n"
        "Use: contextual urgency activity, access, and demand indicators by date and ARS region.\n\n"
        "Do not use as a facility-location layer: the records are regional/time/indicator observations, "
        "not individual service points.\n\n"
        f"Rows fetched: {len(df)}\n"
        f"Columns: {', '.join(df.columns.astype(str)) if not df.empty else 'none'}\n",
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Rows: {len(df)}")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
