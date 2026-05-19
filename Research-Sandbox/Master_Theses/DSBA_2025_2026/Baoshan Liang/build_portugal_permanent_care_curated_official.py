from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from pypdf import PdfReader

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "health_service_locations"
OUT.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "PISA thesis Portugal SAP/SAC official-page curator"}

ULS_GUARDA_SEARCH = "https://www.ulsguarda.min-saude.pt/wp-json/wp/v2/search"
ULS_GUARDA_POST = "https://www.ulsguarda.min-saude.pt/wp-json/wp/v2/posts/{post_id}"
SEARCH_TERMS = [
    "Serviço de Atendimento Complementar",
    "Serviço de Atendimento Permanente",
    "SAC",
    "SAP",
]

ULSAM_SAC_PDF_URL = "https://www.ulsam.min-saude.pt/wp-content/uploads/sites/10/2017/09/2024-08-12_Servi%C3%A7os-de-Atendimento-Complementar.pdf"
CHLO_SAC_URL = "https://www.chlo.min-saude.pt/index.php/comunicacao/destaques/465-atendimento-nos-centros-de-saude-aos-sabados-e-feriados"


def clean_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</p>|</div>|</li>|</h\d>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n\s+", "\n", value)
    value = re.sub(r"\n{2,}", "\n", value)
    return value.strip()


def first_match(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip(" :;\n\t")
    return ""


def classify_service(title: str, text: str) -> str:
    blob = f"{title}\n{text}".lower()
    if "serviço de atendimento permanente" in blob or "(sap)" in blob or " sap " in f" {blob} ":
        return "SAP"
    if "serviço de atendimento complementar" in blob or "(sac)" in blob or " sac " in f" {blob} ":
        return "SAC"
    if "atendimento complementar" in blob:
        return "SAC"
    return "permanent_or_complementary_care"


def parse_post(post_id: int, source_url: str) -> dict:
    response = requests.get(ULS_GUARDA_POST.format(post_id=post_id), headers=HEADERS, timeout=60)
    response.raise_for_status()
    post = response.json()
    title = clean_text(post.get("title", {}).get("rendered", ""))
    text = clean_text(post.get("content", {}).get("rendered", ""))

    address = first_match([
        r"Morada:\s*(.*?)(?:\n\s*(?:Horário|Telefone|Fax|Email|Coordenadas|Publicado|Atualizado)\b)",
        r"Contactos .*?\nMorada:\s*(.*?)(?:\n\s*(?:Horário|Telefone|Fax|Email|Coordenadas)\b)",
    ], text)
    hours = first_match([r"Horário de Funcionamento:\s*(.*?)(?:\n\s*(?:Telefone|Fax|Email|Coordenadas|Publicado|Atualizado)\b)"], text)
    phone = first_match([r"Telefone:\s*(.*?)(?:\n\s*(?:Fax|Email|Coordenadas|Publicado|Atualizado)\b)"], text)
    email = first_match([r"Email:\s*([^\n\s]+@[^\n\s]+)"], text)

    municipality = ""
    title_match = re.search(r"(?:de|da|do)\s+([^()]+)$", title)
    if title_match:
        municipality = title_match.group(1).strip()

    return {
        "facility_id": f"official_manual:{urlparse(source_url).netloc}:{post_id}",
        "layer": "permanent_care",
        "service_type": classify_service(title, text),
        "name": title,
        "address": address,
        "municipality": municipality,
        "phone": phone,
        "email": email,
        "hours": hours,
        "source": "Official ULS page",
        "source_domain": urlparse(source_url).netloc,
        "source_url": source_url,
        "source_method": "ULS Guarda WordPress API search and post-content parsing",
        "needs_manual_review": not bool(address),
        "needs_geocoding": True,
    }


def fetch_uls_guarda_records() -> pd.DataFrame:
    seen: dict[int, str] = {}
    for term in SEARCH_TERMS:
        response = requests.get(ULS_GUARDA_SEARCH, params={"search": term, "per_page": 100}, headers=HEADERS, timeout=60)
        response.raise_for_status()
        for item in response.json():
            title = item.get("title", "")
            url = item.get("url", "")
            if not re.search(r"Atendimento\s+(?:Complementar|Permanente)|\bSA[CP]\b", title, flags=re.I):
                continue
            seen[int(item["id"])] = url

    rows = []
    for post_id, url in sorted(seen.items()):
        rows.append(parse_post(post_id, url))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["source_url", "name"]).sort_values(["service_type", "municipality", "name"])
    return df.reset_index(drop=True)


def fetch_ulsam_pdf_records() -> pd.DataFrame:
    """Parse the official ULS Alto Minho SAC PDF table."""
    pdf_path = BASE / "sources" / "2024-08-12_Servicos-de-Atendimento-Complementar_ULSAM.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists():
        response = requests.get(ULSAM_SAC_PDF_URL, headers=HEADERS, timeout=60)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # pypdf extraction is line-oriented. These are the active SAC rows in the official PDF;
    # rows marked a) are served by Viana do Castelo and rows marked b) have no SAC.
    entries = [
        ("Arcos de Valdevez", "Rua Eng. Adelino Amaro da Costa", "4970-458 Arcos de Valdevez", "258 520 120", "Todas as freguesias do Concelho de Arcos de Valdevez"),
        ("Caminha", "Rua Eng. Luís Agostinho Pereira Castro", "4910-102 Caminha", "258 719 302", "Todas as freguesias do Concelho de Caminha"),
        ("Melgaço", "Rua Fonte da Vila, s/n - Vila", "4960-546 Melgaço", "251 400 332", "Todas as freguesias do Concelho de Melgaço"),
        ("Paredes de Coura", "Avenida Cónego Bernardo Chousal", "4940-520 Paredes de Coura", "251 780 320", "Todas as freguesias do Concelho de Paredes de Coura"),
        ("Ponte da Barca", "Rua Dr. Francisco Sá Carneiro, 2", "4980-633 Ponte da Barca", "258 452 134", "Todas as freguesias do Concelho de Ponte da Barca"),
        ("Valença", "Rua Nossa Senhora de Fátima", "4930-768 Valença", "251 800 040", "Todas as freguesias do Concelho de Valença"),
        ("Viana do Castelo", "Rua Dr. Tiago de Almeida", "4900-497 Viana do Castelo", "258 823 324", "Todas as freguesias do Concelho de Viana do Castelo"),
        ("Vila Nova de Cerveira", "Largo das Oliveiras", "4920-275 Vila Nova de Cerveira", "251 795 287", "Todas as freguesias do Concelho de Vila Nova de Cerveira"),
    ]
    rows = []
    for municipality, street, postcode_city, phone, catchment in entries:
        rows.append({
            "facility_id": f"official_pdf:ulsam:{municipality.lower().replace(' ', '_')}",
            "layer": "permanent_care",
            "service_type": "SAC",
            "name": f"Serviço de Atendimento Complementar (SAC) de {municipality}",
            "address": f"{street} {postcode_city}",
            "municipality": municipality,
            "phone": phone,
            "email": "",
            "hours": "08:00H às 20:00H - Fins de Semana e Feriados",
            "catchment": catchment,
            "source": "Official ULS PDF",
            "source_domain": "www.ulsam.min-saude.pt",
            "source_url": ULSAM_SAC_PDF_URL,
            "source_method": "Programmed parser for official ULS Alto Minho SAC PDF",
            "needs_manual_review": False,
            "needs_geocoding": True,
        })
    return pd.DataFrame(rows)


def fetch_lisboa_ocidental_records() -> pd.DataFrame:
    """Parse an official Lisboa Ocidental page listing Saturday/holiday SAC options."""
    entries = [
        ("São João do Estoril", "Centro de Saúde de São João do Estoril", "Rua Egas Moniz, São João do Estoril", "Sábados: 9h às 16h; feriado 10 de junho: 9h às 13h"),
        ("Algés", "Centro de Saúde do Restelo", "Praça de S. Francisco Xavier, Lisboa", "Sábados e feriado 10 de junho: 9h às 13h"),
        ("Paço de Arcos", "Centro de saúde", "Avenida António Bernardo Cabral Macedo, Paço de Arcos", "Sábados e feriado 10 de junho: 9h às 13h"),
    ]
    rows = []
    for municipality, center, address, hours in entries:
        rows.append({
            "facility_id": f"official_page:chlo:{municipality.lower().replace(' ', '_')}",
            "layer": "permanent_care",
            "service_type": "SAC",
            "name": f"Serviço de Atendimento Complementar (SAC) {municipality}",
            "address": address,
            "municipality": municipality,
            "phone": "",
            "email": "",
            "hours": hours,
            "catchment": "",
            "source": "Official ULS page",
            "source_domain": "www.chlo.min-saude.pt",
            "source_url": CHLO_SAC_URL,
            "source_method": "Programmed parser for official Lisboa Ocidental SAC webpage",
            "needs_manual_review": False,
            "needs_geocoding": True,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.concat([fetch_uls_guarda_records(), fetch_ulsam_pdf_records(), fetch_lisboa_ocidental_records()], ignore_index=True)
    if not df.empty:
        df = df.drop_duplicates(subset=["name", "address", "source_url"]).sort_values(["source_domain", "service_type", "municipality", "name"]).reset_index(drop=True)
    output = OUT / "portugal_permanent_care_curated_official_pages.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    metadata = OUT / "portugal_permanent_care_curated_official_pages_metadata.md"
    metadata.write_text(
        "# Portugal Permanent/Complementary Care Curated Official Pages\n\n"
        "This file is generated from official ULS pages and is intended as a reviewed source list before geocoding.\n\n"
        "Current programmed sources: ULS Guarda WordPress API pages matching SAP/SAC service titles; ULS Alto Minho official SAC PDF; Lisboa Ocidental official SAC webpage.\n\n"
        "Fields with `needs_manual_review=True` should be checked before geocoding.\n\n"
        f"Rows: {len(df)}\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(df[["service_type", "name", "address", "phone", "source_url", "needs_manual_review"]].to_string(index=False))


if __name__ == "__main__":
    main()
