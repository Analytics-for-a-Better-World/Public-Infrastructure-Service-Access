from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


DOWNLOADS = Path(r"C:\Users\joaqu\Downloads")
TERMS = [
    "parvathy",
    "krishnakumari",
    "joyce",
    "fleur",
    "theulen",
    "thesis",
    "dissertation",
    "phd",
    "master",
    "public infrastructure",
    "service access",
    "vietnam",
    "timor",
    "leste",
    "maximal covering",
    "max cover",
    "p-median",
    "optimization",
    "optimisation",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def metadata_value(value: object) -> str:
    if value is None:
        return ""
    return normalize(str(value))


def read_pdf(reader: PdfReader) -> tuple[list[str], list[str]]:
    page_texts: list[str] = []
    errors: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_texts.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - diagnostic script
            errors.append(f"page {idx}: {exc}")
            page_texts.append("")
    return page_texts, errors


def term_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {term: lowered.count(term) for term in TERMS if lowered.count(term)}


def snippets(text: str, max_snippets: int = 6) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for term in TERMS:
        pos = lowered.find(term)
        if pos < 0:
            continue
        start = max(0, pos - 120)
        end = min(len(text), pos + len(term) + 220)
        found.append(normalize(text[start:end]))
        if len(found) >= max_snippets:
            break
    return found


def inspect_pdf_stream(label: str, stream: str | BinaryIO | BytesIO) -> None:
    print(f"\n=== PDF: {label} ===")
    try:
        reader = PdfReader(stream)
    except Exception as exc:
        print(f"ERROR: cannot read PDF: {exc}")
        return

    print(f"pages: {len(reader.pages)}")
    meta = reader.metadata or {}
    interesting_meta = {
        "title": metadata_value(meta.get("/Title")),
        "author": metadata_value(meta.get("/Author")),
        "subject": metadata_value(meta.get("/Subject")),
        "creator": metadata_value(meta.get("/Creator")),
        "producer": metadata_value(meta.get("/Producer")),
        "created": metadata_value(meta.get("/CreationDate")),
        "modified": metadata_value(meta.get("/ModDate")),
    }
    for key, value in interesting_meta.items():
        if value:
            print(f"{key}: {value[:500]}")

    page_texts, errors = read_pdf(reader)
    if errors:
        print("extract_errors:", "; ".join(errors[:5]))
    all_text = "\n".join(page_texts)
    counts = term_counts(all_text)
    print("term_hits:", counts if counts else "{}")

    first_pages = normalize("\n".join(page_texts[:3]))
    if first_pages:
        print("first_pages_excerpt:")
        print(first_pages[:1800])

    hit_snippets = snippets(all_text)
    if hit_snippets:
        print("hit_snippets:")
        for item in hit_snippets:
            print(f"- {item[:700]}")


def inspect_zip(path: Path) -> None:
    print(f"\n=== ZIP: {path} ===")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            print(f"entries: {len(names)}")
            for name in names[:80]:
                info = archive.getinfo(name)
                print(f"- {name} ({info.file_size} bytes)")
            if len(names) > 80:
                print(f"... {len(names) - 80} more entries")

            for name in names:
                if not name.lower().endswith(".pdf"):
                    continue
                try:
                    with archive.open(name) as pdf_file:
                        data = BytesIO(pdf_file.read())
                    inspect_pdf_stream(f"{path.name}!{name}", data)
                except Exception as exc:
                    print(f"ERROR: cannot inspect {name} inside {path.name}: {exc}")
    except Exception as exc:
        print(f"ERROR: cannot read ZIP: {exc}")


def main() -> None:
    print(f"downloads: {DOWNLOADS}")
    pdfs = sorted(DOWNLOADS.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    zips = sorted(DOWNLOADS.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"pdf_count: {len(pdfs)}")
    print(f"zip_count: {len(zips)}")

    for path in pdfs:
        inspect_pdf_stream(str(path), str(path))

    for path in zips:
        inspect_zip(path)


if __name__ == "__main__":
    main()
