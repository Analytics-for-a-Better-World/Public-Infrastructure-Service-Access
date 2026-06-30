from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader


PATHS = [
    Path(r"C:\Users\joaqu\OneDrive - UvA\Parvathy\TimorLeste_Replication_Pack\references\Dissertation_PKKrishnakumari.pdf"),
    Path(r"C:\Users\joaqu\Downloads\Dissertation_PKKrishnakumari.pdf"),
    Path(r"C:\Users\joaqu\OneDrive - UvA\Parvathy\TimorLeste_Replication_Pack\references\Master_Thesis_JoyceAntonissen.pdf"),
    Path(r"C:\github\Public-Infrastructure-Service-Access\publications\Joyce_Optimisation_Model.pdf"),
    Path(r"C:\Users\joaqu\OneDrive - UvA\Parvathy\TimorLeste_Replication_Pack\deck\timor_leste_replication_deck.pdf"),
    Path(r"C:\Users\joaqu\OneDrive - UvA\Parvathy\TimorLeste_Replication_Pack\deck\timor_leste_replication_deck.tex"),
    Path(r"C:\Users\joaqu\OneDrive - UvA\Parvathy\Vietnam_replication\deck\vietnam_replication_deck.pdf"),
    Path(r"C:\Users\joaqu\OneDrive - UvA\Parvathy\Vietnam_replication\deck\vietnam_replication_deck.tex"),
    Path(r"C:\Users\joaqu\OneDrive - UvA\Parvathy\Vietnam_replication\figures\vietnam_polygons_osm_basemap.pdf"),
]

TERMS = [
    "parvathy",
    "krishnakumari",
    "joyce",
    "antonissen",
    "fleur",
    "theulen",
    "timor",
    "leste",
    "vietnam",
    "maximal covering",
    "optimization",
    "optimisation",
    "greedy",
    "grasp",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_info(path: Path) -> dict[str, object]:
    reader = PdfReader(str(path))
    metadata = reader.metadata or {}
    page_sizes = []
    texts = []
    for idx, page in enumerate(reader.pages[:5], start=1):
        box = page.mediabox
        page_sizes.append(
            f"p{idx}:{float(box.width):.1f}x{float(box.height):.1f}pt"
        )
        try:
            texts.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - diagnostic script
            texts.append(f"[extract error: {exc}]")
    all_text = "\n".join(texts)
    lowered = all_text.lower()
    counts = {term: lowered.count(term) for term in TERMS if lowered.count(term)}
    return {
        "pages": len(reader.pages),
        "sizes": page_sizes,
        "title": normalize(str(metadata.get("/Title") or "")),
        "author": normalize(str(metadata.get("/Author") or "")),
        "creator": normalize(str(metadata.get("/Creator") or "")),
        "producer": normalize(str(metadata.get("/Producer") or "")),
        "created": normalize(str(metadata.get("/CreationDate") or "")),
        "modified": normalize(str(metadata.get("/ModDate") or "")),
        "term_hits_first5": counts,
        "excerpt": normalize(all_text)[:1000],
    }


def tex_info(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = re.search(r"\\title(?:\[[^\]]*\])?\{([^}]*)\}", text)
    subtitle = re.search(r"\\subtitle\{([^}]*)\}", text)
    date = re.search(r"\\date\{([^}]*)\}", text)
    figroot = re.search(r"\\newcommand\{\\figroot\}\{([^}]*)\}", text)
    includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text)
    resolved_includes = []
    for include in includes:
        resolved = include
        if figroot:
            resolved = resolved.replace(r"\figroot", figroot.group(1))
        resolved = resolved.replace("/", "\\")
        if re.match(r"^[A-Za-z]:\\", resolved):
            resolved_path = Path(resolved)
        else:
            resolved_path = path.parent / resolved
        resolved_includes.append((include, str(resolved_path), resolved_path.exists()))

    return {
        "title": title.group(1) if title else "",
        "subtitle": subtitle.group(1) if subtitle else "",
        "date": date.group(1) if date else "",
        "frames": len(re.findall(r"\\begin\{frame\}", text)),
        "frame_like": len(re.findall(r"\\begin\{frame\}", text)),
        "figroot": figroot.group(1) if figroot else "",
        "includegraphics": resolved_includes,
        "todo_count": len(re.findall(r"\bTODO\b", text, flags=re.IGNORECASE)),
        "local_path_mentions": sorted(set(re.findall(r"[A-Za-z]:[/\\][^\s}]+", text)))[:20],
    }


def print_dict(info: dict[str, object], indent: str = "  ") -> None:
    for key, value in info.items():
        if isinstance(value, list):
            print(f"{indent}{key}:")
            for item in value:
                print(f"{indent}  - {item}")
        elif value:
            print(f"{indent}{key}: {value}")


def main() -> None:
    for path in PATHS:
        print(f"\n=== {path} ===")
        if not path.exists():
            print("  MISSING")
            continue
        stat = path.stat()
        print(f"  size: {stat.st_size}")
        print(f"  modified: {stat.st_mtime}")
        print(f"  sha256: {sha256(path)}")
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                print_dict(pdf_info(path))
            elif suffix == ".tex":
                print_dict(tex_info(path))
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
