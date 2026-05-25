from __future__ import annotations

from pathlib import Path


def download_if_missing(url: str, destination: str | Path, *, chunk_size: int = 1024 * 1024) -> Path:
    """Download a source artifact once and reuse it on repeated runs."""
    path = Path(destination)
    if path.exists() and path.stat().st_size > 0:
        return path

    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(path)
    return path
