from pathlib import Path
from time import perf_counter as pc
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


from pathlib import Path
from time import perf_counter as pc
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


def validate_pbf_file(pbf_path: str | Path) -> None:
    '''
    Validate that a local .osm.pbf file exists and does not look like HTML or XML.
    '''
    path = Path(pbf_path)

    if not path.exists():
        raise FileNotFoundError(f'PBF file not found: {path}')

    if path.stat().st_size < 1024:
        raise ValueError(f'PBF file is suspiciously small: {path}')

    with path.open('rb') as f:
        head = f.read(256).lower()

    if head.startswith(b'<!doctype html') or head.startswith(b'<html'):
        raise ValueError(f'PBF file is actually an HTML page: {path}')

    if head.startswith(b'<?xml'):
        raise ValueError(f'PBF file is actually an XML document: {path}')


def download_file(
    url: str,
    destination: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    '''
    Download a file if it does not already exist, and validate PBF files.
    '''
    t0 = pc()

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        if verbose:
            print(f'Using existing file: {destination}')
        return destination

    if verbose:
        print(f'Downloading {url}')
        print(f'To {destination}')

    try:
        urlretrieve(url, destination)
    except HTTPError as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f'Failed to download {url} to {destination}, HTTP status {exc.code}.'
        ) from exc
    except URLError as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f'Failed to download {url} to {destination}, network error: {exc.reason}.'
        ) from exc
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    if destination.suffixes[-2:] == ['.osm', '.pbf']:
        try:
            validate_pbf_file(destination)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    if verbose:
        size_mb = destination.stat().st_size / 1024**2
        print(f'Download completed in {pc() - t0:.2f} seconds, size {size_mb:.2f} MB')

    return destination