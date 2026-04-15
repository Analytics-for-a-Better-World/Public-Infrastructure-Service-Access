from __future__ import annotations

import json
import re
from pathlib import Path

from openai import OpenAI


def parse_epsg(value: object) -> int:
    '''
    Convert an EPSG value returned by the model into an integer code.

    Accepts values like:
    - 32648
    - '32648'
    - 'EPSG:32648'
    - 'epsg 32648'

    Parameters
    ----------
    value
        Raw EPSG value returned by the model.

    Returns
    -------
    int
        Parsed EPSG integer.

    Raises
    ------
    ValueError
        If no integer EPSG code can be extracted.
    '''
    if isinstance(value, int):
        return value

    text = str(value).strip()
    match = re.search(r'(\d{4,6})', text)
    if match is None:
        raise ValueError(f'Could not parse projected_epsg from {value!r}')

    return int(match.group(1))


def generate_country_config_module(
    country_code: str,
    countries_dir: str | Path,
    *,
    model: str = 'gpt-5.4',
    overwrite: bool = False,
) -> Path:
    '''
    Generate a new country config module using the OpenAI API.

    Parameters
    ----------
    country_code
        Country identifier, for example 'laos' or 'vietnam'.
    countries_dir
        Directory where country modules are stored.
    model
        OpenAI model name.
    overwrite
        Whether to overwrite an existing module.

    Returns
    -------
    Path
        Path to the generated module.
    '''
    client = OpenAI()

    prompt = f'''
Return only valid JSON.

Country identifier: {country_code!r}

Return exactly these keys:
- iso3
- iso2
- country_name
- country_slug
- projected_epsg
- worldpop_filename

Rules:
- country_slug must be a lowercase python slug with underscores
- projected_epsg must be an integer code only, for example 32648, not "EPSG:32648"
- worldpop_filename must look like iso3 lowercase plus _ppp_2020.tif
- do not include markdown
- do not include explanations
'''

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    payload = json.loads(response.output_text)

    iso3 = str(payload['iso3']).upper()
    iso2 = str(payload['iso2']).upper()
    country_name = str(payload['country_name']).strip()
    country_slug = str(payload['country_slug']).strip().lower()
    projected_epsg = parse_epsg(payload['projected_epsg'])
    worldpop_filename = str(payload['worldpop_filename']).strip()

    module_path = Path(countries_dir) / f'{country_slug}.py'
    if module_path.exists() and not overwrite:
        raise FileExistsError(f'Config already exists: {module_path}')

    module_text = (
        'from countries.base import build_config\n\n\n'
        'CFG = build_config(\n'
        '    {\n'
        f"        'iso3': {iso3!r},\n"
        f"        'iso2': {iso2!r},\n"
        f"        'country_name': {country_name!r},\n"
        f"        'country_slug': {country_slug!r},\n"
        f"        'projected_epsg': {projected_epsg},\n"
        "        'distance_threshold_km': 300.0,\n"
        "        'geofabrik_region': 'asia',\n"
        f"        'worldpop_filename': {worldpop_filename!r},\n"
        "        'plot_title_suffix': 'roads by class, population points, and health facilities',\n"
        "        'candidate_grid_spacing_m': 5000.0,\n"
        "        'candidate_max_snap_dist_m': 5000.0,\n"
        '    }\n'
        ')\n'
    )

    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(module_text, encoding='utf-8')

    return module_path