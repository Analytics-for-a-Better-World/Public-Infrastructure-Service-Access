from __future__ import annotations

import json
import re
from pathlib import Path

from openai import OpenAI


VALID_GEOFABRIK_REGIONS: set[str] = {
    'africa',
    'asia',
    'australia-oceania',
    'central-america',
    'europe',
    'north-america',
    'south-america',
}

WORLDPOP_FILENAME_PATTERN = re.compile(r'^[a-z]{3}_ppp_2020\.tif$')
COUNTRY_SLUG_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
EPSG_PATTERN = re.compile(r'(\d{4,6})')


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
    match = EPSG_PATTERN.search(text)
    if match is None:
        raise ValueError(f'Could not parse projected_epsg from {value!r}')

    return int(match.group(1))


def parse_geofabrik_region(value: object) -> str:
    '''
    Normalize and validate a Geofabrik region string.

    Parameters
    ----------
    value
        Raw region value returned by the model.

    Returns
    -------
    str
        Normalized Geofabrik region.

    Raises
    ------
    ValueError
        If the value is not a supported Geofabrik region.
    '''
    region = str(value).strip().lower()
    if region not in VALID_GEOFABRIK_REGIONS:
        raise ValueError(
            f'Invalid geofabrik_region {value!r}. '
            f'Expected one of: {sorted(VALID_GEOFABRIK_REGIONS)!r}'
        )
    return region


def validate_country_slug(value: object) -> str:
    '''
    Validate and normalize a country slug for use as a Python module name.

    Parameters
    ----------
    value
        Raw slug value returned by the model.

    Returns
    -------
    str
        Validated lowercase country slug.

    Raises
    ------
    ValueError
        If the slug is not a safe Python style module name.
    '''
    slug = str(value).strip().lower()
    if not COUNTRY_SLUG_PATTERN.fullmatch(slug):
        raise ValueError(
            f'Invalid country_slug {value!r}. '
            "Expected something like 'timor_leste' or 'luxembourg'."
        )
    return slug


def validate_worldpop_filename(value: object, iso3: str) -> str:
    '''
    Validate the expected WorldPop filename.

    Parameters
    ----------
    value
        Raw filename value returned by the model.
    iso3
        The validated ISO3 code used to cross check the filename prefix.

    Returns
    -------
    str
        Validated WorldPop filename.

    Raises
    ------
    ValueError
        If the filename does not match the expected naming convention.
    '''
    filename = str(value).strip().lower()
    if not WORLDPOP_FILENAME_PATTERN.fullmatch(filename):
        raise ValueError(
            f'Unexpected worldpop_filename {value!r}. '
            "Expected format like 'tls_ppp_2020.tif'."
        )

    expected_prefix = iso3.lower()
    if not filename.startswith(f'{expected_prefix}_'):
        raise ValueError(
            f'worldpop_filename {filename!r} does not match iso3 {iso3!r}.'
        )

    return filename


def validate_iso_code(value: object, *, length: int, field_name: str) -> str:
    '''
    Validate and normalize an ISO code.

    Parameters
    ----------
    value
        Raw ISO code value returned by the model.
    length
        Required code length, usually 2 or 3.
    field_name
        Name of the field being validated.

    Returns
    -------
    str
        Uppercase ISO code.

    Raises
    ------
    ValueError
        If the code is not alphabetic or has the wrong length.
    '''
    code = str(value).strip().upper()
    if len(code) != length or not code.isalpha():
        raise ValueError(
            f'Invalid {field_name} {value!r}. Expected {length} alphabetic characters.'
        )
    return code


def validate_country_name(value: object) -> str:
    '''
    Validate a country name.

    Parameters
    ----------
    value
        Raw country name returned by the model.

    Returns
    -------
    str
        Cleaned country name.

    Raises
    ------
    ValueError
        If the name is empty.
    '''
    name = str(value).strip()
    if not name:
        raise ValueError('country_name must not be empty.')
    return name


def extract_json_object(text: str) -> dict[str, object]:
    '''
    Extract and parse a JSON object from a text response.

    Parameters
    ----------
    text
        Raw text returned by the model.

    Returns
    -------
    dict[str, object]
        Parsed JSON object.

    Raises
    ------
    ValueError
        If a JSON object cannot be found or parsed.
    '''
    text = text.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if match is None:
            raise ValueError('Could not find a JSON object in model output.') from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError('Could not parse JSON object from model output.') from exc

    if not isinstance(payload, dict):
        raise ValueError('Model output must be a JSON object.')

    return payload


def validate_positive_float(value: float, field_name: str) -> float:
    '''
    Validate that a float is strictly positive.

    Parameters
    ----------
    value
        Value to validate.
    field_name
        Name of the field being validated.

    Returns
    -------
    float
        Validated float.

    Raises
    ------
    ValueError
        If the value is not strictly positive.
    '''
    if value <= 0:
        raise ValueError(f'{field_name} must be positive, got {value!r}.')
    return float(value)


def validate_payload(payload: dict[str, object]) -> dict[str, object]:
    '''
    Validate and normalize the model payload.

    Parameters
    ----------
    payload
        Parsed JSON object returned by the model.

    Returns
    -------
    dict[str, object]
        Validated and normalized payload.

    Raises
    ------
    ValueError
        If required keys are missing or values are invalid.
    '''
    required_keys = {
        'iso3',
        'iso2',
        'country_name',
        'country_slug',
        'projected_epsg',
        'geofabrik_region',
        'worldpop_filename',
    }

    missing_keys = sorted(required_keys - payload.keys())
    extra_keys = sorted(payload.keys() - required_keys)

    if missing_keys:
        raise ValueError(f'Model output is missing keys: {missing_keys!r}')
    if extra_keys:
        raise ValueError(f'Model output contains unexpected keys: {extra_keys!r}')

    iso3 = validate_iso_code(payload['iso3'], length=3, field_name='iso3')
    iso2 = validate_iso_code(payload['iso2'], length=2, field_name='iso2')
    country_name = validate_country_name(payload['country_name'])
    country_slug = validate_country_slug(payload['country_slug'])
    projected_epsg = parse_epsg(payload['projected_epsg'])
    geofabrik_region = parse_geofabrik_region(payload['geofabrik_region'])
    worldpop_filename = validate_worldpop_filename(payload['worldpop_filename'], iso3)

    return {
        'iso3': iso3,
        'iso2': iso2,
        'country_name': country_name,
        'country_slug': country_slug,
        'projected_epsg': projected_epsg,
        'geofabrik_region': geofabrik_region,
        'worldpop_filename': worldpop_filename,
    }


def build_country_config_module_text(
    payload: dict[str, object],
    *,
    distance_threshold_km: float = 300.0,
    plot_title_suffix: str = 'roads by class, population points, and health facilities',
    candidate_grid_spacing_m: float = 5000.0,
    candidate_max_snap_dist_m: float = 5000.0,
) -> str:
    '''
    Build the Python module text for a country config.

    Parameters
    ----------
    payload
        Validated country configuration payload.
    distance_threshold_km
        Default maximum distance threshold in kilometers.
    plot_title_suffix
        Suffix appended to the map title.
    candidate_grid_spacing_m
        Default grid spacing for candidate sites in meters.
    candidate_max_snap_dist_m
        Default maximum snapping distance for candidate sites in meters.

    Returns
    -------
    str
        Python source code for the generated module.
    '''
    distance_threshold_km = validate_positive_float(
        distance_threshold_km,
        'distance_threshold_km',
    )
    candidate_grid_spacing_m = validate_positive_float(
        candidate_grid_spacing_m,
        'candidate_grid_spacing_m',
    )
    candidate_max_snap_dist_m = validate_positive_float(
        candidate_max_snap_dist_m,
        'candidate_max_snap_dist_m',
    )

    iso3 = str(payload['iso3'])
    iso2 = str(payload['iso2'])
    country_name = str(payload['country_name'])
    country_slug = str(payload['country_slug'])
    projected_epsg = int(payload['projected_epsg'])
    geofabrik_region = str(payload['geofabrik_region'])
    worldpop_filename = str(payload['worldpop_filename'])
    plot_title_suffix = str(plot_title_suffix).strip()

    if not plot_title_suffix:
        raise ValueError('plot_title_suffix must not be empty.')

    return (
        'from countries.base import build_config\n\n\n'
        'CFG = build_config(\n'
        '    {\n'
        f"        'iso3': {iso3!r},\n"
        f"        'iso2': {iso2!r},\n"
        f"        'country_name': {country_name!r},\n"
        f"        'country_slug': {country_slug!r},\n"
        f"        'projected_epsg': {projected_epsg},\n"
        f"        'distance_threshold_km': {distance_threshold_km},\n"
        f"        'geofabrik_region': {geofabrik_region!r},\n"
        f"        'worldpop_filename': {worldpop_filename!r},\n"
        f"        'plot_title_suffix': {plot_title_suffix!r},\n"
        f"        'candidate_grid_spacing_m': {candidate_grid_spacing_m},\n"
        f"        'candidate_max_snap_dist_m': {candidate_max_snap_dist_m},\n"
        '    }\n'
        ')\n'
    )


def generate_country_config_module(
    country_code: str,
    countries_dir: str | Path,
    *,
    model: str = 'gpt-5.4',
    overwrite: bool = False,
    distance_threshold_km: float = 300.0,
    plot_title_suffix: str = 'roads by class, population points, and health facilities',
    candidate_grid_spacing_m: float = 5000.0,
    candidate_max_snap_dist_m: float = 5000.0,
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
    distance_threshold_km
        Default maximum distance threshold in kilometers.
    plot_title_suffix
        Suffix appended to the map title.
    candidate_grid_spacing_m
        Default grid spacing for candidate sites in meters.
    candidate_max_snap_dist_m
        Default maximum snapping distance for candidate sites in meters.

    Returns
    -------
    Path
        Path to the generated module.

    Raises
    ------
    FileExistsError
        If the target module already exists and overwrite is False.
    ValueError
        If the model output is missing required fields or contains invalid values.
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
- geofabrik_region
- worldpop_filename

Rules:
- country_slug must be a lowercase python slug with underscores
- projected_epsg must be an integer code only, for example 32648
- geofabrik_region must be exactly one of:
  africa, asia, australia-oceania, central-america, europe, north-america, south-america
- worldpop_filename must look like iso3 lowercase plus _ppp_2020.tif
- do not include markdown
- do not include explanations
'''

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    payload = extract_json_object(response.output_text)
    validated_payload = validate_payload(payload)

    module_path = Path(countries_dir) / f"{validated_payload['country_slug']}.py"
    if module_path.exists() and not overwrite:
        raise FileExistsError(f'Config already exists: {module_path}')

    module_text = build_country_config_module_text(
        validated_payload,
        distance_threshold_km=distance_threshold_km,
        plot_title_suffix=plot_title_suffix,
        candidate_grid_spacing_m=candidate_grid_spacing_m,
        candidate_max_snap_dist_m=candidate_max_snap_dist_m,
    )

    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(module_text, encoding='utf-8')

    return module_path