from __future__ import annotations

import importlib
from pathlib import Path


_ALIAS_MAP: dict[str, str] = {
    'netherlands': 'netherlands',
    'nld': 'netherlands',
    'nl': 'netherlands',
    'portugal': 'portugal',
    'prt': 'portugal',
    'pt': 'portugal',
    'timor_leste': 'timor_leste',
    'timor-leste': 'timor_leste',
    'tls': 'timor_leste',
    'tl': 'timor_leste',
    'vietnam': 'vietnam',
    'viet_nam': 'vietnam',
    'vnm': 'vietnam',
    'vn': 'vietnam',
    'laos': 'laos',
    'lao': 'laos',
    'la': 'laos',
    'monaco': 'monaco',
    'mco': 'monaco',
    'mc': 'monaco',
    'nauru': 'nauru',
    'nru': 'nauru',
    'nr': 'nauru',
    'tuvalu': 'tuvalu',
    'tuv': 'tuvalu',
    'tv': 'tuvalu',
    'liechtenstein': 'liechtenstein',
    'lie': 'liechtenstein',
    'li': 'liechtenstein',
    'marshall_islands': 'marshall_islands',
    'marshall': 'marshall_islands',
    'mhl': 'marshall_islands',
    'mh': 'marshall_islands',
}


def normalize_country_code(country_code: str) -> str:
    """Normalize a country code or name to a module key."""
    return country_code.strip().lower().replace('-', '_').replace(' ', '_')



def resolve_country_module_name(country_code: str) -> str:
    """Resolve aliases such as ``nld`` or ``timor-leste`` to module names."""
    normalized = normalize_country_code(country_code)
    return _ALIAS_MAP.get(normalized, normalized)



def load_cfg(country_code: str) -> object:
    """Load an existing country config, or generate it if missing."""
    module_name = resolve_country_module_name(country_code)

    try:
        module = importlib.import_module(f'countries.{module_name}')
        return module.CFG
    except ModuleNotFoundError:
        from distance_pipeline.use_openai import generate_country_config_module

        try:
            generated_path = generate_country_config_module(
                country_code=country_code,
                countries_dir=Path(__file__).resolve().parents[1] / 'countries',
            )
        except Exception as exc:
            raise RuntimeError(
                f'Could not generate config for {country_code!r}: {exc}'
            ) from exc

        importlib.invalidate_caches()
        module = importlib.import_module(f'countries.{generated_path.stem}')
        return module.CFG
