import abw_distance_matrix
from abw_distance_matrix.api import load_cfg, normalize_country_code


def test_package_import_is_lightweight() -> None:
    assert abw_distance_matrix.__version__ == "0.1.0"


def test_public_api_loads_country_configs() -> None:
    assert normalize_country_code("timor-leste") == "timor_leste"
    cfg = load_cfg("tls")
    assert cfg.iso3 == "TLS"
