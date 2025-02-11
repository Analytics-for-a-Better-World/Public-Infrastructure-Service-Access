import geopandas as gpd
import pycountry
import pytest
from shapely.geometry import Polygon

from pisa.administrative_area import AdministrativeArea


def get_shape_data_by_country(country: pycountry.ExistingCountries.data_class_base, admin_level: int) -> gpd.GeoDataFrame:
    if admin_level == 0:
        data = {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}
    else:
        data = {
            f"NAME_{admin_level}": ["AreaA", "AreaB"],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
            ],
        }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


# Automatically patch _download_admin_areas to use the dummy downloader.
@pytest.fixture(autouse=True)
def patch_download(mocker):
    mocker.patch.object(
        AdministrativeArea,
        "_download_admin_areas",
        lambda self, country, admin_level: get_shape_data_by_country(country, admin_level)
    )


def test_get_admin_area_names_level_0():
    admin_area = AdministrativeArea("Timor-Leste", admin_level=0)
    names = admin_area.get_admin_area_names()
    assert names == [admin_area.country.name]


def test_get_admin_area_names_level_1():
    admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
    names = admin_area.get_admin_area_names()
    assert names == ["AreaA", "AreaB"]


def test_get_admin_area_boundaries_found():
    admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
    geom = admin_area.get_admin_area_boundaries("AreaA")
    expected = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert geom.equals(expected)


def test_get_admin_area_boundaries_not_found():
    admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
    with pytest.raises(Exception) as excinfo:
        admin_area.get_admin_area_boundaries("NonExistentArea")
    assert "not found" in str(excinfo.value)


def test_get_iso3_country_code():
    admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
    iso_code = admin_area.get_iso3_country_code()
    country = pycountry.countries.get(name="Timor-Leste")
    assert iso_code == country.alpha_3.lower()
