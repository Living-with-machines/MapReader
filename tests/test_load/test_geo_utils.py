import pytest
from pytest import approx
from mapreader.load import geo_utils, loader
from pathlib import Path

@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"

def test_extractGeoInfo(sample_dir):
    image_ID = "cropped_101200740.27.tif"
    image_path = f"{sample_dir}/{image_ID}"
    shape, crs, coord = geo_utils.extractGeoInfo(image_path)
    assert shape == (9,9,3)
    assert crs == "EPSG:27700"
    assert coord == approx(
        (534348, 192378, 534349, 192379), rel=1e-0)


def test_reproject(sample_dir):
    image_ID = "cropped_101200740.27.tif"
    image_path = f"{sample_dir}/{image_ID}"
    _, _, new_crs, reprojected_coord, size_in_m = geo_utils.reprojectGeoInfo(
        image_path, calc_size_in_m="gc"
    )
    assert new_crs == "EPSG:4326"
    assert reprojected_coord == approx((-0.061, 51.6142, -0.0610, 51.614), rel=1e-2)
    print(size_in_m)
    assert size_in_m == approx((0.5904, 0.6209, 0.594, 0.6209), rel=1e-2)

def test_versus_loader(sample_dir):
    image_ID = "cropped_101200740.27.tif"
    image_path = f"{sample_dir}/{image_ID}"
    shape, _, _, reprojected_coords, size_in_m = geo_utils.reprojectGeoInfo(image_path, calc_size_in_m="great-circle")
    geotiff = loader(image_path)
    geotiff.add_geo_info()
    assert geotiff.parents["cropped_101200740.27.tif"]["shape"] == shape
    assert geotiff.parents["cropped_101200740.27.tif"]["coordinates"] == approx(reprojected_coords)
    loader_size_in_m, _, _ = geotiff._calc_pixel_height_width("cropped_101200740.27.tif", method = "great-circle",)
    assert loader_size_in_m == approx(size_in_m)
    
