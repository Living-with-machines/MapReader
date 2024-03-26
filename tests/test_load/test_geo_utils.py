from __future__ import annotations

from pathlib import Path

import pytest
from pytest import approx

from mapreader.load import geo_utils, loader


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"


def test_extractGeoInfo(sample_dir):
    image_id = "cropped_geo.tif"
    image_path = f"{sample_dir}/{image_id}"
    shape, crs, coord = geo_utils.extractGeoInfo(image_path)
    assert shape == (9, 9, 3)
    assert crs == "EPSG:27700"
    assert coord == approx((534348, 192378, 534349, 192379), rel=1e-0)


def test_extractGeoInfo_errors(sample_dir):
    image_id = "cropped_74488689.png"
    image_path = f"{sample_dir}/{image_id}"
    with pytest.raises(ValueError, match="No coordinates found"):
        geo_utils.extractGeoInfo(image_path)


def test_reproject(sample_dir):
    image_id = "cropped_geo.tif"
    image_path = f"{sample_dir}/{image_id}"
    _, _, new_crs, reprojected_coord, size_in_m = geo_utils.reproject_geo_info(
        image_path, calc_size_in_m="gc"
    )
    assert new_crs == "EPSG:4326"
    assert reprojected_coord == approx((-0.061, 51.6142, -0.0610, 51.614), rel=1e-2)
    print(size_in_m)
    assert size_in_m == approx((0.62, 0.62, 0.62, 0.62), rel=1e-2)


def test_versus_loader(sample_dir):
    image_id = "cropped_geo.tif"
    image_path = f"{sample_dir}/{image_id}"
    shape, _, _, reprojected_coords, size_in_m = geo_utils.reproject_geo_info(
        image_path, calc_size_in_m="great-circle"
    )
    geotiff = loader(image_path)
    geotiff.add_geo_info()
    assert geotiff.parents["cropped_geo.tif"]["shape"] == shape
    assert geotiff.parents["cropped_geo.tif"]["coordinates"] == approx(
        reprojected_coords
    )
    loader_size_in_m, _, _ = geotiff._calc_pixel_height_width(
        "cropped_geo.tif",
        method="great-circle",
    )
    assert loader_size_in_m == approx(size_in_m)
