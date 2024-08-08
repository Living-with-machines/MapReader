from __future__ import annotations

import logging
import os
from pathlib import Path
from random import randint

import geopandas as geopd
import pandas as pd
import PIL
import pytest
from PIL import Image
from pytest import approx
from shapely.geometry import Polygon

from mapreader.load.images import MapImages

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s:%(message)s")


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"


@pytest.fixture
def image_id():
    return "cropped_74488689.png"


@pytest.fixture
def init_maps(sample_dir, image_id, tmp_path):
    """Initializes MapImages object (with metadata from csv and patches).

    Returns
    -------
    tuple
        maps (MapImages object), parent_list (== image_id) and patch_list (list of patches).
    """
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.patchify_all(patch_size=3, path_save=tmp_path)  # gives 9 patches
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()

    return maps, parent_list, patch_list


@pytest.fixture
def init_dataframes(sample_dir, image_id, tmp_path):
    """Initializes MapImages object (with metadata from csv and patches) and creates parent and patch dataframes.

    Returns
    -------
    tuple
        path to parent and patch dataframes
    """
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.patchify_all(patch_size=3, path_save=tmp_path)  # gives 9 patches
    maps.add_center_coord(tree_level="parent")
    maps.add_patch_polygons()
    maps.calc_pixel_stats()
    _, _ = maps.convert_images(save=True)
    assert os.path.isfile("./parent_df.csv")
    assert os.path.isfile("./patch_df.csv")

    return "./parent_df.csv", "./patch_df.csv"


@pytest.fixture
def ts_metadata_keys():
    return ["name", "url", "coordinates", "crs", "published_date", "grid_bb", "polygon"]


@pytest.fixture
def metadata_df():
    return pd.DataFrame(
        {
            "name": ["file1.png", "file2.png", "file3.png"],
            "coord": [(1.1, 1.5), (2.1, 1.0), (3.1, 4.5)],
            "other": [1, 2, 3],
        }
    )


@pytest.fixture
def metadata_keys():
    return ["parent_id", "image_path", "shape", "name", "coord", "other"]


@pytest.fixture
def matching_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png", "file3.png"]
    for file in files:
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        Image.new("RGB", (9, 9), color=rand_color).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep=",")
    metadata_df.to_csv(f"{test_path}/metadata_df.tsv", sep="\t")
    metadata_df.to_excel(f"{test_path}/metadata_df.xlsx")
    return test_path


@pytest.fixture
def extra_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png"]
    for file in files:
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        Image.new("RGB", (9, 9), color=rand_color).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep=",")
    return test_path


@pytest.fixture
def missing_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png", "file3.png", "file4.png"]
    for file in files:
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        Image.new("RGB", (9, 9), color=rand_color).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep=",")
    return test_path


# --- test init ---


def test_init_png(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0
    assert isinstance(maps, MapImages)
    str(maps)
    len(maps)


def test_init_png_grayscale(sample_dir):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0
    assert isinstance(maps, MapImages)
    maps.add_shape()
    assert maps.parents[image_id]["shape"] == (9, 9, 1)


def test_init_tiff(sample_dir):
    image_id = "cropped_non_geo.tif"
    tiffs = MapImages(f"{sample_dir}/{image_id}")
    assert len(tiffs) == 1
    assert isinstance(tiffs, MapImages)


def test_init_geotiff(sample_dir):
    image_id = "cropped_geo.tif"
    geotiffs = MapImages(f"{sample_dir}/{image_id}")
    assert len(geotiffs) == 1
    assert isinstance(geotiffs, MapImages)


def test_init_parent_path(sample_dir, image_id, caplog):
    caplog.set_level(logging.INFO)
    maps = MapImages(
        f"{sample_dir}/{image_id}",
        tree_level="patch",
        parent_path=f"{sample_dir}/{image_id}",
    )
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 1

    # without passing tree level should get warning

    maps = MapImages(f"{sample_dir}/{image_id}", parent_path=f"{sample_dir}/{image_id}")
    assert "Ignoring `parent_path` as `tree_level`  is set to 'parent'." in caplog.text
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0


def test_init_tiff_32bit_error(sample_dir):
    image_id = "cropped_32bit.tif"
    with pytest.raises(NotImplementedError, match="Image mode"):
        MapImages(f"{sample_dir}/{image_id}")


def test_init_non_image_error(sample_dir):
    file_name = "ts_downloaded_maps.csv"
    with pytest.raises(PIL.UnidentifiedImageError, match="not an image"):
        MapImages(f"{sample_dir}/{file_name}")


def test_init_fake_tree_level_error(sample_dir, image_id):
    with pytest.raises(ValueError, match="parent or patch"):
        MapImages(f"{sample_dir}/{image_id}", tree_level="fake")


# --- test ``add_metadata`` ---

# first test ``add_metadata`` works for png files


def test_add_metadata(sample_dir, image_id, ts_metadata_keys):
    # metadata csv
    maps_csv = MapImages(f"{sample_dir}/{image_id}")
    maps_csv.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")

    # metadata tsv
    maps_tsv = MapImages(f"{sample_dir}/{image_id}")
    maps_tsv.add_metadata(f"{sample_dir}/ts_downloaded_maps.tsv", delimiter="\t")

    # metadata xlsx
    maps_xlsx = MapImages(f"{sample_dir}/{image_id}")
    maps_xlsx.add_metadata(f"{sample_dir}/ts_downloaded_maps.xlsx")

    for maps in [maps_csv, maps_tsv, maps_xlsx]:
        assert all([k in maps.parents[image_id].keys() for k in ts_metadata_keys])
        assert isinstance(maps.parents[image_id]["coordinates"], tuple)
        assert maps.parents[image_id]["coordinates"] == approx(
            (-4.83, 55.80, -4.21, 56.059), rel=1e-2
        )


# check for mismatched metadata


def test_matching_metadata(matching_metadata_dir, metadata_df, metadata_keys):
    my_files_csv = MapImages(f"{matching_metadata_dir}/*png")
    my_files_csv.add_metadata(f"{matching_metadata_dir}/metadata_df.csv")

    my_files_xlsx = MapImages(f"{matching_metadata_dir}/*png")
    my_files_xlsx.add_metadata(f"{matching_metadata_dir}/metadata_df.xlsx")

    my_files_df = MapImages(f"{matching_metadata_dir}/*png")
    my_files_df.add_metadata(metadata_df)

    for my_files in [my_files_csv, my_files_xlsx, my_files_df]:
        for parent_id in my_files.list_parents():
            assert list(my_files.parents[parent_id].keys()) == metadata_keys


def test_missing_metadata_csv_ignore_mismatch(missing_metadata_dir, metadata_keys):
    my_files = MapImages(f"{missing_metadata_dir}/*png")
    assert len(my_files) == 4
    my_files.add_metadata(
        f"{missing_metadata_dir}/metadata_df.csv", ignore_mismatch=True
    )
    for parent_id in ["file1.png", "file2.png", "file3.png"]:
        assert list(my_files.parents[parent_id].keys()) == metadata_keys
    assert list(my_files.parents["file4.png"].keys()) == [
        "parent_id",
        "image_path",
        "shape",
    ]


def test_missing_metadata_csv_errors(missing_metadata_dir):
    my_files = MapImages(f"{missing_metadata_dir}/*png")
    assert len(my_files) == 4
    with pytest.raises(ValueError, match="missing information"):
        my_files.add_metadata(f"{missing_metadata_dir}/metadata_df.csv")


def test_extra_metadata_csv_ignore_mismatch(extra_metadata_dir, metadata_keys):
    my_files = MapImages(f"{extra_metadata_dir}/*png")
    assert len(my_files) == 2
    my_files.add_metadata(f"{extra_metadata_dir}/metadata_df.csv", ignore_mismatch=True)
    for parent_id in my_files.list_parents():
        assert list(my_files.parents[parent_id].keys()) == metadata_keys


def test_extra_metadata_csv_errors(extra_metadata_dir):
    my_files = MapImages(f"{extra_metadata_dir}/*png")
    assert len(my_files) == 2
    with pytest.raises(ValueError, match="information about non-existent images"):
        my_files.add_metadata(f"{extra_metadata_dir}/metadata_df.csv")


#  test other ``add_metadata`` args


def test_add_metadata_index_col(matching_metadata_dir):
    my_files = MapImages(f"{matching_metadata_dir}/*png")
    assert len(my_files) == 3
    my_files.add_metadata(
        f"{matching_metadata_dir}/metadata_df.csv", index_col="name"
    )  # pass index col arg
    metadata_keys = [
        "parent_id",
        "image_path",
        "shape",
        "Unnamed: 0",
        "coord",
        "other",
        "name",
    ]
    for parent_id in my_files.list_parents():
        assert list(my_files.parents[parent_id].keys()) == metadata_keys


def test_add_metadata_columns(matching_metadata_dir, metadata_df):
    my_files_csv = MapImages(f"{matching_metadata_dir}/*png")
    my_files_csv.add_metadata(
        f"{matching_metadata_dir}/metadata_df.csv", columns=["name", "coord"]
    )

    my_files_xlsx = MapImages(f"{matching_metadata_dir}/*png")
    my_files_xlsx.add_metadata(
        f"{matching_metadata_dir}/metadata_df.xlsx", columns=["name", "coord"]
    )

    my_files_df = MapImages(f"{matching_metadata_dir}/*png")
    my_files_df.add_metadata(metadata_df, columns=["name", "coord"])  # pass columns arg

    metadata_keys = ["parent_id", "image_path", "shape", "name", "coord"]
    for my_files in [my_files_csv, my_files_xlsx, my_files_df]:
        for parent_id in my_files.list_parents():
            assert list(my_files.parents[parent_id].keys()) == metadata_keys
            assert isinstance(my_files.parents[parent_id]["coord"], tuple)


def test_add_metadata_parent(sample_dir, image_id, init_dataframes, ts_metadata_keys):
    parent_df, _ = init_dataframes
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(parent_df, tree_level="parent")
    assert all(
        [
            x in maps.parents[image_id].keys()
            for x in [*ts_metadata_keys, "center_lat", "center_lon"]
        ]
    )
    assert isinstance(maps.parents[image_id]["shape"], tuple)
    assert isinstance(maps.parents[image_id]["coordinates"], tuple)


def test_add_metadata_patch(sample_dir, image_id, init_dataframes, tmp_path):
    parent_df, patch_df = init_dataframes
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    maps.add_metadata(parent_df, tree_level="parent")  # add this too just in case
    maps.add_metadata(patch_df, tree_level="patch")
    patch_id = maps.list_patches()[0]
    expected_cols = [
        "parent_id",
        "shape",
        "pixel_bounds",
        "coordinates",
        "polygon",
        "mean_pixel_R",
        "mean_pixel_A",
        "std_pixel_R",
        "std_pixel_A",
    ]
    assert all([x in maps.patches[patch_id].keys() for x in expected_cols])
    for k in ["shape", "pixel_bounds", "coordinates"]:
        assert isinstance(maps.patches[patch_id][k], tuple)
    assert isinstance(
        "polygon", str
    )  # expect this to be a string, reformed into polygon later


# other ``add_metadata`` errors


def test_metadata_not_found(matching_metadata_dir):
    my_files = MapImages(f"{matching_metadata_dir}/*png")
    assert len(my_files) == 3
    with pytest.raises(ValueError):
        my_files.add_metadata(f"{matching_metadata_dir}/fakefile.csv")


def test_metadata_missing_name_or_image_id(matching_metadata_dir):
    my_files = MapImages(f"{matching_metadata_dir}/*png")
    assert len(my_files) == 3
    incomplete_metadata_df = pd.DataFrame(
        {"coord": [(1.1, 1.5), (2.1, 1.0), (3.1, 4.5)], "other": [1, 2, 3]}
    )
    incomplete_metadata_df.to_csv(
        f"{matching_metadata_dir}/incomplete_metadata_df.csv", sep=","
    )
    with pytest.raises(
        ValueError, match="'name' or 'image_id' should be one of the columns"
    ):
        my_files.add_metadata(incomplete_metadata_df)
    with pytest.raises(
        ValueError, match="'name' or 'image_id' should be one of the columns"
    ):
        my_files.add_metadata(f"{matching_metadata_dir}/incomplete_metadata_df.csv")


# --- test ``add_geo_info`` ---


def test_loader_add_geo_info(sample_dir):
    # check it works for geotiff
    image_id = "cropped_geo.tif"
    geotiffs = MapImages(f"{sample_dir}/{image_id}")
    geotiffs.add_geo_info()
    assert all(
        [k in geotiffs.parents[image_id].keys() for k in ["shape", "coordinates"]]
    )
    assert geotiffs.parents[image_id]["coordinates"] == approx(
        (-0.061, 51.6142, -0.0610, 51.614), rel=1e-2
    )

    # check nothing happens for png/tiff (no metadata)
    image_id = "cropped_74488689.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    keys = list(maps.parents[image_id].keys())
    maps.add_geo_info()
    assert list(maps.parents[image_id].keys()) == keys

    image_id = "cropped_non_geo.tif"
    tiff = MapImages(f"{sample_dir}/{image_id}")
    keys = list(tiff.parents[image_id].keys())
    tiff.add_geo_info()
    assert list(tiff.parents[image_id].keys()) == keys


# --- test patchify ---


def test_patchify_pixels(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    assert len(parent_list) == 1
    assert len(patch_list) == 9
    assert os.path.isfile(f"{tmp_path}/patch-0-0-3-3-#{image_id}#.png")


def test_patchify_pixels_square(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=5, path_save=f"{tmp_path}_square", square_cuts=True)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    print(patch_list, flush=True)
    assert len(parent_list) == 1
    assert len(patch_list) == 4
    assert os.path.isfile(f"{tmp_path}_square/patch-0-0-5-5-#{image_id}#.png")
    assert os.path.isfile(f"{tmp_path}_square/patch-4-4-9-9-#{image_id}#.png")


def test_patchify_meters(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.patchify_all(patch_size=10000, method="meters", path_save=f"{tmp_path}_meters")
    assert os.path.isfile(f"{tmp_path}_meters/patch-0-0-2-2-#{image_id}#.png")
    assert len(maps.list_patches()) == 25


def test_patchify_grayscale(sample_dir, tmp_path):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    assert len(parent_list) == 1
    assert len(patch_list) == 9
    assert os.path.isfile(f"{tmp_path}/patch-0-0-3-3-#{image_id}#.png")


def test_patchify_meters_errors(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    with pytest.raises(ValueError, match="add coordinate information"):
        maps.patchify_all(patch_size=10000, method="meters", path_save=tmp_path)


def test_patchify_pixels_overlap(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=4, path_save=tmp_path, overlap=0.5)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    print(patch_list, flush=True)
    assert len(parent_list) == 1
    assert len(patch_list) == 25
    assert os.path.isfile(f"{tmp_path}/patch-0-0-4-4-#{image_id}#.png")
    assert os.path.isfile(f"{tmp_path}/patch-2-2-6-6-#{image_id}#.png")
    assert os.path.isfile(f"{tmp_path}/patch-6-6-9-9-#{image_id}#.png")
    assert os.path.isfile(f"{tmp_path}/patch-8-8-9-9-#{image_id}#.png")


# --- test other functions ---


def test_load_patches(init_maps, sample_dir, tmp_path):
    maps, _, _ = init_maps

    # create tiff patches
    geotiff_path = f"{sample_dir}/cropped_geo.tif"
    tiff_maps = MapImages(geotiff_path)
    tiff_maps.add_geo_info()
    tiff_maps.patchify_all(patch_size=3, path_save=f"{tmp_path}_tiffs")

    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 9
    maps.load_patches(f"{tmp_path}_tiffs", parent_paths=geotiff_path, add_geo_info=True)
    assert "coordinates" in maps.parents["cropped_geo.tif"].keys()
    assert len(maps.list_parents()) == 2
    assert len(maps.list_patches()) == 18  # 9 for each

    maps.load_patches(f"{tmp_path}_tiffs", parent_paths=geotiff_path, clear_images=True)
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 9


def test_load_parents(init_maps, image_id, sample_dir):
    maps, _, _ = init_maps

    geotiff_path = f"{sample_dir}/cropped_geo.tif"
    assert len(maps.list_parents()) == 1
    maps.load_parents(geotiff_path, overwrite=False, add_geo_info=True)
    assert len(maps.list_parents()) == 2
    assert all(map in maps.list_parents() for map in [image_id, "cropped_geo.tif"])
    assert "coordinates" in maps.parents["cropped_geo.tif"].keys()

    tiff_path = f"{sample_dir}/cropped_non_geo.tif"
    maps.load_parents(tiff_path, overwrite=True)
    assert len(maps.list_parents()) == 1
    assert maps.list_parents() == ["cropped_non_geo.tif"]


def test_add_shape(init_maps, image_id):
    maps, _, patch_list = init_maps

    maps.parents[image_id].pop("shape")
    assert "shape" not in maps.parents[image_id].keys()
    maps.add_shape(tree_level="parent")
    assert "shape" in maps.parents[image_id].keys()

    maps.patches[patch_list[0]].pop("shape")
    assert "shape" not in maps.patches[patch_list[0]].keys()
    maps.add_shape(tree_level="patch")
    assert "shape" in maps.patches[patch_list[0]].keys()


def test_calc_coords_from_grid_bb(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv", columns=["name", "grid_bb", "crs"]
    )
    assert "coordinates" not in maps.parents[image_id]
    maps.add_coords_from_grid_bb()
    assert "coordinates" in maps.parents[image_id]
    assert maps.parents[image_id]["coordinates"] == approx(
        (-4.83, 55.80, -4.21, 56.059), rel=1e-2
    )


def test_calc_coords_from_grid_bb_warning(sample_dir, image_id, caplog):
    caplog.set_level(logging.INFO)
    maps = MapImages(f"{sample_dir}/{image_id}")
    assert all([x not in maps.parents[image_id] for x in ["coordinates", "grid_bb"]])
    maps.add_coords_from_grid_bb()
    assert "No grid bounding box" in caplog.text
    assert "coordinates" not in maps.parents[image_id]


def test_calc_coords_from_grid_bb_error(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv", columns=["name", "grid_bb", "crs"]
    )
    assert "coordinates" not in maps.parents[image_id]
    maps.parents[image_id]["grid_bb"] = 123
    with pytest.raises(ValueError, match="Unexpected grid_bb"):
        maps.add_coords_from_grid_bb()


def test_coord_functions(init_maps, image_id, sample_dir, caplog):
    caplog.set_level(logging.INFO)
    # test for png with added metadata
    maps, _, patch_list = init_maps
    maps.add_center_coord()
    assert "dlon" in maps.parents[image_id].keys()
    assert "center_lon" in maps.patches[patch_list[0]].keys()

    # test for geotiff with added geoinfo
    image_id = "cropped_geo.tif"
    geotiffs = MapImages(f"{sample_dir}/{image_id}")
    geotiffs.add_geo_info()
    geotiffs.add_coord_increments()
    geotiffs.add_center_coord(tree_level="parent")
    assert "dlon" in geotiffs.parents[image_id].keys()
    assert "center_lon" in geotiffs.parents[image_id].keys()

    # test for tiff with no geo info (i.e. no coords so nothing should happen)
    image_id = "cropped_non_geo.tif"
    tiffs = MapImages(f"{sample_dir}/{image_id}")
    keys = list(tiffs.parents[image_id].keys())
    tiffs.add_coord_increments()
    tiffs.add_center_coord(tree_level="parent")
    assert "'coordinates' could not be found" in caplog.text
    assert list(tiffs.parents[image_id].keys()) == keys


def test_add_patch_coords(init_maps):
    maps, _, patch_list = init_maps
    maps.patches[patch_list[0]].pop("coordinates")
    assert "coordinates" not in maps.patches[patch_list[0]].keys()
    maps.add_patch_coords()
    assert "coordinates" in maps.patches[patch_list[0]].keys()


def test_add_patch_polygons(init_maps):
    maps, _, patch_list = init_maps
    maps.patches[patch_list[0]].pop("polygon")
    assert "polygon" not in maps.patches[patch_list[0]].keys()
    maps.add_patch_polygons()
    assert "polygon" in maps.patches[patch_list[0]].keys()
    assert isinstance(maps.patches[patch_list[0]]["polygon"], Polygon)


def test_save_patches_as_geotiffs(init_maps):
    maps, _, _ = init_maps
    maps.save_patches_as_geotiffs()
    patch_id = maps.list_patches()[0]
    assert "geotiff_path" in maps.patches[patch_id].keys()
    assert os.path.isfile(maps.patches[patch_id]["geotiff_path"])


def test_save_patches_as_geotiffs_edge_patches(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.patchify_all(patch_size=8, path_save=tmp_path)
    patch_list = maps.list_patches()
    assert len(maps.list_patches()) == 4
    maps.save_patches_as_geotiffs()
    for patch in patch_list:
        pixel_bounds = maps.patches[patch]["pixel_bounds"]
        patch_width = pixel_bounds[2] - pixel_bounds[0]
        patch_height = pixel_bounds[3] - pixel_bounds[1]
        assert "geotiff_path" in maps.patches[patch].keys()
        assert os.path.isfile(maps.patches[patch]["geotiff_path"])
        img = Image.open(maps.patches[patch]["geotiff_path"])
        assert img.height == patch_height
        assert img.width == patch_width


def test_save_patches_as_geotiffs_grayscale(sample_dir, tmp_path):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    metadata = pd.read_csv(f"{sample_dir}/ts_downloaded_maps.csv", index_col=0)
    metadata.loc[0, "name"] = "cropped_L.png"
    maps.add_metadata(metadata)
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    maps.save_patches_as_geotiffs()
    patch_id = maps.list_patches()[0]
    assert "geotiff_path" in maps.patches[patch_id].keys()
    assert os.path.isfile(maps.patches[patch_id]["geotiff_path"])


def test_save_to_geojson(init_maps, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    maps, _, _ = init_maps
    maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    assert os.path.exists(f"{tmp_path}/patches.geojson")
    geo_df = geopd.read_file(f"{tmp_path}/patches.geojson")
    assert "geometry" in geo_df.columns
    assert str(geo_df.crs.to_string()) == "EPSG:4326"
    assert isinstance(geo_df["geometry"][0], Polygon)

    maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    assert "File already exists" in caplog.text


def test_save_to_geojson_missing_data(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv", columns=["name", "coordinates", "crs"]
    )
    maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    assert os.path.exists(f"{tmp_path}/patches.geojson")
    geo_df = geopd.read_file(f"{tmp_path}/patches.geojson")
    assert "geometry" in geo_df.columns
    assert str(geo_df.crs.to_string()) == "EPSG:4326"
    assert isinstance(geo_df["geometry"][0], Polygon)


def test_save_to_geojson_polygon_strings(
    sample_dir, image_id, init_dataframes, tmp_path
):
    parent_df, patch_df = init_dataframes
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    maps.add_metadata(parent_df, tree_level="parent")
    maps.add_metadata(patch_df, tree_level="patch")
    patch_id = maps.list_patches()[0]
    assert isinstance(maps.patches[patch_id]["polygon"], str)
    maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    assert os.path.exists(f"{tmp_path}/patches.geojson")
    geo_df = geopd.read_file(f"{tmp_path}/patches.geojson")
    assert "geometry" in geo_df.columns
    assert str(geo_df.crs.to_string()) == "EPSG:4326"
    assert isinstance(geo_df["geometry"][0], Polygon)


def test_calc_pixel_stats(init_maps, sample_dir, tmp_path):
    maps, _, patch_list = init_maps
    maps.calc_pixel_stats()
    expected_cols = [
        "mean_pixel_R",
        "mean_pixel_G",
        "mean_pixel_B",
        "mean_pixel_A",
        "std_pixel_R",
        "std_pixel_G",
        "std_pixel_B",
        "std_pixel_A",
    ]
    assert all([col in maps.patches[patch_list[0]].keys() for col in expected_cols])

    # geotiff/tiff will not have alpha channel, so only RGB returned
    image_id = "cropped_geo.tif"
    geotiffs = MapImages(f"{sample_dir}/{image_id}")
    geotiffs.patchify_all(patch_size=3, path_save=tmp_path)
    patch_list = geotiffs.list_patches()
    geotiffs.calc_pixel_stats()
    expected_cols = [
        "mean_pixel_R",
        "mean_pixel_G",
        "mean_pixel_B",
        "std_pixel_R",
        "std_pixel_G",
        "std_pixel_B",
    ]
    assert all([col in geotiffs.patches[patch_list[0]].keys() for col in expected_cols])


def test_loader_convert_images(init_maps):
    maps, _, _ = init_maps
    parent_df, patch_df = maps.convert_images()
    assert parent_df.shape == (1, 13)
    assert patch_df.shape == (9, 7)
    parent_df, patch_df = maps.convert_images(save=True)
    assert os.path.isfile("./parent_df.csv")
    assert os.path.isfile("./patch_df.csv")
    os.remove("./parent_df.csv")
    os.remove("./patch_df.csv")
    parent_df, patch_df = maps.convert_images(save=True, save_format="excel")
    assert os.path.isfile("./parent_df.xlsx")
    assert os.path.isfile("./patch_df.xlsx")
    os.remove("./parent_df.xlsx")
    os.remove("./patch_df.xlsx")


def test_loader_convert_images_errors(init_maps):
    maps, _, _ = init_maps
    with pytest.raises(ValueError, match="``save_format`` should be one of"):
        maps.convert_images(save=True, save_format="json")


def test_save_parents_as_geotiffs(init_maps, sample_dir, image_id):
    maps, _, _ = init_maps
    maps.save_parents_as_geotiffs()
    image_id = image_id.split(".")[0]
    assert os.path.isfile(f"{sample_dir}/{image_id}.tif")


def test_save_parents_as_geotiffs_grayscale(sample_dir, tmp_path):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    metadata = pd.read_csv(f"{sample_dir}/ts_downloaded_maps.csv", index_col=0)
    metadata.loc[0, "name"] = "cropped_L.png"
    maps.add_metadata(metadata)
    maps.save_parents_as_geotiffs()
    assert "geotiff_path" in maps.parents[image_id].keys()
    assert os.path.isfile(maps.parents[image_id]["geotiff_path"])


def test_save_parents_as_geotiffs_error(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    assert "coordinates" not in maps.parents[image_id].keys()
    with pytest.raises(ValueError, match="Cannot locate coordinates"):
        maps.save_parents_as_geotiffs(rewrite=True)


def test_show_sample_png(init_maps, monkeypatch):
    maps, parent_list, patch_list = init_maps
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    maps.show_sample(num_samples=1, tree_level="parent")
    maps.show_sample(num_samples=1, tree_level="patch")


def test_show_sample_grayscale(sample_dir, tmp_path, monkeypatch):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    maps.show_sample(num_samples=1, tree_level="parent")
    maps.show_sample(num_samples=1, tree_level="patch")
