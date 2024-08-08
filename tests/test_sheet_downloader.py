from __future__ import annotations

import logging
import os
from ast import literal_eval
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from pytest import approx
from shapely.geometry import LineString, MultiPolygon, Polygon

from mapreader import SheetDownloader
from mapreader.download.data_structures import GridBoundingBox
from mapreader.download.tile_loading import TileDownloader
from mapreader.download.tile_merging import TileMerger

# Set up logging
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent / "sample_files"


@pytest.fixture
def sheet_downloader(sample_dir):
    test_json = f"{sample_dir}/test_json.json"  # contains 6 features
    download_url = (
        "https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png"
    )
    return SheetDownloader(test_json, download_url)


def test_init(sheet_downloader):
    sd = sheet_downloader
    assert len(sd) == 6
    assert sd.crs == "EPSG:4326"


def test_init_errors(sample_dir):
    test_json = f"{sample_dir}/test_json.json"  # crs changed to EPSG:3857 (note: coordinates are wrong in file)
    download_url = (
        "https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png"
    )
    with pytest.raises(ValueError, match="file not found"):
        SheetDownloader("fake_file.json", download_url)
    with pytest.raises(ValueError, match="string or list of strings"):
        SheetDownloader(test_json, 10)


def test_get_polygons(sheet_downloader):
    sd = sheet_downloader
    sd.get_polygons()
    assert sd.polygons is True
    assert (isinstance(sd.features[i]["polygon"], Polygon) for i in sd.features)


# TODO: add a test for when there are multiple geometries in one of the map's
# features, i.e. when get_polygons is run and only one of the feature's
# polygons is selected => logger warning for "Multiple geometries found in map"


def test_get_grid_bb(sheet_downloader):
    sd = sheet_downloader
    sd.get_grid_bb()
    assert sd.grid_bbs is True
    assert (isinstance(sd.features[i]["grid_bb"], GridBoundingBox) for i in sd.features)
    assert str(sd.features[0]["grid_bb"].lower_corner) == "(14, 8147, 5300)"
    assert str(sd.features[0]["grid_bb"].upper_corner) == "(14, 8150, 5302)"

    sd.get_grid_bb(10)
    assert str(sd.features[0]["grid_bb"].lower_corner) == "(10, 509, 331)"
    assert str(sd.features[0]["grid_bb"].upper_corner) == "(10, 509, 331)"


def test_get_grid_bb_errors(sample_dir):
    test_json = f"{sample_dir}/test_json_epsg3857.json"  # crs changed to EPSG:3857 (note: coordinates are wrong in file)
    download_url = (
        "https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png"
    )
    sd = SheetDownloader(test_json, download_url)
    with pytest.raises(NotImplementedError, match="EPSG:4326"):
        sd.get_grid_bb()


def test_extract_wfs_id_nos(sheet_downloader):
    sd = sheet_downloader
    sd.extract_wfs_id_nos()
    assert sd.wfs_id_nos is True
    assert sd.features[0]["wfs_id_no"] == 16320


def test_extract_published_dates(sheet_downloader):
    sd = sheet_downloader
    sd.extract_published_dates()
    assert sd.published_dates is True
    assert sd.features[0]["properties"]["published_date"] == 1900
    assert (
        sd.features[3]["properties"]["published_date"] == 1896
    )  # metadata has "1894 to 1896" - method takes end date only

    sd.extract_published_dates(date_col=["properties", "YEAR"])  # list of keys
    assert sd.features[0]["properties"]["published_date"] == 2023
    assert sd.features[3]["properties"]["published_date"] == 1894

    sd.extract_published_dates(date_col="test_date")  # keys as str
    assert sd.features[0]["properties"]["published_date"] == 2021
    assert sd.features[3]["properties"]["published_date"] == 2021


def test_published_dates_value_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="as string or list of strings"):
        sd.extract_published_dates(date_col=1)
    with pytest.raises(ValueError, match="not an integer"):
        sd.extract_published_dates(date_col="id")


def test_published_dates_key_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(KeyError):
        sd.extract_published_dates(date_col=["properties", "fake_key"])
    with pytest.raises(KeyError):
        sd.extract_published_dates(date_col="fake_key")  # str instead of list


def test_get_merged_polygon(sheet_downloader):
    sd = sheet_downloader
    sd.get_merged_polygon()
    assert isinstance(sd.merged_polygon, MultiPolygon)


def test_get_minmax_latlon(sheet_downloader, caplog):
    caplog.set_level(logging.INFO)
    sd = sheet_downloader
    sd.get_minmax_latlon()
    assert "Min lat: 51.49344796, max lat: 54.2089733" in caplog.text
    assert "Min lon: -4.7682, max lon: -0.16093917" in caplog.text


# queries


def test_query_by_wfs_ids(sheet_downloader):
    sd = sheet_downloader
    sd.query_map_sheets_by_wfs_ids(16320)  # test single wfs_id
    assert sd.wfs_id_nos is True
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[0]

    sd.query_map_sheets_by_wfs_ids([16320, 16321])  # test list of wfs_ids
    assert len(sd.found_queries) == 2
    assert sd.found_queries == sd.features[:2]

    sd.query_map_sheets_by_wfs_ids(132, append=True)  # test append
    assert len(sd.found_queries) == 3
    assert sd.found_queries[2] == sd.features[3]


def test_query_by_wfs_ids_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="as int or list of ints"):
        sd.query_map_sheets_by_wfs_ids("string")
    with pytest.raises(ValueError, match="as int or list of ints"):
        sd.query_map_sheets_by_wfs_ids(21.4)


def test_query_by_polygon(sheet_downloader):
    sd = sheet_downloader
    polygon = Polygon(
        [
            [-0.98078243, 53.45664144],
            [-0.9806354, 53.48556079],
            [-0.90790637, 53.48540701],
            [-0.90810282, 53.45648782],
            [-0.98078243, 53.45664144],
        ]
    )  # should match to features[0]
    sd.query_map_sheets_by_polygon(polygon)  # test mode = 'within'
    assert sd.polygons is True
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[0]

    sd.query_map_sheets_by_polygon(
        polygon, mode="intersects"
    )  # test mode = 'intersects'
    assert len(sd.found_queries) == 2
    assert sd.found_queries == sd.features[:2]

    another_polygon = Polygon(
        [
            [-0.23045502, 51.49344796],
            [-0.23053988, 51.52237709],
            [-0.16097999, 51.52243594],
            [-0.16093917, 51.49350674],
            [-0.23045502, 51.49344796],
        ]
    )  # should match to features[3]
    sd.query_map_sheets_by_polygon(another_polygon, append=True)  # test append
    assert len(sd.found_queries) == 3
    assert sd.found_queries[2] == sd.features[3]


def test_query_by_polygon_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="pass polygon as shapely.geometry.Polygon"):
        sd.query_map_sheets_by_polygon([1, 2])
    polygon = Polygon(
        [
            [-4.79999994, 54.48000003],
            [-5.39999994, 54.48000003],
            [-5.40999994, 54.74000003],
            [-4.80999994, 54.75000003],
            [-4.79999994, 54.48000003],
        ]
    )  # should match to features[0]

    with pytest.raises(
        NotImplementedError, match='``mode="within"`` or ``mode="intersects"``'
    ):
        sd.query_map_sheets_by_polygon(polygon, mode="fake mode")


def test_query_by_coords(sheet_downloader):
    sd = sheet_downloader
    sd.query_map_sheets_by_coordinates((-0.99, 53.43))
    assert sd.polygons is True
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[1]

    sd.query_map_sheets_by_coordinates((-0.23, 51.5), append=True)  # test append
    assert len(sd.found_queries) == 2
    assert sd.found_queries[1] == sd.features[3]


def test_query_by_coords_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="pass coords as a tuple"):
        sd.query_map_sheets_by_coordinates("string")


def test_query_by_line(sheet_downloader):
    sd = sheet_downloader
    line = LineString([(-0.99, 53.43), (-0.93, 53.46)])
    sd.query_map_sheets_by_line(line)
    assert sd.polygons is True
    assert len(sd.found_queries) == 2
    assert sd.found_queries == sd.features[:2]

    another_line = LineString([(-0.2, 51.5), (-0.21, 51.6)])
    sd.query_map_sheets_by_line(another_line, append=True)  # test append
    assert len(sd.found_queries) == 3
    assert sd.found_queries[2] == sd.features[3]


def test_query_by_line_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="pass line as shapely.geometry.LineString"):
        sd.query_map_sheets_by_line("str")


def test_query_by_string(sheet_downloader):
    sd = sheet_downloader
    sd.query_map_sheets_by_string("Westminster", ["properties", "PARISH"])
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[3]

    sd.query_map_sheets_by_string(
        "Six_Inch_GB_WFS.16320", "id", append=True
    )  # test append + w/ keys as string
    assert len(sd.found_queries) == 2
    assert sd.found_queries[1] == sd.features[0]

    sd.query_map_sheets_by_string("III.SW")  # test w/ no keys
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[1]


def test_query_by_string_value_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="pass ``string`` as a string"):
        sd.query_map_sheets_by_string(10)
    with pytest.raises(ValueError, match="as string or list of strings"):
        sd.query_map_sheets_by_string("Westminster", 10)


def test_query_by_string_key_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(KeyError, match="not found in features dictionary"):
        sd.query_map_sheets_by_string("Nottinghamshire", ["fake_key"])


# download


@pytest.fixture(scope="function")
def mock_response(monkeypatch):
    def mock_download_tiles(self, *args, **kwargs):
        os.makedirs(self.temp_folder, exist_ok=True)
        return

    monkeypatch.setattr(TileDownloader, "download_tiles", mock_download_tiles)

    def mock_merge(self, *args, **kwargs):
        os.makedirs(self.output_folder, exist_ok=True)

        merged_image = Image.new("RGBA", (10, 10))

        if kwargs["file_name"] is None:
            file_name = self._get_output_name(kwargs["grid_bb"])
        else:
            file_name = kwargs["file_name"]

        out_path = f"{self.output_folder}{file_name}.{self.img_output_format[0]}"
        if not kwargs["overwrite"]:
            i = 1
            while os.path.exists(out_path):
                out_path = (
                    f"{self.output_folder}{file_name}_{i}.{self.img_output_format[0]}"
                )
                i += 1
        merged_image.save(out_path, self.img_output_format[1])
        return out_path

    monkeypatch.setattr(TileMerger, "merge", mock_merge)


def test_download_all(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    # zoom level 14
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps_14/"
    metadata_fname = "test_metadata.csv"
    sd.download_all_map_sheets(maps_path, metadata_fname, force=True)
    assert os.path.exists(f"{maps_path}/map_102352861.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 6
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
    ]
    assert all(
        name in list(df["name"])
        for name in [
            "map_101602026.png",
            "map_101602038.png",
            "map_102352861.png",
            "map_91617032.png",
            "map_101603986.png",
            "map_101603986_1.png",
        ]
    )
    # test coords
    assert literal_eval(df.loc[0, "coordinates"]) == approx(
        (-0.98876953125, 53.448806835427575, -0.90087890625, 53.48804553605621),
        rel=1e-6,
    )
    # zoom level 16
    sd.get_grid_bb(16)
    assert sd.grid_bbs is True
    maps_path = tmp_path / "test_maps_16/"
    metadata_fname = "test_metadata.csv"
    sd.download_all_map_sheets(maps_path, metadata_fname, force=True)
    assert os.path.exists(f"{maps_path}/map_102352861.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 6
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
    ]
    print(list(df["name"]))
    assert all(
        name in list(df["name"])
        for name in [
            "map_101602026.png",
            "map_101602038.png",
            "map_102352861.png",
            "map_91617032.png",
            "map_101603986.png",
            "map_101603986_1.png",
        ]
    )


def test_download_all_kwargs(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    # zoom level 14
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps_14/"
    metadata_fname = "test_metadata.csv"
    kwargs = {
        "metadata_to_save": {"test1": ["properties", "test"], "test2": "id"},
        "date_col": "test_date",
        "force": True,
    }
    sd.download_all_map_sheets(maps_path, metadata_fname, **kwargs)
    assert os.path.exists(f"{maps_path}/map_102352861.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 6
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
        "test1",
        "test2",
    ]
    assert all(
        name in list(df["name"])
        for name in [
            "map_101602026.png",
            "map_101602038.png",
            "map_102352861.png",
            "map_91617032.png",
            "map_101603986.png",
            "map_101603986_1.png",
        ]
    )
    # test coords
    assert literal_eval(df.loc[0, "coordinates"]) == approx(
        (-0.98876953125, 53.448806835427575, -0.90087890625, 53.48804553605621),
        rel=1e-6,
    )
    assert df.loc[3, "published_date"] == 2021
    assert df.loc[3, "test1"] == "test"
    assert df.loc[3, "test2"] == "Six_Inch_GB_WFS.132"


def test_download_by_wfs_ids(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    sd.download_map_sheets_by_wfs_ids(
        16320, maps_path, metadata_fname, force=True
    )  # test single wfs_id
    assert os.path.exists(f"{maps_path}/map_101602026.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 1
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
    ]
    assert df.loc[0, "name"] == "map_101602026.png"

    sd.download_map_sheets_by_wfs_ids(
        [16320, 16321], maps_path, metadata_fname, force=True
    )  # test list of wfs_ids
    assert os.path.exists(f"{maps_path}/map_101602038.png")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 2  # should have only downloaded/added one extra map
    assert df.loc[1, "name"] == "map_101602038.png"
    sd.download_map_sheets_by_wfs_ids(
        16320, maps_path, metadata_fname, overwrite=True, force=True
    )  # test overwrite
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 2
    assert df.loc[0, "name"] == "map_101602026.png"
    assert df.loc[1, "name"] == "map_101602038.png"



def test_download_same_image_names(sheet_downloader, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    sd.download_map_sheets_by_wfs_ids(
        [107, 116], maps_path, metadata_fname, force=True
    )  # 107 and 116 both refer to https://maps.nls.uk/view/101603986
    assert os.path.exists(f"{maps_path}/map_101603986.png")
    assert os.path.exists(f"{maps_path}/map_101603986_1.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 2
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
    ]
    assert df.loc[0, "name"] == "map_101603986.png"
    assert df.loc[1, "name"] == "map_101603986_1.png"

    # run again, nothing should happen
    sd.download_map_sheets_by_wfs_ids([107, 116], maps_path, metadata_fname, force=True)
    assert '"map_101603986.png" already exists. Skipping download.' in caplog.text
    assert '"map_101603986_1.png" already exists. Skipping download.' in caplog.text

    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 2

    # now overwrite them, but check we don't add new ones (_2, _3 etc.)
    sd.download_map_sheets_by_wfs_ids(
        [107, 116], maps_path, metadata_fname, overwrite=True, force=True
    )  # 107 and 116 both refer to https://maps.nls.uk/view/101603986
    assert os.path.exists(f"{maps_path}/map_101603986.png")
    assert os.path.exists(f"{maps_path}/map_101603986_1.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 2
    assert df.loc[0, "name"] == "map_101603986.png"
    assert df.loc[1, "name"] == "map_101603986_1.png"


def test_download_by_wfs_ids_errors(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    with pytest.raises(ValueError, match="as int or list of ints"):
        sd.download_map_sheets_by_wfs_ids(
            "string", maps_path, metadata_fname, force=True
        )
    with pytest.raises(ValueError, match="as int or list of ints"):
        sd.download_map_sheets_by_wfs_ids(21.4, maps_path, metadata_fname, force=True)

    with pytest.raises(ValueError, match="No map sheets"):
        sd.download_map_sheets_by_wfs_ids(12, maps_path, metadata_fname)


def test_download_by_polygon(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    polygon = Polygon(
        [
            [-0.98078243, 53.45664144],
            [-0.9806354, 53.48556079],
            [-0.90790637, 53.48540701],
            [-0.90810282, 53.45648782],
            [-0.98078243, 53.45664144],
        ]
    )  # should match to features[0]
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    sd.download_map_sheets_by_polygon(
        polygon, maps_path, metadata_fname, force=True
    )  # test mode = 'within'
    assert os.path.exists(f"{maps_path}/map_101602026.png"), os.listdir(maps_path)
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 1
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
    ]
    assert df.loc[0, "name"] == "map_101602026.png"

    sd.download_map_sheets_by_polygon(
        polygon, maps_path, metadata_fname, mode="intersects", force=True
    )  # test mode = 'intersects', now 2 maps
    assert os.path.exists(f"{maps_path}/map_101602038.png")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 2  # should have only downloaded/added one extra map
    assert df.loc[1, "name"] == "map_101602038.png"


def test_download_by_polygon_errors(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    polygon = Polygon([[0, 1], [1, 2], [2, 3], [3, 4], [0, 1]])
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    with pytest.raises(
        NotImplementedError, match='``mode="within"`` or ``mode="intersects"``'
    ):
        sd.download_map_sheets_by_polygon(
            polygon, maps_path, metadata_fname, mode="fake mode", force=True
        )
    with pytest.raises(ValueError, match="out of map metadata bounds"):
        sd.download_map_sheets_by_polygon(
            polygon, maps_path, metadata_fname, force=True
        )


def test_download_by_coords(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    sd.download_map_sheets_by_coordinates(
        (-0.99, 53.43), maps_path, metadata_fname, force=True
    )
    assert os.path.exists(f"{maps_path}/map_101602038.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 1
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
    ]
    assert df.loc[0, "name"] == "map_101602038.png"


def test_download_by_coords_errors(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    with pytest.raises(ValueError, match="out of map metadata bounds"):
        sd.download_map_sheets_by_coordinates(
            (0, 1), maps_path, metadata_fname, force=True
        )


def test_download_by_line(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    line = LineString([(-0.99, 53.43), (-0.93, 53.46)])
    sd.download_map_sheets_by_line(line, maps_path, metadata_fname, force=True)
    assert os.path.exists(f"{maps_path}/map_101602026.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 2
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
    ]
    assert list(df["name"]) == ["map_101602026.png", "map_101602038.png"]


def test_download_by_line_errors(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    line = LineString([(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="out of map metadata bounds"):
        sd.download_map_sheets_by_line(line, maps_path, metadata_fname, force=True)


def test_download_by_string(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    sd.download_map_sheets_by_string(
        "Westminster", ["properties", "PARISH"], maps_path, metadata_fname, force=True
    )  # test w/ keys list
    assert os.path.exists(f"{maps_path}/map_91617032.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 1
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
    ]
    assert df.loc[0, "name"] == "map_91617032.png"

    sd.download_map_sheets_by_string(
        "Six_Inch_GB_WFS.16320", "id", maps_path, metadata_fname, force=True
    )  # test append + w/ keys as string
    assert os.path.exists(f"{maps_path}/map_101602026.png")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 2
    assert df.loc[1, "name"] == "map_101602026.png"

    sd.download_map_sheets_by_string(
        "III.SW", path_save=maps_path, metadata_fname=metadata_fname, force=True
    )  # test w/ no keys
    assert os.path.exists(f"{maps_path}/map_101602038.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 3
    assert df.loc[2, "name"] == "map_101602038.png"


def test_download_by_string_value_errors(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    with pytest.raises(ValueError, match="pass ``string`` as a string"):
        sd.download_map_sheets_by_string(
            10, path_save=maps_path, metadata_fname=metadata_fname, force=True
        )
    with pytest.raises(ValueError, match="as string or list of strings"):
        sd.download_map_sheets_by_string(
            "Westminster", 10, maps_path, metadata_fname, force=True
        )


def test_download_by_string_key_errors(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    with pytest.raises(KeyError, match="not found in features dictionary"):
        sd.download_map_sheets_by_string(
            "Nottinghamshire", ["fake_key"], maps_path, metadata_fname, force=True
        )


def test_download_by_queries(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    sd.query_map_sheets_by_wfs_ids([16320, 132])  # features[0] and [3]
    sd.query_map_sheets_by_coordinates((-0.23, 51.5), append=True)  # features[3]
    assert len(sd.found_queries) == 2
    sd.download_map_sheets_by_queries(maps_path, metadata_fname, force=True)
    assert os.path.exists(f"{maps_path}/map_101602026.png")
    assert os.path.exists(f"{maps_path}/map_91617032.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    df = pd.read_csv(f"{maps_path}/{metadata_fname}", sep=",", index_col=0)
    assert len(df) == 2
    assert list(df.columns) == [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
    ]
    assert list(df["name"]) == ["map_101602026.png", "map_91617032.png"]


def test_download_by_queries_errors(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.get_grid_bb(14)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    with pytest.raises(ValueError, match="No query results"):
        sd.download_map_sheets_by_queries(maps_path, metadata_fname, force=True)


def test_data_warning_error(sheet_downloader, tmp_path, mock_response):
    sd = sheet_downloader
    sd.features.extend(sd.features * 100)
    sd.get_grid_bb(16)
    maps_path = tmp_path / "test_maps/"
    metadata_fname = "test_metadata.csv"
    with pytest.raises(
        Warning, match="Please confirm download by setting ``force=True``"
    ):
        sd.download_all_map_sheets(maps_path, metadata_fname)
