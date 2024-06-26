from __future__ import annotations

import json
import logging
import os
import re
import shutil
from functools import reduce

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# import cartopy.crs as ccrs - would be good to get this fixed (i think by conda package)
import numpy as np
import pandas as pd
from pyproj.crs import CRS
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union
from tqdm.auto import tqdm

from .downloader_utils import get_grid_bb_from_polygon, get_polygon_from_grid_bb
from .tile_loading import DEFAULT_TEMP_FOLDER, TileDownloader
from .tile_merging import TileMerger

# Set up logging
logger = logging.getLogger(__name__)


class SheetDownloader:
    """
    A class to download map sheets using metadata.
    """

    def __init__(
        self,
        metadata_path: str,
        download_url: str | list,
    ) -> None:
        """
        Initialize SheetDownloader class

        Parameters
        ----------
        metadata_path : str
            path to metadata.json
        download_url : Union[str, list]
            The base URL pattern used to download tiles from the server. This
            should contain placeholders for the x coordinate (``x``), the y
            coordinate (``y``) and the zoom level (``z``).
        """
        self.polygons = False
        self.grid_bbs = False
        self.wfs_id_nos = False
        self.published_dates = False
        self.found_queries = []
        self.merged_polygon = None

        assert isinstance(
            metadata_path, str
        ), "[ERROR] Please pass metadata_path as string."

        if os.path.isfile(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
                self.features = self.metadata["features"]
                print(self.__str__())

        else:
            raise ValueError("[ERROR] Metadata file not found.")

        if isinstance(download_url, str):
            my_ts = [download_url]
        elif isinstance(download_url, list):
            my_ts = download_url
        else:
            raise ValueError(
                "[ERROR] Please pass ``download_url`` as string or list of strings."
            )

        self.tile_server = my_ts

        crs_string = CRS(self.metadata["crs"]["properties"]["name"])
        self.crs = crs_string.to_string()

    def __str__(self) -> str:
        info = f"[INFO] Metadata file has {self.__len__()} item(s)."
        return info

    def __len__(self) -> int:
        return len(self.features)

    def get_polygons(self) -> None:
        """
        For each map in metadata, creates a polygon from map geometry and saves to ``features`` dictionary.
        """
        for feature in self.features:
            polygon = shape(feature["geometry"])
            map_name = feature["properties"]["IMAGE"]
            if len(polygon.geoms) != 1:
                f"[WARNING] Multiple geometries found in map {map_name}. Using first instance."
            feature["polygon"] = polygon.geoms[0]

        self.polygons = True

    def get_grid_bb(self, zoom_level: int | None = 14) -> None:
        """
        For each map in metadata, creates a grid bounding box from map polygons and saves to ``features`` dictionary.

        Parameters
        ----------
        zoom_level : int, optional
            The zoom level to use when creating the grid bounding box.
            Later used when downloading maps, by default 14.
        """
        if not self.polygons:
            self.get_polygons()

        if self.crs != "EPSG:4326":
            raise NotImplementedError(
                "[ERROR] At the moment, MapReader can only create grid bounding boxes and download map sheets using coordinates in WGS1984 (aka EPSG:4326)."
            )

        for feature in self.features:
            polygon = feature["polygon"]
            grid_bb = get_grid_bb_from_polygon(polygon, zoom_level)

            feature["grid_bb"] = grid_bb

        self.grid_bbs = True

    def extract_wfs_id_nos(self) -> None:
        """
        For each map in metadata, extracts WFS ID numbers from WFS information and saves to ``features`` dictionary.
        """
        for feature in self.features:
            wfs_id = feature["id"]
            wfs_id_no = wfs_id.split(sep=".")[-1]

            feature["wfs_id_no"] = eval(wfs_id_no)

        self.wfs_id_nos = True

    def extract_published_dates(
        self,
        date_col: str | list | None = None,
    ) -> None:
        """
        For each map in metadata, extracts publication date and saves to ``features`` dictionary.

        Parameters
        ----------
            date_col : str or list, optional
                A key or list of keys which map to the metadata field containing the publication date.
                Multilayer keys should be passed as a list. e.g.:

                - "key1" will extract ``self.features[i]["key1"]``
                - ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``

                If  None, ["properties"]["WFS_TITLE"] will be used as keys. Date will then be extracted by regex searching for "Published: XXX".
                By default None.
        """
        for feature in self.features:
            map_name = feature["properties"]["IMAGE"]

            if date_col:
                if isinstance(date_col, str):
                    date_col = [date_col]
                if not isinstance(date_col, list):
                    raise ValueError(
                        "[ERROR] Please pass ``date_col`` as string or list of strings."
                    )

                try:
                    published_date = reduce(
                        lambda d, key: d[key], date_col, feature
                    )  # reduce(function, sequence to go through, initial)
                except KeyError as err:
                    raise KeyError(
                        f"[ERROR] {date_col} not found in features dictionary."
                    ) from err

                if published_date == "":  # missing date is fine
                    print(f"[WARNING] No published date detected in {map_name}.")
                    feature["properties"]["published_date"] = []

                else:
                    try:
                        feature["properties"]["published_date"] = int(published_date)
                    except (
                        ValueError,
                        TypeError,
                    ) as err:  # suggests there is something wrong with the choice of date_col
                        raise ValueError(
                            f"[ERROR] Published date is not an integer for map {map_name}. Check your ``date_col`` is correct."
                        ) from err

            else:
                wfs_title = feature["properties"]["WFS_TITLE"]
                published_date = re.findall(
                    r"Published.*[\D]([\d]+)", wfs_title, flags=re.IGNORECASE
                )

                if len(published_date) > 0:  # if date is found
                    if len(published_date) > 1:
                        print(
                            f"[WARNING] Multiple published dates detected in map {map_name}. Using first date."
                        )

                    feature["properties"]["published_date"] = int(published_date[0])

                else:
                    print(f"[WARNING] No published date detected in map {map_name}.")
                    feature["properties"]["published_date"] = []

        self.published_dates = True

    def get_merged_polygon(self) -> None:
        """
        Creates a multipolygon representing all maps in metadata.
        """

        if not self.polygons:
            self.get_polygons()

        polygon_list = [feature["polygon"] for feature in self.features]

        merged_polygon = unary_union(polygon_list)
        self.merged_polygon = merged_polygon

    def get_minmax_latlon(self) -> None:
        """
        Prints minimum and maximum latitudes and longitudes of all maps in metadata.
        """
        if self.merged_polygon is None:
            self.get_merged_polygon()

        min_x, min_y, max_x, max_y = self.merged_polygon.bounds
        print(
            f"[INFO] Min lat: {min_y}, max lat: {max_y} \n\
[INFO] Min lon: {min_x}, max lon: {max_x}"
        )

    ## queries
    def query_map_sheets_by_wfs_ids(
        self,
        wfs_ids: list | int,
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find map sheets by WFS ID numbers.

        Parameters
        ----------
        wfs_ids : Union[list, int]
            The WFS ID numbers of the maps to download.
        append : bool, optional
            Whether to append to current query results list or, if False, start a new list.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False
        """
        if not self.wfs_id_nos:
            self.extract_wfs_id_nos()

        if isinstance(wfs_ids, list):
            requested_maps = wfs_ids
        elif isinstance(wfs_ids, int):
            requested_maps = [wfs_ids]
        else:
            raise ValueError("[ERROR] Please pass ``wfs_ids`` as int or list of ints.")

        if not append:
            self.found_queries = []  # reset each time

        for feature in self.features:
            wfs_id_no = feature["wfs_id_no"]

            if wfs_id_no in requested_maps:
                if feature not in self.found_queries:  # only append if new item
                    self.found_queries.append(feature)

        if print:
            self.print_found_queries()

    def query_map_sheets_by_polygon(
        self,
        polygon: Polygon,
        mode: str | None = "within",
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find map sheets which are found within or intersecting with a defined polygon.

        Parameters
        ----------
        polygon : Polygon
            shapely Polygon
        mode : str, optional
            The mode to use when finding maps.
            Options of ``"within"``, which returns all map sheets which are completely within the defined polygon,
            and ``"intersects""``, which returns all map sheets which intersect/overlap with the defined polygon.
            By default "within".
        append : bool, optional
            Whether to append to current query results list or, if False, start a new list.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False

        Notes
        -----
        Use ``create_polygon_from_latlons()`` to create polygon.
        """
        if not isinstance(polygon, Polygon):
            raise ValueError(
                "[ERROR] Please pass polygon as shapely.geometry.Polygon object.\n\
[HINT] Use ``create_polygon_from_latlons()`` to create polygon."
            )

        if mode not in ["within", "intersects"]:
            raise NotImplementedError(
                '[ERROR] Please use ``mode="within"`` or ``mode="intersects"``.'
            )

        if not self.polygons:
            self.get_polygons()

        if not append:
            self.found_queries = []  # reset each time

        for feature in self.features:
            map_polygon = feature["polygon"]

            if mode == "within":
                if map_polygon.within(polygon):
                    if map_polygon not in self.found_queries:  # only append if new item
                        self.found_queries.append(feature)
            elif mode == "intersects":
                if map_polygon.intersects(polygon):
                    if feature not in self.found_queries:  # only append if new item
                        self.found_queries.append(feature)

        if print:
            self.print_found_queries()

    def query_map_sheets_by_coordinates(
        self,
        coords: tuple,
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find maps sheets which contain a defined set of coordinates.
        Coordinates are (x,y).

        Parameters
        ----------
        coords : tuple
            Coordinates in ``(x,y)`` format.
        append : bool, optional
            Whether to append to current query results list or, if False, start a new list.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False
        """
        if not isinstance(coords, tuple):
            raise ValueError("[ERROR] Please pass coords as a tuple in the form (x,y).")

        coords = Point(coords)

        if not self.polygons:
            self.get_polygons()

        if not append:
            self.found_queries = []  # reset each time

        for feature in self.features:
            map_polygon = feature["polygon"]

            if map_polygon.contains(coords):
                if feature not in self.found_queries:  # only append if new item
                    self.found_queries.append(feature)

        if print:
            self.print_found_queries()

    def query_map_sheets_by_line(
        self,
        line: LineString,
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find maps sheets which intersect with a line.

        Parameters
        ----------
        line : LineString
            shapely LineString
        append : bool, optional
            Whether to append to current query results list or, if False, start a new list.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False

        Notes
        -----
        Use ``create_line_from_latlons()`` to create line.
        """

        if not isinstance(line, LineString):
            raise ValueError(
                "[ERROR] Please pass line as shapely.geometry.LineString object.\n\
[HINT] Use ``create_line_from_latlons()`` to create line."
            )

        if not self.polygons:
            self.get_polygons()

        if not append:
            self.found_queries = []  # reset each time

        for feature in self.features:
            map_polygon = feature["polygon"]

            if map_polygon.intersects(line):
                if feature not in self.found_queries:  # only append if new item
                    self.found_queries.append(feature)

        if print:
            self.print_found_queries()

    def query_map_sheets_by_string(
        self,
        string: str,
        keys: str | list | None = None,
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find map sheets by searching for a string in a chosen metadata field.

        Parameters
        ----------
        string : str
            The string to search for.
            Can be raw string and use regular expressions.
        keys : str or list, optional
            A key or list of keys used to get the metadata field to search in.

            Key(s) will be passed to each features dictionary.
            Multilayer keys should be passed as a list. e.g. ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``.

            If ``None``, will search in all metadata fields. By default ``None``.
        append : bool, optional
            Whether to append to current query results list or, if False, start a new list.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False

        Notes
        -----
        ``string`` is case insensitive.
        """
        if not isinstance(string, str):
            raise ValueError("[ERROR] Please pass ``string`` as a string.")

        if keys is None:
            keys = []
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, list):
            raise ValueError("[ERROR] Please pass key(s) as string or list of strings.")

        if not append:
            self.found_queries = []  # reset each time

        for feature in self.features:
            try:
                field_to_search = reduce(
                    lambda d, key: d[key], keys, feature
                )  # reduce(function, sequence to go through, initial)
            except KeyError as err:
                raise KeyError(
                    f"[ERROR] {keys} not found in features dictionary."
                ) from err

            match = bool(re.search(string, str(field_to_search), re.IGNORECASE))

            if match:
                if feature not in self.found_queries:  # only append if new item
                    self.found_queries.append(feature)

        if print:
            self.print_found_queries()

    def print_found_queries(self) -> None:
        """
        Prints query results.
        """
        if not self.polygons:
            self.get_polygons()

        if len(self.found_queries) == 0:
            logger.info("No query results found/saved.")
        else:
            divider = 14 * "="
            print(f"{divider}\nQuery results:\n{divider}")
            for feature in self.found_queries:
                map_url = feature["properties"]["IMAGEURL"]
                map_bounds = feature["polygon"].bounds
                print(f"URL:     \t{map_url}")
                print(f"coordinates (bounds):  \t{map_bounds}")
                print(20 * "-")

    ## download
    def _initialise_downloader(self):
        """
        Initialise TileDownloader object
        """
        self.downloader = TileDownloader(self.tile_server)

    def _initialise_merger(self, path_save: str):
        """
        Initialise TileMerger object.

        Parameters
        ----------
        path_save : str
            Path to save merged items (i.e. whole map sheets)
        """
        self.merger = TileMerger(output_folder=f"{path_save}/")

    def _check_map_sheet_exists(self, feature: dict, metadata_fname) -> bool:
        """
        Checks if a map sheet is already saved.

        Parameters
        ----------
        feature : dict

        Returns
        -------
        bool
            img_path if file exists, False if not.
        """
        path_save = self.merger.output_folder

        try:
            # get image id with same coords in metadata
            existing_metadata_df = pd.read_csv(
                f"{path_save}{metadata_fname}", sep=",", index_col=0
            )
        except FileNotFoundError:
            return False

        polygon = get_polygon_from_grid_bb(feature["grid_bb"])
        if str(polygon.bounds) in existing_metadata_df["coordinates"].values:
            image_id = existing_metadata_df[
                existing_metadata_df["coordinates"] == str(polygon.bounds)
            ].iloc[0]["name"]
        else:
            return False  # coordinates not in metadata means image doesn't exist

        if os.path.exists(f"{path_save}{image_id}"):
            try:
                # check image is valid
                mpimg.imread(f"{path_save}{image_id}")
                return image_id
            except OSError:
                return False
        return False

    def _download_map(
        self,
        feature: dict,
        existing_id: str | bool,
        download_in_parallel: bool = True,
        overwrite: bool = False,
    ) -> str | bool:
        """
        Downloads a single map sheet and saves as png file.

        Parameters
        ----------
        feature : dict
            The feature for which to download the map sheet.
        existing_id : str | bool
            The existing image id if the map sheet already exists.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.

        Returns
        -------
        str or bool
            image path if map was downloaded successfully, False if not.
        """
        self.downloader.download_tiles(
            feature["grid_bb"], download_in_parallel=download_in_parallel
        )

        if existing_id is False:
            map_name = f"map_{feature['properties']['IMAGE']}"
        else:
            map_name = existing_id[:-4]  # remove file extension (assuming .png)

        img_path = self.merger.merge(
            feature["grid_bb"], file_name=map_name, overwrite=overwrite
        )

        if img_path is not False:
            print(f'[INFO] Downloaded "{img_path}"')
        else:
            print(f'[WARNING] Download of "{img_path}" was unsuccessful.')

        shutil.rmtree(DEFAULT_TEMP_FOLDER)
        return img_path

    def _save_metadata(
        self,
        feature: dict,
        out_filepath: str,
        img_path: str,
        metadata_to_save: dict | None = None,
        **kwargs: dict | None,
    ) -> None:
        """
        Creates list of selected metadata items and saves to a csv file.
        If file exists, metadata list is appended.

        Parameters
        ----------
        feature : dict
            The feature for which to extract the metadata from
        out_filepath : str
            The path to save metadata csv.
        img_path : str
            The path to the downloaded map sheet.
        metadata_to_save : dict, optional
            A dictionary containing column names (str) and metadata keys (str or list) to save to metadata csv.
            Multilayer keys should be passed as a list, i.e. ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``.

            e.g. ``{"county": ["properties", "COUNTY"], "id": "id"}``
        **kwargs: dict, optional
            Keyword arguments to pass to the ``extract_published_dates()`` method.

        Returns
        -------
        list
            List of selected metadata (to be saved)

        Notes
        -----
        Default metadata items are: name, url, coordinates, crs, published_date, grid_bb.
        Additional items can be added using ``metadata_to_save``.
        """
        metadata_cols = [
            "name",
            "url",
            "coordinates",
            "crs",
            "published_date",
            "grid_bb",
        ]
        metadata_dict = {col: None for col in metadata_cols}

        # get default metadata
        metadata_dict["name"] = os.path.basename(img_path)
        metadata_dict["url"] = str(feature["properties"]["IMAGEURL"])
        if not self.published_dates:
            date_col = kwargs.get("date_col", None)
            self.extract_published_dates(date_col=date_col)
        metadata_dict["published_date"] = feature["properties"]["published_date"]
        metadata_dict["grid_bb"] = feature["grid_bb"]
        polygon = get_polygon_from_grid_bb(
            metadata_dict["grid_bb"]
        )  # use grid_bb to get coords of actually downloaded tiles
        metadata_dict["coordinates"] = polygon.bounds
        metadata_dict["crs"] = self.crs

        if metadata_to_save:
            for col, metadata_key in metadata_to_save.items():
                if isinstance(metadata_key, str):
                    metadata_key = [metadata_key]
                if not isinstance(metadata_key, list):
                    raise ValueError(
                        "[ERROR] Please pass ``metadata_to_save`` metadata key(s) as a string or list of strings."
                    )

                try:
                    metadatum = reduce(lambda d, key: d[key], metadata_key, feature)
                except KeyError as err:
                    raise KeyError(
                        f"[ERROR] {metadata_key} not found in features dictionary."
                    ) from err

                metadata_dict[col] = metadatum

        new_metadata_df = pd.DataFrame.from_dict(metadata_dict, orient="index").T

        if os.path.exists(out_filepath):
            existing_metadata_df = pd.read_csv(out_filepath, sep=",", index_col=0)
            metadata_df = pd.concat(
                [existing_metadata_df, new_metadata_df], ignore_index=True
            )
            metadata_df = metadata_df.loc[
                metadata_df.astype(str).drop_duplicates(subset=metadata_cols).index
            ]  # https://stackoverflow.com/questions/43855462/pandas-drop-duplicates-method-not-working-on-dataframe-containing-lists
        else:
            metadata_df = new_metadata_df

        metadata_df.to_csv(out_filepath, sep=",")

    def _download_map_sheets(
        self,
        features: list,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ):
        """Download map sheets from a list of features.

        Parameters
        ----------
        features : list
            list of features to download
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the ``_save_metadata()`` method.
        """

        for feature in tqdm(features):
            existing_id = self._check_map_sheet_exists(feature, metadata_fname)
            if (
                not overwrite and existing_id is not False
            ):  # if map already exists and overwrite is False then skip
                print(f'[INFO] "{existing_id}" already exists. Skipping download.')
                continue
            img_path = self._download_map(
                feature,
                existing_id,
                download_in_parallel=download_in_parallel,
                overwrite=overwrite,
            )
            if img_path is not False:
                metadata_path = f"{path_save}/{metadata_fname}"
                self._save_metadata(
                    feature=feature,
                    out_filepath=metadata_path,
                    img_path=img_path,
                    **kwargs,
                )

    def download_all_map_sheets(
        self,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads all map sheets in metadata.

        Parameters
        ----------
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the ``_download_map_sheets()`` method.
        """
        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        features = self.features
        self._download_map_sheets(
            features,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

    def download_map_sheets_by_wfs_ids(
        self,
        wfs_ids: list | int,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads map sheets by WFS ID numbers.

        Parameters
        ----------
        wfs_ids : Union[list, int]
            The WFS ID numbers of the maps to download.
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the ``_download_map_sheets()`` method.
        """

        if not self.wfs_id_nos:
            self.extract_wfs_id_nos()

        if isinstance(wfs_ids, list):
            requested_maps = wfs_ids
        elif isinstance(wfs_ids, int):
            requested_maps = [wfs_ids]
        else:
            raise ValueError("[ERROR] Please pass ``wfs_ids`` as int or list of ints.")

        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        wfs_id_list = [feature["wfs_id_no"] for feature in self.features]
        if set(wfs_id_list).isdisjoint(set(requested_maps)):
            raise ValueError("[ERROR] No map sheets with given WFS ID numbers found.")

        features = []
        for feature in self.features:
            wfs_id_no = feature["wfs_id_no"]
            if wfs_id_no in requested_maps:
                features.append(feature)

        self._download_map_sheets(
            features,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

    def download_map_sheets_by_polygon(
        self,
        polygon: Polygon,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        mode: str | None = "within",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads any map sheets which are found within or intersecting with a defined polygon.

        Parameters
        ----------
        polygon : Polygon
            shapely Polygon
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        mode : str, optional
            The mode to use when finding maps.
            Options of ``"within"``, which returns all map sheets which are completely within the defined polygon,
            and ``"intersects""``, which returns all map sheets which intersect/overlap with the defined polygon.
            By default "within".
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the ``_download_map_sheets()`` method.

        Notes
        -----
        Use ``create_polygon_from_latlons()`` to create polygon.
        """
        if not isinstance(polygon, Polygon):
            raise ValueError(
                "[ERROR] Please pass polygon as shapely.geometry.Polygon object.\n\
[HINT] Use ``create_polygon_from_latlons()`` to create polygon."
            )

        if mode not in [
            "within",
            "intersects",
        ]:
            raise NotImplementedError(
                '[ERROR] Please use ``mode="within"`` or ``mode="intersects"``.'
            )

        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        if self.merged_polygon is None:
            self.get_merged_polygon()

        if self.merged_polygon.disjoint(polygon):
            raise ValueError("[ERROR] Polygon is out of map metadata bounds.")

        features = []
        for feature in self.features:
            map_polygon = feature["polygon"]

            if mode == "within":
                if map_polygon.within(polygon):
                    features.append(feature)
            elif mode == "intersects":
                if map_polygon.intersects(polygon):
                    features.append(feature)

        self._download_map_sheets(
            features,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

    def download_map_sheets_by_coordinates(
        self,
        coords: tuple,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads any maps sheets which contain a defined set of coordinates.
        Coordinates are (x,y).

        Parameters
        ----------
        coords : tuple
            Coordinates in ``(x,y)`` format.
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the ``_download_map_sheets()`` method.
        """

        if not isinstance(coords, tuple):
            raise ValueError("[ERROR] Please pass coords as a tuple in the form (x,y).")

        coords = Point(coords)

        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        if self.merged_polygon is None:
            self.get_merged_polygon()

        if self.merged_polygon.disjoint(coords):
            raise ValueError("[ERROR] Coordinates are out of map metadata bounds.")

        features = []
        for feature in self.features:
            map_polygon = feature["polygon"]
            if map_polygon.contains(coords):
                features.append(feature)

        self._download_map_sheets(
            features,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

    def download_map_sheets_by_line(
        self,
        line: LineString,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads any maps sheets which intersect with a line.

        Parameters
        ----------
        line : LineString
            shapely LineString
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the ``_download_map_sheets()`` method.

        Notes
        -----
        Use ``create_line_from_latlons()`` to create line.
        """

        if not isinstance(line, LineString):
            raise ValueError(
                "[ERROR] Please pass line as shapely.geometry.LineString object.\n\
[HINT] Use ``create_line_from_latlons()`` to create line."
            )

        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        if self.merged_polygon is None:
            self.get_merged_polygon()

        if self.merged_polygon.disjoint(line):
            raise ValueError("[ERROR] Line is out of map metadata bounds.")

        features = []
        for feature in self.features:
            map_polygon = feature["polygon"]

            if map_polygon.intersects(line):
                features.append(feature)

        self._download_map_sheets(
            features,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

    def download_map_sheets_by_string(
        self,
        string: str,
        keys: str | list = None,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Download map sheets by searching for a string in a chosen metadata field.

        Parameters
        ----------
        string : str
            The string to search for.
            Can be raw string and use regular expressions.
        keys : str or list, optional
            A key or list of keys used to get the metadata field to search in.

            Key(s) will be passed to each features dictionary.
            Multilayer keys should be passed as a list. e.g. ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``.

            If ``None``, will search in all metadata fields. By default ``None``.
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the ``_download_map_sheets()`` method.

        Notes
        -----
        ``string`` is case insensitive.
        """
        if not isinstance(string, str):
            raise ValueError("[ERROR] Please pass ``string`` as a string.")

        if keys is None:
            keys = []
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, list):
            raise ValueError("[ERROR] Please pass key(s) as string or list of strings.")

        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        features = []
        for feature in self.features:
            try:
                field_to_search = reduce(
                    lambda d, key: d[key], keys, feature
                )  # reduce(function, sequence to go through, initial)
            except KeyError as err:
                raise KeyError(
                    f"[ERROR] {keys} not found in features dictionary."
                ) from err

            match = bool(re.search(string, str(field_to_search), re.IGNORECASE))

            if match:
                features.append(feature)

        self._download_map_sheets(
            features,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

    def download_map_sheets_by_queries(
        self,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads map sheets saved as query results.

        Parameters
        ----------
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the ``_download_map_sheets()`` method.
        """
        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        if len(self.found_queries) == 0:
            raise ValueError("[ERROR] No query results found/saved.")

        features = self.found_queries
        self._download_map_sheets(
            features,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

    def hist_published_dates(self, **kwargs) -> None:
        """
        Plots a histogram of the publication dates of maps in metadata.

        Parameters
        ----------
        **kwargs : dict, optional
            A dictionary containing keyword arguments to pass to plotting function.
            See matplotlib.pyplot.hist() for acceptable values.

            e.g. ``**dict(fc='c', ec='k')``

        Notes
        -----
        bins and range already set when plotting so are invalid kwargs.
        """
        if not self.published_dates:
            raise ValueError("[ERROR] Please first run ``extract_published_dates()``")

        published_dates = [
            feature["properties"]["published_date"] for feature in self.features
        ]
        min_date = min(published_dates)
        max_date = max(published_dates)
        date_range = max_date - min_date
        print(min_date, max_date, date_range)

        plt.hist(published_dates, bins=date_range, range=(min_date, max_date), **kwargs)
        plt.locator_params(integer=True)
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.xlabel("Published date", size=18)
        plt.ylabel("Counts", size=18)
        plt.show()

    def plot_features_on_map(
        self,
        features: list,
        map_extent: str | (list | tuple) | None = None,
        add_id: bool | None = True,
    ) -> None:
        """
        Plots boundaries of map sheets on a map using ``cartopy`` library, (if available).

        Parameters
        ----------
        map_extent : Union[str, list, tuple, None], optional
            The extent of the underlying map to be plotted.

            If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
            If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
            If None, the map extent will be set automatically.
            By default None.
        add_id : bool, optional
            Whether to add an ID (WFS ID number) to each map sheet, by default True.
        """
        if self.crs != "EPSG:4326":
            print(
                "[WARNING] This method assumes your coordinates are projected using EPSG 4326. The plot may therefore be incorrect."
            )

        if add_id:
            if not self.wfs_id_nos:
                self.extract_wfs_id_nos()

        plt.figure(figsize=[15, 15])

        try:
            import cartopy.crs as ccrs

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", color="black", linewidth=1)

            if isinstance(map_extent, str):
                if map_extent in ["uk", "UK", "United Kingdom"]:
                    extent = [-8.08999993, 1.81388127, 49.8338702, 60.95000002]
                    ax.set_extent(extent)
                else:
                    raise NotImplementedError(
                        "[ERROR] Currently only UK is implemented. \
Try passing coordinates (min_x, max_x, min_y, max_y) instead or leave blank to auto-set map extent."
                    )
            elif isinstance(map_extent, (list, tuple)):
                ax.set_extent(map_extent)
            else:
                pass

            for feature in features:
                coords = np.array(feature["geometry"]["coordinates"][0][0])

                # Plot coordinates
                plt.plot(
                    coords[:, 0],
                    coords[:, 1],
                    c="r",
                    linewidth=0.5,
                    transform=ccrs.Geodetic(),
                )

                if add_id:
                    text_id = feature["wfs_id_no"]
                    plt.text(
                        np.mean(coords[:, 0]) - 0.15,
                        np.mean(coords[:, 1]) - 0.05,
                        f"{text_id}",
                        color="r",
                    )

        except ImportError:
            print(
                "[WARNING] Cartopy is not installed. \
If you would like to install it, please follow instructions at https://scitools.org.uk/cartopy/docs/latest/installing.html"
            )

            ax = plt.axes()

            for feature in features:
                text_id = feature["wfs_id_no"]
                coords = np.array(feature["geometry"]["coordinates"][0][0])

                plt.plot(coords[:, 0], coords[:, 1], c="r", alpha=0.5)

                if add_id:
                    plt.text(
                        np.mean(coords[:, 0]) - 0.15,
                        np.mean(coords[:, 1]) - 0.05,
                        f"{text_id}",
                        color="r",
                    )

        plt.show()

    def plot_all_metadata_on_map(
        self,
        map_extent: str | (list | tuple) | None = None,
        add_id: bool | None = True,
    ) -> None:
        """
        Plots boundaries of all map sheets in metadata on a map using ``cartopy`` library (if available).

        Parameters
        ----------
        map_extent : Union[str, list, tuple, None], optional
            The extent of the underlying map to be plotted.

            If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
            If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
            If None, the map extent will be set automatically.
            By default None.
        add_id : bool, optional
            Whether to add an ID (WFS ID number) to each map sheet, by default True.
        """

        features_to_plot = self.features
        self.plot_features_on_map(features_to_plot, map_extent, add_id)

    def plot_queries_on_map(
        self,
        map_extent: str | (list | tuple) | None = None,
        add_id: bool | None = True,
    ) -> None:
        """
        Plots boundaries of query results on a map using ``cartopy`` library (if available).

        Parameters
        ----------
        map_extent : Union[str, list, tuple, None], optional
            The extent of the underlying map to be plotted.

            If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
            If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
            If None, the map extent will be set automatically.
            By default None.
        add_id : bool, optional
            Whether to add an ID (WFS ID number) to each map sheet, by default True.
        """

        features_to_plot = self.found_queries
        self.plot_features_on_map(features_to_plot, map_extent, add_id)
