from __future__ import annotations

try:
    from geopy.distance import geodesic, great_circle
except ImportError:
    pass

import logging
import os
import random
import re
import warnings
from ast import literal_eval
from glob import glob
from typing import Literal

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import rasterio
from PIL import Image, ImageOps, ImageStat
from pyproj import Transformer
from rasterio.plot import reshape_as_raster
from shapely import wkt
from shapely.geometry import Polygon, box
from tqdm.auto import tqdm

from mapreader.download.data_structures import GridBoundingBox, GridIndex
from mapreader.download.downloader_utils import get_polygon_from_grid_bb

os.environ[
    "USE_PYGEOS"
] = "0"  # see here https://github.com/geopandas/geopandas/issues/2691
import geopandas as geopd  # noqa: E402

# Ignore warnings
warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)


class MapImages:
    """
    Class to manage a collection of image paths and construct image objects.

    Parameters
    ----------
    path_images : str or None, optional
        Path to the directory containing images (accepts wildcards). By
        default, ``False``
    file_ext : str or False, optional
        The file extension of the image files to be loaded, ignored if file types are specified in ``path_images`` (e.g. with ``"./path/to/dir/*png"``).
        By default ``False``.
    tree_level : str, optional
        Level of the image hierarchy to construct. The value can be
        ``"parent"`` (default) and ``"patch"``.
    parent_path : str, optional
        Path to parent images (if applicable), by default ``None``.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the ``_images_constructor``
        method.

    Attributes
    ----------
    path_images : list
        List of paths to the image files.
    images : dict
        A dictionary containing the constructed image data. It has two levels
        of hierarchy, ``"parent"`` and ``"patch"``, depending on the value of
        the ``tree_level`` parameter.
    """

    def __init__(
        self,
        path_images: str | None = None,
        file_ext: str | bool | None = False,
        tree_level: str | None = "parent",
        parent_path: str | None = None,
        **kwargs: dict,
    ):
        """Initializes the MapImages class."""

        if path_images:
            self.path_images = self._resolve_file_path(path_images, file_ext)

        else:
            self.path_images = []

        # Create images variable (MAIN object variable)
        # New methods (e.g., reading/loading) should construct images this way
        self.images = {"parent": {}, "patch": {}}
        self.parents = self.images["parent"]
        self.patches = self.images["patch"]

        for image_path in tqdm(self.path_images):
            self._images_constructor(
                image_path=image_path,
                parent_path=parent_path,
                tree_level=tree_level,
                **kwargs,
            )

    @staticmethod
    def _resolve_file_path(file_path, file_ext=None):
        if file_ext:
            if os.path.isdir(file_path):
                files = glob(os.path.abspath(f"{file_path}/*.{file_ext}"))
            else:  # if not dir
                files = glob(os.path.abspath(file_path))
                files = [file for file in files if file.split(".")[-1] == file_ext]

        else:
            if os.path.isdir(file_path):
                files = glob(os.path.abspath(f"{file_path}/*.*"))
            else:
                files = glob(os.path.abspath(file_path))

        # check for issues
        if len(files) == 0:
            raise ValueError("[ERROR] No files found!")
        test_ext = files[0].split(".")[-1]
        if not all(file.split(".")[-1] == test_ext for file in files):
            raise ValueError(
                "[ERROR] Directory with multiple file types detected - please specify file extension (`patch_file_ext`) or, pass path to specific file types (wildcards accepted)."
            )

        return files

    def __len__(self) -> int:
        return int(len(self.parents) + len(self.patches))

    def __str__(self) -> Literal[""]:
        print(f"#images: {self.__len__()}")

        print(f"\n#parents: {len(self.parents)}")
        for i, img in enumerate(self.parents):
            try:
                print(os.path.relpath(img))
            except ValueError:  # if no rel path (e.g. mounted on different drives)
                print(os.path.abspath(img))
            if i >= 10:
                print("...")
                break

        print(f"\n#patches: {len(self.patches)}")
        for i, img in enumerate(self.patches):
            try:
                print(os.path.relpath(img))
            except ValueError:  # if no rel path (e.g. mounted on different drives)
                print(os.path.abspath(img))
            if i >= 10:
                print("...")
                break
        return ""

    def _images_constructor(
        self,
        image_path: str,
        parent_path: str | None = None,
        tree_level: str | None = "parent",
        **kwargs: dict,
    ) -> None:
        """
        Constructs image data from the given image path and parent path and adds it to the ``MapImages`` instance's ``images`` attribute.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        parent_path : str, optional
            Path to the parent image (if applicable), by default ``None``.
        tree_level : str, optional
            Level of the image hierarchy to construct, either ``"parent"``
            (default) or ``"parent"``.
        **kwargs : dict, optional
            Additional keyword arguments to be included in the constructed
            image data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``tree_level`` is not set to ``"parent"`` or ``"patch"``.

        Notes
        -----
        This method assumes that the ``images`` attribute has been initialized on the MapImages instance as a dictionary with two levels of hierarchy, ``"parent"`` and ``"patch"``. The image data is added to the corresponding level based on the value of ``tree_level``.
        """

        if tree_level not in ["parent", "patch"]:
            raise ValueError(
                f"[ERROR] tree_level can only be set to parent or patch, not: {tree_level}"  # noqa
            )

        if tree_level == "parent":
            if parent_path:
                print(
                    "[WARNING] Ignoring `parent_path` as `tree_level`  is set to 'parent'."
                )
                parent_path = None
            parent_id = None

        abs_image_path, image_id, _ = self._convert_image_path(image_path)

        self._check_image_mode(image_path)

        # if parent_path is defined get absolute parent path and parent id (tree_level = "patch" is implied)
        if parent_path:
            abs_parent_path, parent_id, _ = self._convert_image_path(parent_path)

        # add image, coords (if present), shape and other kwargs to dictionary
        self.images[tree_level][image_id] = {
            "parent_id": parent_id,
            "image_path": abs_image_path,
        }
        if tree_level == "parent":
            try:
                self._add_geo_info_id(image_id, verbose=False)
            except:
                pass

        self._add_shape_id(image_id)
        for k, v in kwargs.items():
            self.images[tree_level][image_id][k] = v

        if parent_id:  # tree_level = 'patch' is implied
            if parent_id not in self.parents.keys():
                self.parents[parent_id] = {
                    "parent_id": None,
                    "image_path": abs_parent_path,
                    "patches": [],
                }

            # add patch to parent
            self._add_patch_to_parent(image_id)

    @staticmethod
    def _check_image_mode(image_path):
        try:
            img = Image.open(image_path)
        except PIL.UnidentifiedImageError:
            raise PIL.UnidentifiedImageError(
                f"[ERROR] {image_path} is not an image file.\n\n\
See https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open for more information."
            )

        if img.mode not in ["1", "L", "LA", "I", "P", "RGB", "RGBA"]:
            raise NotImplementedError(
                f"[ERROR] Image mode '{img.mode}' not currently accepted.\n\n\
Please save your image(s) as one the following image modes: 1, L, LA, I, P, RGB or RGBA.\n\
See https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes for more information."
            )

    @staticmethod
    def _convert_image_path(inp_path: str) -> tuple[str, str, str]:
        """
        Convert an image path into an absolute path and find basename and directory name.

        Parameters
        ----------
        inp_path : str
            Input path to split.

        Returns
        -------
        tuple
            A tuple containing the absolute path, basename and directory name.
        """
        abs_path = os.path.abspath(inp_path)
        path_basename = os.path.basename(abs_path)
        path_dirname = os.path.dirname(abs_path)
        return abs_path, path_basename, path_dirname

    def add_metadata(
        self,
        metadata: str | pd.DataFrame,
        index_col: int | str | None = 0,
        delimiter: str | None = ",",
        columns: list[str] | None = None,
        tree_level: str | None = "parent",
        ignore_mismatch: bool | None = False,
    ) -> None:
        """
        Add metadata information to the images dictionary.

        Parameters
        ----------
        metadata : str or pandas.DataFrame
            Path to a ``csv`` (or similar), ``xls`` or ``xlsx`` file or a pandas DataFrame that contains the metadata information.
        index_col : int or str, optional
            Column to use as the index when reading the file and converting into a panda.DataFrame.
            Accepts column indices or column names.
            By default ``0`` (first column).

            Only used if a file path is provided as the ``metadata`` parameter.
            Ignored if ``columns`` parameter is passed.
        delimiter : str, optional
            Delimiter used in the ``csv`` file, by default ``","``.

            Only used if a ``csv`` file path is provided as
            the ``metadata`` parameter.
        columns : list, optional
            List of columns indices or names to add to MapImages.
            If ``None`` is passed, all columns will be used.
            By default ``None``.
        tree_level : str, optional
            Determines which images dictionary (``"parent"`` or ``"patch"``)
            to add the metadata to, by default ``"parent"``.
        ignore_mismatch : bool, optional
            Whether to error if metadata with mismatching information is passed.
            By default ``False``.

        Raises
        ------
        ValueError
            If metadata is not a pandas DataFrame or a ``csv``, ``xls`` or ``xlsx`` file path.

            If 'name' or 'image_id' is not one of the columns in the metadata.

        Returns
        -------
        None

        Notes
        ------
        Your metadata file must contain an column which contains the image IDs (filenames) of your images.
        This should have a column name of either ``name`` or ``image_id``.

        Existing information in your ``MapImages`` object will be overwritten if there are overlapping column headings in your metadata file/dataframe.
        """

        if isinstance(metadata, pd.DataFrame):
            if columns:
                metadata_df = metadata[columns].copy()
            else:
                metadata_df = metadata.copy()
                columns = list(metadata_df.columns)

        else:  # if not df
            if os.path.isfile(metadata):
                if metadata.endswith(("xls", "xlsx")):
                    if columns:
                        metadata_df = pd.read_excel(
                            metadata,
                            usecols=columns,
                        )
                    else:
                        metadata_df = pd.read_excel(
                            metadata,
                            index_col=index_col,
                        )
                        columns = list(metadata_df.columns)

                elif metadata.endswith("sv"):  # csv, tsv, etc
                    if columns:
                        metadata_df = pd.read_csv(
                            metadata, usecols=columns, delimiter=delimiter
                        )
                    else:
                        metadata_df = pd.read_csv(
                            metadata, index_col=index_col, delimiter=delimiter
                        )
                        columns = list(metadata_df.columns)

            else:
                raise ValueError(
                    "[ERROR] ``metadata`` should either be the path to a ``csv`` (or similar), ``xls`` or ``xlsx`` file or a pandas DataFrame."  # noqa
                )

        # identify image_id column
        # what to do if "name" or "image_id" are index col?
        if metadata_df.index.name in ["name", "image_id"]:
            metadata_df[metadata_df.index.name] = metadata_df.index
            columns = list(metadata_df.columns)

        if "name" in columns:
            image_id_col = "name"
            if "image_id" in columns:
                print(
                    "[WARNING] Both 'name' and 'image_id' columns exist! Using 'name' as index"  # noqa
                )
        elif "image_id" in columns:
            image_id_col = "image_id"
        else:
            raise ValueError(
                "[ERROR] 'name' or 'image_id' should be one of the columns."
            )

        if any(metadata_df.duplicated(subset=image_id_col)):
            print(
                "[WARNING] Duplicates found in metadata. Keeping only first instance of each duplicated value"
            )
            metadata_df.drop_duplicates(subset=image_id_col, inplace=True, keep="first")

        # look for non-intersecting values
        missing_metadata = set(self.images[tree_level].keys()) - set(
            metadata_df[image_id_col]
        )
        extra_metadata = set(metadata_df[image_id_col]) - set(
            self.images[tree_level].keys()
        )

        if not ignore_mismatch:
            if len(missing_metadata) != 0 and len(extra_metadata) != 0:
                raise ValueError(
                    f"[ERROR] Metadata is missing information for: {[*missing_metadata]}. \n\
[ERROR] Metadata contains information about non-existent images: {[*extra_metadata]}"
                )
            elif len(missing_metadata) != 0:
                raise ValueError(
                    f"[ERROR] Metadata is missing information for: {[*missing_metadata]}"
                )
            elif len(extra_metadata) != 0:
                raise ValueError(
                    f"[ERROR] Metadata contains information about non-existent images: {[*extra_metadata]}"
                )

        for key in self.images[tree_level].keys():
            if key in missing_metadata:
                continue
            else:
                data_series = metadata_df[metadata_df[image_id_col] == key].squeeze()
                for column, item in data_series.items():
                    try:
                        self.images[tree_level][key][column] = literal_eval(item)
                    except:
                        self.images[tree_level][key][column] = item

    def show_sample(
        self,
        num_samples: int,
        tree_level: str | None = "patch",
        random_seed: int | None = 65,
        **kwargs: dict,
    ) -> None:
        """
        Display a sample of images from a particular level in the image
        hierarchy.

        Parameters
        ----------
        num_samples : int
            The number of images to display.
        tree_level : str, optional
            The level of the hierarchy to display images from, which can be
            ``"patch"`` or ``"parent"``. By default "patch".
        random_seed : int, optional
            The random seed to use for reproducibility. Default is ``65``.
        **kwargs : dict, optional
            Additional keyword arguments to pass to
            ``matplotlib.pyplot.figure()``.

        Returns
        -------
        matplotlib.Figure
            The figure generated
        """
        # set random seed for reproducibility
        random.seed(random_seed)

        image_ids = list(self.images[tree_level].keys())
        num_samples = min(len(image_ids), num_samples)
        sample_image_ids = random.sample(image_ids, k=num_samples)

        figsize = kwargs.get("figsize", (15, num_samples * 2))
        plt.figure(figsize=figsize)

        for i, image_id in enumerate(sample_image_ids):
            plt.subplot(num_samples // 3 + 1, 3, i + 1)
            img = Image.open(self.images[tree_level][image_id]["image_path"])
            plt.title(image_id, size=8)

            # check if grayscale
            if len(img.getbands()) == 1:
                plt.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()

    def list_parents(self) -> list[str]:
        """Return list of all parents"""
        return list(self.parents.keys())

    def list_patches(self) -> list[str]:
        """Return list of all patches"""
        return list(self.patches.keys())

    def add_shape(self, tree_level: str | None = "parent") -> None:
        """
        Add a shape to each image in the specified level of the image
        hierarchy.

        Parameters
        ----------
        tree_level : str, optional
            The level of the hierarchy to add shapes to, either ``"parent"``
            (default) or ``"patch"``.

        Returns
        -------
        None

        Notes
        -----
        The method runs :meth:`mapreader.load.images.MapImages._add_shape_id`
        for each image present at the ``tree_level`` provided.
        """
        print(f"[INFO] Add shape, tree level: {tree_level}")

        image_ids = list(self.images[tree_level].keys())
        for image_id in image_ids:
            self._add_shape_id(image_id=image_id)

    def add_coords_from_grid_bb(self, verbose: bool = False) -> None:
        logger.info("Adding coordinates, tree level: parent")

        parent_list = self.list_parents()

        for parent_id in parent_list:
            if "grid_bb" not in self.parents[parent_id].keys():
                print(
                    f"[WARNING] No grid bounding box found for {parent_id}. Suggestion: run add_metadata or add_geo_info."  # noqa
                )
                continue

            self._add_coords_from_grid_bb_id(image_id=parent_id, verbose=verbose)

    def add_coord_increments(self, verbose: bool | None = False) -> None:
        """
        Adds coordinate increments to each image at the parent level.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose outputs, by default ``False``.

        Returns
        -------
        None

        Notes
        -----
        The method runs
        :meth:`mapreader.load.images.MapImages._add_coord_increments_id`
        for each image present at the parent level, which calculates
        pixel-wise delta longitude (``dlon``) and delta latitude (``dlat``)
        for the image and adds the data to it.
        """
        logger.info("Add coord-increments, tree level: parent")

        parent_list = self.list_parents()

        for parent_id in parent_list:
            if "coordinates" not in self.parents[parent_id].keys():
                print(
                    f"[WARNING] No coordinates found for {parent_id}. Suggestion: run add_metadata or add_geo_info."  # noqa
                )
                continue

            self._add_coord_increments_id(image_id=parent_id, verbose=verbose)

    def add_patch_coords(self, verbose: bool = False) -> None:
        """Add coordinates to all patches in patches dictionary.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose outputs.
            By default, ``False``
        """
        patch_list = self.list_patches()

        for patch_id in tqdm(patch_list):
            self._add_patch_coords_id(patch_id, verbose)

    def add_patch_polygons(self, verbose: bool = False) -> None:
        """Add polygon to all patches in patches dictionary.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose outputs.
            By default, ``False``
        """
        patch_list = self.list_patches()

        for patch_id in tqdm(patch_list):
            self._add_patch_polygons_id(patch_id, verbose)

    def add_center_coord(
        self, tree_level: str | None = "patch", verbose: bool | None = False
    ) -> None:
        """
        Adds center coordinates to each image at the specified tree level.

        Parameters
        ----------
        tree_level: str, optional
            The tree level where the center coordinates will be added. It can
            be either ``"parent"`` or ``"patch"`` (default).
        verbose: bool, optional
            Whether to print verbose outputs, by default ``False``.

        Returns
        -------
        None

        Notes
        -----
        The method runs
        :meth:`mapreader.load.images.MapImages._add_center_coord_id`
        for each image present at the ``tree_level`` provided, which calculates
        central longitude and latitude (``center_lon`` and ``center_lat``) for
        the image and adds the data to it.
        """
        print(f"[INFO] Add center coordinates, tree level: {tree_level}")

        image_ids = list(self.images[tree_level].keys())

        already_checked_parent_ids = (
            []
        )  # for if tree_level is patch, only print error message once for each parent image
        for image_id in image_ids:
            if tree_level == "parent":
                if "coordinates" not in self.parents[image_id].keys():
                    print(
                        f"[WARNING] 'coordinates' could not be found in {image_id}. Suggestion: run add_metadata or add_geo_info"  # noqa
                    )
                    continue

            if tree_level == "patch":
                parent_id = self.patches[image_id]["parent_id"]

                if "coordinates" not in self.parents[parent_id].keys():
                    if parent_id not in already_checked_parent_ids:
                        print(
                            f"[WARNING] 'coordinates' could not be found in {parent_id} so center coordinates cannot be calculated for it's patches. Suggestion: run add_metadata or add_geo_info."  # noqa
                        )
                        already_checked_parent_ids.append(parent_id)
                    continue

            self._add_center_coord_id(image_id=image_id, verbose=verbose)

    def _add_shape_id(
        self,
        image_id: int | str,
    ) -> None:
        """
        Add shape (image_height, image_width, image_channels) of the image
        with specified ``image_id`` to the metadata.

        Parameters
        ----------
        image_id : int or str
            The ID of the image to add shape metadata to.

        Returns
        -------
        None
            This method does not return anything. It modifies the metadata of
            the ``images`` property in-place.

        Notes
        -----
        The shape of the image is obtained by loading the image from its
        ``image_path`` value and getting its shape.
        """
        tree_level = self._get_tree_level(image_id)

        try:
            img = Image.open(self.images[tree_level][image_id]["image_path"])
            # shape = (hwc)
            height = img.height
            width = img.width
            channels = len(img.getbands())

            self.images[tree_level][image_id]["shape"] = (height, width, channels)
        except OSError:
            raise ValueError(
                f'[ERROR] Problem with "{image_id}". Please either redownload or remove from list of images to load.'
            )

    def _add_coords_from_grid_bb_id(
        self, image_id: int | str, verbose: bool = False
    ) -> None:
        grid_bb = self.parents[image_id]["grid_bb"]

        if isinstance(grid_bb, str):
            cell1, cell2 = re.findall(r"\(.*?\)", grid_bb)

            z1, x1, y1 = literal_eval(cell1)
            z2, x2, y2 = literal_eval(cell2)

            cell1 = GridIndex(x1, y1, z1)
            cell2 = GridIndex(x2, y2, z2)

            grid_bb = GridBoundingBox(cell1, cell2)

        if isinstance(grid_bb, GridBoundingBox):
            polygon = get_polygon_from_grid_bb(grid_bb)
            coordinates = polygon.bounds
            self.parents[image_id]["coordinates"] = coordinates

        else:
            raise ValueError(f"[ERROR] Unexpected grid_bb format for {image_id}.")

    def _add_coord_increments_id(
        self, image_id: int | str, verbose: bool | None = False
    ) -> None:
        """
        Add pixel-wise delta longitude (``dlon``) and delta latitude
        (``dlat``) to the metadata of the image with the specified ``image_id``
        in the parent tree level.

        Parameters
        ----------
        image_id : int or str
            The ID of the image to add coordinate increments metadata to.
        verbose : bool, optional
            Whether to print warning messages when coordinate or shape
            metadata cannot be found. Default is ``False``.

        Returns
        -------
        None
            This method does not return anything. It modifies the metadata of
            the image in-place.

        Notes
        -----
        Coordinate increments (dlon and dlat) are calculated using the
        following formula:

        .. code-block:: python

            dlat = abs(lat_max - lat_min) / image_height
            dlon = abs(lon_max - lon_min) / image_width

        ``lon_min``, ``lat_min``, ``lon_max`` and ``lat_max`` are the coordinate
        bounds of the image, and ``image_height`` and ``image_width`` are the
        height and width of the image in pixels respectively.

        This method assumes that the coordinate and shape metadata of the
        image have already been added to the metadata.

        If the coordinate metadata cannot be found, a warning message will be
        printed if ``verbose=True``.

        If the shape metadata cannot be found, this method will call the
        :meth:`mapreader.load.images.MapImages._add_shape_id` method to add
        it.
        """

        if "coordinates" not in self.parents[image_id].keys():
            self._print_if_verbose(
                f"[WARNING]'coordinates' could not be found in {image_id}. Suggestion: run add_metadata or add_geo_info.",
                verbose,
            )
            return

        if "shape" not in self.parents[image_id].keys():
            self._add_shape_id(image_id)

        image_height, image_width, _ = self.parents[image_id]["shape"]

        # Extract coordinates from image (xmin, ymin, xmax, ymax)
        min_x, min_y, max_x, max_y = self.parents[image_id]["coordinates"]

        # Calculate dlon and dlat
        dlon = abs(max_x - min_x) / image_width
        dlat = abs(max_y - min_y) / image_height

        self.parents[image_id]["dlon"] = dlon
        self.parents[image_id]["dlat"] = dlat

    def _add_patch_coords_id(self, image_id: str, verbose: bool = False) -> None:
        """Get coordinates of a patch

        Parameters
        ----------
        image_id : str
            The ID of the patch
        verbose : bool, optional
            Whether to print verbose outputs.
            By default, ``False``.

        Return
        -------
        None
        """
        parent_id = self.patches[image_id]["parent_id"]

        if "coordinates" not in self.parents[parent_id].keys():
            self._print_if_verbose(
                f"[WARNING] No coordinates found in  {parent_id} (parent of {image_id}). Suggestion: run add_metadata or add_geo_info.",
                verbose,
            )
            return

        else:
            if not all([k in self.parents[parent_id].keys() for k in ["dlat", "dlon"]]):
                self._add_coord_increments_id(parent_id)

            # get min_x and min_y and pixel-wise dlon and dlat for parent image
            parent_min_x, parent_min_y, parent_max_x, parent_max_y = self.parents[
                parent_id
            ]["coordinates"]
            dlon = self.parents[parent_id]["dlon"]
            dlat = self.parents[parent_id]["dlat"]

            pixel_bounds = self.patches[image_id]["pixel_bounds"]

            # get patch coords
            min_x = (pixel_bounds[0] * dlon) + parent_min_x
            min_y = parent_max_y - (pixel_bounds[3] * dlat)
            max_x = (pixel_bounds[2] * dlon) + parent_min_x
            max_y = parent_max_y - (pixel_bounds[1] * dlat)

            self.patches[image_id]["coordinates"] = (min_x, min_y, max_x, max_y)
            self.patches[image_id]["crs"] = self.parents[parent_id]["crs"]

    def _add_patch_polygons_id(self, image_id: str, verbose: bool = False) -> None:
        """Create polygon from a patch and save to patch dictionary.

        Parameters
        ----------
        image_id : str
            The ID of the patch
        verbose : bool, optional
            Whether to print verbose outputs.
            By default, ``False``.

        Return
        -------
        None
        """

        if "coordinates" not in self.patches[image_id].keys():
            self._add_patch_coords_id(image_id, verbose)

        if "coordinates" in self.patches[image_id].keys():
            coords = self.patches[image_id]["coordinates"]
            self.patches[image_id]["polygon"] = box(*coords)

    def _add_center_coord_id(
        self,
        image_id: int | str,
        verbose: bool | None = False,
    ) -> None:
        """
        Calculates and adds center coordinates (longitude as ``center_lon``
        and latitude as ``center_lat``) to a given image_id's dictionary.

        Parameters
        ----------
        image_id : int or str
            The ID of the patch to add center coordinates to.
        verbose : bool, optional
            Whether to print warning messages or not. Defaults to ``False``.

        Returns
        -------
        None
        """

        tree_level = self._get_tree_level(image_id)

        if "coordinates" not in self.images[tree_level][image_id].keys():
            if tree_level == "parent":
                self._print_if_verbose(
                    f"[WARNING] No coordinates found for {image_id}. Suggestion: run add_metadata or add_geo_info.",
                    verbose,
                )
                return

            if tree_level == "patch":
                self._add_patch_coords_id(image_id, verbose)

        if "coordinates" in self.images[tree_level][image_id].keys():
            self._print_if_verbose(
                f"[INFO] Reading 'coordinates' from {image_id}.", verbose
            )

            min_x, min_y, max_x, max_y = self.images[tree_level][image_id][
                "coordinates"
            ]
            self.images[tree_level][image_id]["center_lat"] = np.mean([min_y, max_y])
            self.images[tree_level][image_id]["center_lon"] = np.mean([min_x, max_x])

    def _calc_pixel_height_width(
        self,
        parent_id: int | str,
        method: str | None = "great-circle",
        verbose: bool | None = False,
    ) -> tuple[tuple, float, float]:
        """
        Calculate the height and width of each pixel in a given image in meters.

        Parameters
        ----------
        parent_id : int or str
            The ID of the parent image to calculate pixel size.
        method : str, optional
            Method to use for calculating image size in meters.

            Possible values: ``"great-circle"`` (default), ``"gc"``, ``"great_circle"``, ``"geodesic"`` or ``"gd"``.
            ``"great-circle"``, ``"gc"`` and ``"great_circle"`` compute size using the great-circle distance formula,
            while ``"geodesic"`` and ``"gd"`` computes size using the geodesic distance formula.
        verbose : bool, optional
            If ``True``, print additional information during the calculation.
            Default is ``False``.

        Returns
        -------
        tuple
            Tuple containing the size of the image in meters (as a tuple of left, bottom, right and top distances) and the mean pixel height and width in meters.

        Notes
        -----
        This method requires the parent image to have location metadata added
        with either the :meth:`mapreader.load.images.MapImages.add_metadata`
        or :meth:`mapreader.load.images.MapImages.add_geo_info` methods.

        The calculations are performed using the ``geopy.distance.geodesic``
        and ``geopy.distance.great_circle`` methods. Thus, the method requires
        the ``geopy`` package to be installed.
        """

        if "coordinates" not in self.parents[parent_id].keys():
            print(
                f"[WARNING] 'coordinates' could not be found in {parent_id}. Suggestion: run add_metadata or add_geo_info."  # noqa
            )
            return

        if "shape" not in self.parents[parent_id].keys():
            self._add_shape_id(parent_id)

        height, width, _ = self.parents[parent_id]["shape"]
        xmin, ymin, xmax, ymax = self.parents[parent_id]["coordinates"]

        # Calculate the size of image in meters
        if method in ["geodesic", "gd"]:
            bottom = geodesic((ymin, xmin), (ymin, xmax)).meters
            right = geodesic((ymin, xmax), (ymax, xmax)).meters
            top = geodesic((ymax, xmax), (ymax, xmin)).meters
            left = geodesic((ymax, xmin), (ymin, xmin)).meters

        elif method in ["gc", "great-circle", "great_circle"]:
            bottom = great_circle((ymin, xmin), (ymin, xmax)).meters
            right = great_circle((ymin, xmax), (ymax, xmax)).meters
            top = great_circle((ymax, xmax), (ymax, xmin)).meters
            left = great_circle((ymax, xmin), (ymin, xmin)).meters

        else:
            raise NotImplementedError(
                f'[ERROR] Method must be one of "great-circle", "great_circle", "gc", "geodesic" or "gd", not: {method}'
            )

        size_in_m = (left, bottom, right, top)  # anticlockwise order

        mean_pixel_height = np.mean([right / height, left / height])
        mean_pixel_width = np.mean([bottom / width, top / width])

        self._print_if_verbose(
            f"[INFO] Size in meters of left/bottom/right/top: {left:.2f}/{bottom:.2f}/{right:.2f}/{top:.2f}",
            verbose,
        )
        self._print_if_verbose(
            f"Each pixel is ~{mean_pixel_height:.3f} X {mean_pixel_width:.3f} meters (height x width).",
            verbose,
        )  # noqa

        return size_in_m, mean_pixel_height, mean_pixel_width

    def patchify_all(
        self,
        method: str | None = "pixel",
        patch_size: int | None = 100,
        tree_level: str | None = "parent",
        path_save: str | None = None,
        add_to_parents: bool | None = True,
        square_cuts: bool | None = False,
        resize_factor: bool | None = False,
        output_format: str | None = "png",
        rewrite: bool | None = False,
        verbose: bool | None = False,
    ) -> None:
        """
        Patchify all images in the specified ``tree_level`` and (if ``add_to_parents=True``) add the patches to the MapImages instance's ``images`` dictionary.

        Parameters
        ----------
        method : str, optional
            Method used to patchify images, choices between ``"pixel"`` (default)
            and ``"meters"`` or ``"meter"``.
        patch_size : int, optional
            Number of pixels/meters in both x and y to use for slicing, by
            default ``100``.
        tree_level : str, optional
            Tree level, choices between ``"parent"`` or ``"patch``, by default
            ``"parent"``.
        path_save : str, optional
            Directory to save the patches.
            If None, will be set as f"patches_{patch_size}_{method}" (e.g. "patches_100_pixel").
            By default None.
        add_to_parents : bool, optional
            If True, patches will be added to the MapImages instance's
            ``images`` dictionary, by default ``True``.
        square_cuts : bool, optional
            If True, all patches will have the same number of pixels in
            x and y, by default ``False``.
        resize_factor : bool, optional
            If True, resize the images before patchifying, by default ``False``.
        output_format : str, optional
            Format to use when writing image files, by default ``"png"``.
        rewrite : bool, optional
            If True, existing patches will be rewritten, by default ``False``.
        verbose : bool, optional
            If True, progress updates will be printed throughout, by default
            ``False``.

        Returns
        -------
        None
        """

        image_ids = self.images[tree_level].keys()
        original_patch_size = patch_size

        if path_save is None:
            path_save = f"patches_{patch_size}_{method}"

        print(f'[INFO] Saving patches in directory named "{path_save}".')

        for image_id in tqdm(image_ids):
            image_path = self.images[tree_level][image_id]["image_path"]

            try:
                full_path = print(os.path.relpath(image_path))
            except ValueError:  # if no rel path (e.g. mounted on different drives)
                full_path = print(os.path.abspath(image_path))

            self._print_if_verbose(f"[INFO] Patchifying {full_path}", verbose)

            # make sure the dir exists
            self._make_dir(path_save)

            if method in ["meters", "meter"]:
                if "coordinates" not in self.images[tree_level][image_id].keys():
                    raise ValueError(
                        "[ERROR] Please add coordinate information first. Suggestion: Run add_metadata or add_geo_info."  # noqa
                    )

                mean_pixel_height = self._calc_pixel_height_width(image_id)[1]
                patch_size = int(
                    original_patch_size / mean_pixel_height
                )  ## check this is correct - should patch be different size in x and y?

            if square_cuts:
                print(
                    "[WARNING] Square cuts is deprecated as of version 1.1.3 and will soon be removed."
                )

                self._patchify_by_pixel_square(
                    image_id=image_id,
                    patch_size=patch_size,
                    path_save=path_save,
                    add_to_parents=add_to_parents,
                    resize_factor=resize_factor,
                    output_format=output_format,
                    rewrite=rewrite,
                    verbose=verbose,
                )

            else:
                self._patchify_by_pixel(
                    image_id=image_id,
                    patch_size=patch_size,
                    path_save=path_save,
                    add_to_parents=add_to_parents,
                    resize_factor=resize_factor,
                    output_format=output_format,
                    rewrite=rewrite,
                    verbose=verbose,
                )

    def _patchify_by_pixel(
        self,
        image_id: str,
        patch_size: int,
        path_save: str,
        add_to_parents: bool | None = True,
        resize_factor: bool | None = False,
        output_format: str | None = "png",
        rewrite: bool | None = False,
        verbose: bool | None = False,
    ):
        """Patchify one image and (if ``add_to_parents=True``) add the patch to the MapImages instance's ``images`` dictionary.

        Parameters
        ----------
        image_id : str
            The ID of the image to patchify
        patch_size : int
            Number of pixels in both x and y to use for slicing
        path_save : str
            Directory to save the patches.
        add_to_parents : bool, optional
            If True, patches will be added to the MapImages instance's
            ``images`` dictionary, by default ``True``.
        resize_factor : bool, optional
            If True, resize the images before patchifying, by default ``False``.
        output_format : str, optional
            Format to use when writing image files, by default ``"png"``.
        rewrite : bool, optional
            If True, existing patches will be rewritten, by default ``False``.
        verbose : bool, optional
            If True, progress updates will be printed throughout, by default
            ``False``.
        """
        tree_level = self._get_tree_level(image_id)

        parent_path = self.images[tree_level][image_id]["image_path"]
        img = Image.open(parent_path)

        if resize_factor:
            original_height, original_width = img.height, img.width
            img = img.resize(
                (
                    int(original_width / resize_factor),
                    int(original_height / resize_factor),
                )
            )

        height, width = img.height, img.width

        for x in range(0, width, patch_size):
            for y in range(0, height, patch_size):
                max_x = min(x + patch_size, width)
                max_y = min(y + patch_size, height)

                min_x = x
                min_y = y

                patch_id = f"patch-{min_x}-{min_y}-{max_x}-{max_y}-#{image_id}#.{output_format}"
                patch_path = os.path.join(path_save, patch_id)
                patch_path = os.path.abspath(patch_path)

                if os.path.isfile(patch_path) and not rewrite:
                    self._print_if_verbose(
                        f"[INFO] File already exists: {patch_path}.", verbose
                    )

                else:
                    patch = img.crop((min_x, min_y, max_x, max_y))
                    if max_x == width:
                        patch = ImageOps.pad(
                            patch, (patch_size, patch.height), centering=(0, 0)
                        )
                    if max_y == height:
                        patch = ImageOps.pad(
                            patch, (patch.width, patch_size), centering=(0, 0)
                        )

                    # check patch size
                    if patch.height != patch_size or patch.width != patch_size:
                        raise ValueError(
                            f"[ERROR] Patch size is {patch.height}x{patch.width} instead of {patch_size}x{patch_size}."
                        )

                    patch.save(patch_path, output_format)

                if add_to_parents:
                    self._images_constructor(
                        image_path=patch_path,
                        parent_path=parent_path,
                        tree_level="patch",
                        pixel_bounds=(min_x, min_y, max_x, max_y),
                    )
                    self._add_patch_coords_id(patch_id)
                    self._add_patch_polygons_id(patch_id)

    def _patchify_by_pixel_square(
        self,
        image_id: str,
        patch_size: int,
        path_save: str,
        add_to_parents: bool | None = True,
        resize_factor: bool | None = False,
        output_format: str | None = "png",
        rewrite: bool | None = False,
        verbose: bool | None = False,
    ):
        """Patchify one image and (if ``add_to_parents=True``) add the patch to the MapImages instance's ``images`` dictionary.
        Use square cuts for patches at edges.

        Parameters
        ----------
        image_id : str
            The ID of the image to patchify
        patch_size : int
            Number of pixels in both x and y to use for slicing
        path_save : str
            Directory to save the patches.
        add_to_parents : bool, optional
            If True, patches will be added to the MapImages instance's
            ``images`` dictionary, by default ``True``.
        resize_factor : bool, optional
            If True, resize the images before patchifying, by default ``False``.
        output_format : str, optional
            Format to use when writing image files, by default ``"png"``.
        rewrite : bool, optional
            If True, existing patches will be rewritten, by default ``False``.
        verbose : bool, optional
            If True, progress updates will be printed throughout, by default
            ``False``.
        """
        tree_level = self._get_tree_level(image_id)

        parent_path = self.images[tree_level][image_id]["image_path"]
        img = Image.open(parent_path)

        if resize_factor:
            original_height, original_width = img.height, img.width
            img = img.resize(
                (
                    int(original_width / resize_factor),
                    int(original_height / resize_factor),
                )
            )

        height, width = img.height, img.width

        for x in range(0, width, patch_size):
            for y in range(0, height, patch_size):
                max_x = min(x + patch_size, width)
                max_y = min(y + patch_size, height)

                # move min_x and min_y back a bit so the patch is square
                min_x = x - (patch_size - (max_x - x))
                min_y = y - (patch_size - (max_y - y))

                patch_id = f"patch-{min_x}-{min_y}-{max_x}-{max_y}-#{image_id}#.{output_format}"
                patch_path = os.path.join(path_save, patch_id)
                patch_path = os.path.abspath(patch_path)

                if os.path.isfile(patch_path) and not rewrite:
                    self._print_if_verbose(
                        f"[INFO] File already exists: {patch_path}.", verbose
                    )

                else:
                    self._print_if_verbose(
                        f'[INFO] Creating "{patch_id}". Number of pixels in x,y: {max_x - min_x},{max_y - min_y}.',
                        verbose,
                    )

                    patch = img.crop((min_x, min_y, max_x, max_y))
                    patch.save(patch_path, output_format)

                if add_to_parents:
                    self._images_constructor(
                        image_path=patch_path,
                        parent_path=parent_path,
                        tree_level="patch",
                        pixel_bounds=(min_x, min_y, max_x, max_y),
                    )
                    self._add_patch_coords_id(patch_id)
                    self._add_patch_polygons_id(patch_id)

    def _add_patch_to_parent(self, patch_id: str) -> None:
        """
        Add patch to parent.

        Parameters
        ----------
        patch_id : str
            The ID of the patch to be added

        Returns
        -------
        None

        Notes
        -----
        This method adds patches to their corresponding parent image.

        It checks if the parent image has any patches, and if not, it creates
        a list of patches and assigns it to the parent. If the parent image
        already has a list of patches, the method checks if the current patch
        is already in the list. If not, the patch is added to the list.
        """
        patch_parent = self.patches[patch_id]["parent_id"]

        if patch_parent is None:
            return

        if patch_parent not in self.parents.keys():
            self.load_parents(parent_ids=patch_parent)

        if "patches" not in self.parents[patch_parent].keys():
            self.parents[patch_parent]["patches"] = [patch_id]
        else:
            if patch_id not in self.parents[patch_parent]["patches"]:
                self.parents[patch_parent]["patches"].append(patch_id)

    def _make_dir(self, path_make: str, exists_ok: bool | None = True) -> None:
        """
        Helper method to make directories.

        ..
            Private method.
        """
        os.makedirs(path_make, exist_ok=exists_ok)

    def calc_pixel_stats(
        self,
        parent_id: str | None = None,
        calc_mean: bool | None = True,
        calc_std: bool | None = True,
        verbose: bool | None = False,
    ) -> None:
        """
        Calculate the mean and standard deviation of pixel values for all
        channels of all patches of
        a given parent image. Store the results in the MapImages instance's
        ``images`` dictionary.

        Parameters
        ----------
        parent_id : str or None, optional
            The ID of the parent image to calculate pixel stats for.
            If ``None``, calculate pixel stats for all parent images.
            By default, None
        calc_mean : bool, optional
            Whether to calculate mean pixel values. By default, ``True``.
        calc_std : bool, optional
            Whether to calculate standard deviation of pixel values.
            By default, ``True``.
        verbose : bool, optional
            Whether to print verbose outputs. By default, ``False``.

        Returns
        -------
        None

        Notes
        -----
        - Pixel stats are calculated for patches of the parent image
          specified by ``parent_id``.
        - If ``parent_id`` is ``None``, pixel stats are calculated for all
          parent images in the object.
        - If mean or standard deviation of pixel values has already been
          calculated for a patch, the calculation is skipped.
        - Pixel stats are stored in the ``images`` attribute of the
          ``MapImages`` instance, under the ``patch`` key for each patch.
        - If no patches are found for a parent image, a warning message is
          displayed and the method moves on to the next parent image.
        """
        # Get correct parent ID
        if parent_id is None:
            parent_ids = self.list_parents()
        else:
            parent_ids = [parent_id]

        for parent_id in tqdm(parent_ids):
            self._print_if_verbose(
                f"\n[INFO] Calculating pixel stats for patches of image: {parent_id}",
                verbose,
            )

            if "patches" not in self.parents[parent_id]:
                print(f"[WARNING] No patches found for: {parent_id}")
                continue

            list_patches = self.parents[parent_id]["patches"]

            for patch in list_patches:
                patch_data = self.patches[patch]
                patch_keys = patch_data.keys()
                img = Image.open(patch_data["image_path"])
                height = img.height
                width = img.width

                # for edge patches, crop the patch image to the correct size first
                min_x, min_y, max_x, max_y = self.patches[patch]["pixel_bounds"]
                if width != max_x - min_x:
                    width = max_x - min_x
                if height != max_y - min_y:
                    height = max_y - min_y
                img = img.crop((0, 0, width, height))

                bands = img.getbands()

                if calc_mean:
                    if "mean_pixel" in patch_keys:
                        calc_mean = False
                if calc_std:
                    if "std_pixel" in patch_keys:
                        calc_std = False

                img_stat = ImageStat.Stat(img)

                if calc_mean:
                    img_mean = img_stat.mean
                    self.patches[patch]["mean_pixel"] = np.mean(img_mean) / 255
                    for i, band in enumerate(bands):
                        # Calculate mean pixel values
                        self.patches[patch][f"mean_pixel_{band}"] = img_mean[i] / 255

                if calc_std:
                    img_std = img_stat.stddev
                    self.patches[patch]["std_pixel"] = np.mean(img_std) / 255
                    for i, band in enumerate(bands):
                        # Calculate std pixel values
                        self.patches[patch][f"std_pixel_{band}"] = img_std[i] / 255

    def convert_images(
        self,
        save: bool | None = False,
        save_format: str | None = "csv",
        delimiter: str | None = ",",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert the ``MapImages`` instance's ``images`` dictionary into pandas
        DataFrames for easy manipulation.

        Parameters
        ----------

        save : bool, optional
            Whether to save the dataframes as files. By default ``False``.
        save_format : str, optional
            If ``save = True``, the file format to use when saving the dataframes.
            Options of csv ("csv") or excel ("excel" or "xlsx").
            By default, "csv".
        delimiter : str, optional
            The delimiter to use when saving the dataframe. By default ``","``.

        Returns
        -------
        tuple of two pandas DataFrames
            The method returns a tuple of two DataFrames: One for the
            ``parent`` images and one for the ``patch`` images.
        """
        parent_df = pd.DataFrame.from_dict(self.parents, orient="index")
        patch_df = pd.DataFrame.from_dict(self.patches, orient="index")

        parent_df.index.set_names("image_id", inplace=True)
        patch_df.index.set_names("image_id", inplace=True)

        if save:
            if save_format == "csv":
                parent_df.to_csv("parent_df.csv", sep=delimiter)
                print('[INFO] Saved parent dataframe as "parent_df.csv"')
                patch_df.to_csv("patch_df.csv", sep=delimiter)
                print('[INFO] Saved patch dataframe as "patch_df.csv"')
            elif save_format in ["excel", "xlsx"]:
                parent_df.to_excel("parent_df.xlsx")
                print('[INFO] Saved parent dataframe as "parent_df.xlsx"')
                patch_df.to_excel("patch_df.xlsx")
                print('[INFO] Saved patch dataframe as "patch_df.xslx"')

            else:
                raise ValueError(
                    f'[ERROR] ``save_format`` should be one of "csv", "excel" or "xlsx". Not {save_format}.'
                )

        return parent_df, patch_df

    def show_parent(
        self,
        parent_id: str,
        column_to_plot: str | None = None,
        **kwargs: dict,
    ) -> None:
        """
        A wrapper method for `.show()` which plots all patches of a
        specified parent (`parent_id`).

        Parameters
        ----------
        parent_id : str
            ID of the parent image to be plotted.
        column_to_plot : str, optional
            Column whose values will be plotted on patches, by default ``None``.
        **kwargs: Dict
            Key words to pass to ``show`` method.
            See help text for ``show`` for more information.

        Returns
        -------
        list
            A list of figures created by the method.

        Notes
        -----
        This is a wrapper method. See the documentation of the
        :meth:`mapreader.load.images.MapImages.show` method for more detail.
        """
        patch_ids = self.parents[parent_id]["patches"]
        figures = self.show(patch_ids, column_to_plot=column_to_plot, **kwargs)

        return figures

    def show(
        self,
        image_ids: str | list[str],
        column_to_plot: str | None = None,
        figsize: tuple | None = (10, 10),
        plot_parent: bool | None = True,
        patch_border: bool | None = True,
        border_color: str | None = "r",
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float | None = 1.0,
        cmap: str | None = "viridis",
        discrete_cmap: int | None = 256,
        plot_histogram: bool | None = False,
        save_kml_dir: bool | str | None = False,
        image_width_resolution: int | None = None,
        kml_dpi_image: int | None = None,
    ) -> None:
        """
        Plot images from a list of `image_ids`.

        Parameters
        ----------
        image_ids : str or list
            Image ID or list of image IDs to be plotted.
        column_to_plot : str, optional
            Column whose values will be plotted on patches, by default ``None``.
        plot_parent : bool, optional
            If ``True``, parent image will be plotted in background, by
            default ``True``.
        figsize : tuple, optional
            The size of the figure to be plotted. By default, ``(10,10)``.
        patch_border : bool, optional
            If ``True``, a border will be placed around each patch, by
            default ``True``.
        border_color : str, optional
            The color of the border. Default is ``"r"``.
        vmin : float, optional
            The minimum value for the colormap.
            If ``None``, will be set to minimum value in ``column_to_plot``, by default ``None``.
        vmax : float, optional
            The maximum value for the colormap.
            If ``None``, will be set to the maximum value in ``column_to_plot``, by default ``None``.
        alpha : float, optional
            Transparency level for plotting ``value`` with floating point
            values ranging from 0.0 (transparent) to 1 (opaque), by default ``1.0``.
        cmap : str, optional
            Color map used to visualize chosen ``column_to_plot`` values, by default ``"viridis"``.
        discrete_cmap : int, optional
            Number of discrete colors to use in color map, by default ``256``.
        plot_histogram : bool, optional
            If ``True``, plot histograms of the ``value`` of images. By default ``False``.
        save_kml_dir : str or bool, optional
            If ``True``, save KML files of the images. If a string is provided,
            it is the path to the directory in which to save the KML files. If
            set to ``False``, no files are saved. By default ``False``.
        image_width_resolution : int or None, optional
            The pixel width to be used for plotting. If ``None``, the
            resolution is not changed. Default is ``None``.

            Note: Only relevant when ``tree_level="parent"``.
        kml_dpi_image : int or None, optional
            The resolution, in dots per inch, to create KML images when
            ``save_kml_dir`` is specified (as either ``True`` or with path).
            By default ``None``.

        Returns
        -------
        list
            A list of figures created by the method.
        """
        # create list, if not already a list
        if isinstance(image_ids, str):
            image_ids = [image_ids]

        if not isinstance(image_ids, list):
            raise ValueError("[ERROR] Please pass image_ids as str or list of strings.")

        if all(self._get_tree_level(image_id) == "parent" for image_id in image_ids):
            tree_level = "parent"
        elif all(self._get_tree_level(image_id) == "patch" for image_id in image_ids):
            tree_level = "patch"
        else:
            raise ValueError("[ERROR] Image IDs must all be at the same tree level")

        figures = []
        if tree_level == "parent":
            for image_id in tqdm(image_ids):
                image_path = self.parents[image_id]["image_path"]
                img = Image.open(image_path)

                # if image_width_resolution is specified, resize the image
                if image_width_resolution:
                    new_width = int(image_width_resolution)
                    rescale_factor = new_width / img.width
                    new_height = int(img.height * rescale_factor)
                    img = img.resize((new_width, new_height), Image.LANCZOS)

                fig = plt.figure(figsize=figsize)
                plt.axis("off")

                # check if grayscale
                if len(img.getbands()) == 1:
                    plt.imshow(img, cmap="gray", vmin=0, vmax=255, zorder=1)
                else:
                    plt.imshow(img, zorder=1)

                if column_to_plot:
                    print(
                        "[WARNING] Values are only plotted on patches. If you'd like to plot values on all patches of a parent image, use ``show_parent`` instead."
                    )

                if save_kml_dir:
                    if "coordinates" not in self.parents[image_id].keys():
                        print(
                            f"[WARNING] 'coordinates' could not be found in {image_id} so no KML file can be created/saved."  # noqa
                        )
                        continue
                    else:
                        os.makedirs(save_kml_dir, exist_ok=True)
                        kml_out_path = os.path.join(save_kml_dir, image_id)

                        plt.savefig(
                            kml_out_path,
                            bbox_inches="tight",
                            pad_inches=0,
                            dpi=kml_dpi_image,
                        )

                        self._create_kml(
                            kml_out_path=kml_out_path,
                            column_to_plot=column_to_plot,
                            coords=self.parents[image_id]["coordinates"],
                            counter=-1,
                        )

                plt.title(image_id)
                figures.append(fig)

            return figures

        elif tree_level == "patch":
            # Collect parents information
            parent_images = {}
            for image_id in image_ids:
                parent_id = self.patches[image_id].get("parent_id", None)

                if parent_id is None:
                    print(f"[WARNING] {image_id} has no parent. Skipping.")
                    continue

                if parent_id not in parent_images.keys():
                    parent_images[parent_id] = {
                        "image_path": self.parents[parent_id]["image_path"],
                        "patches": [image_id],
                    }
                else:
                    parent_images[parent_id]["patches"].append(image_id)

            # plot each parent
            for parent_id in tqdm(parent_images.keys()):
                fig, ax = plt.subplots(figsize=figsize)
                ax.axis("off")

                # initialize values_array - will be filled with values of 'value'
                parent_height, parent_width, _ = self.parents[parent_id]["shape"]
                values_array = np.full((parent_height, parent_width), np.nan)

                for patch_id in self.parents[parent_id]["patches"]:
                    patch_dic = self.patches[patch_id]
                    pixel_bounds = patch_dic[
                        "pixel_bounds"
                    ]  # min_x, min_y, max_x, max_y

                    # Set the values for each patch
                    if column_to_plot:
                        patch_value = patch_dic.get(column_to_plot, None)

                        # assign values to values_array
                        values_array[
                            pixel_bounds[1] : pixel_bounds[3],
                            pixel_bounds[0] : pixel_bounds[2],
                        ] = patch_value

                    if patch_border:
                        patch_xy = (pixel_bounds[0], pixel_bounds[1])
                        patch_width = pixel_bounds[2] - pixel_bounds[0]  # max_x - min_x
                        patch_height = (
                            pixel_bounds[3] - pixel_bounds[1]
                        )  # max_y - min_y
                        rect = patches.Rectangle(
                            patch_xy,
                            patch_width,
                            patch_height,
                            fc="none",
                            ec=border_color,
                            lw=1,
                            zorder=20,
                        )
                        ax.add_patch(rect)

                if column_to_plot:
                    vmin = vmin if vmin else np.min(values_array)
                    vmax = vmax if vmax else np.max(values_array)

                    # set discrete colorbar
                    cmap = plt.get_cmap(cmap, discrete_cmap)

                    values_plot = ax.imshow(
                        values_array,
                        zorder=10,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        alpha=alpha,
                    )

                    fig.colorbar(values_plot, shrink=0.8)

                if plot_parent:
                    parent_path = parent_images[parent_id]["image_path"]
                    parent_image = Image.open(parent_path)

                    # check if grayscale
                    if len(parent_image.getbands()) == 1:
                        ax.imshow(parent_image, cmap="gray", vmin=0, vmax=255)
                    else:
                        ax.imshow(parent_image)

                if save_kml_dir:
                    os.makedirs(save_kml_dir, exist_ok=True)
                    kml_out_path = os.path.join(save_kml_dir, parent_id)
                    plt.savefig(
                        f"{kml_out_path}",
                        bbox_inches="tight",
                        pad_inches=0,
                        dpi=kml_dpi_image,
                    )
                    self._create_kml(
                        kml_out_path=kml_out_path,
                        column_to_plot=column_to_plot,
                        coords=self.parents[image_id]["coordinates"],
                        counter=-1,
                    )

                ax.set_title(parent_id)
                figures.append(fig)

                if column_to_plot and plot_histogram:
                    self._hist_values_array(column_to_plot, values_array)

            return figures

    def _create_kml(
        self,
        kml_out_path: str,
        column_to_plot: str,
        coords: list | tuple,
        counter: int | None = -1,
    ) -> None:
        """Create a KML file.

        ..
            Private method.

        Parameters
        ----------
        path2kml : str
            Directory to save KML file.
        value : _type_
            Column to plot on underlying image.
        coords : list or tuple
            Coordinates of the bounding box.
        counter : int, optional
            Counter to be used for HREF, by default `-1`.
        """

        try:
            import simplekml
        except ImportError:
            raise ImportError("[ERROR] simplekml is needed to create KML outputs.")

        (lon_min, lat_min, lon_max, lat_max) = coords  # (xmin, ymin, xmax, ymax)

        # -----> create KML
        kml = simplekml.Kml()
        ground = kml.newgroundoverlay(name=str(counter))
        if counter == -1:
            ground.icon.href = f"./{column_to_plot}"
        else:
            ground.icon.href = f"./{column_to_plot}_{counter}"

        ground.latlonbox.north = lat_max
        ground.latlonbox.south = lat_min
        ground.latlonbox.east = lon_max
        ground.latlonbox.west = lon_min
        # ground.latlonbox.rotation = -14

        kml.save(f"{kml_out_path}.kml")

    def _hist_values_array(
        self,
        column_to_plot,
        values_array,
    ):
        plt.figure(figsize=(7, 5))
        plt.hist(
            values_array.flatten(),
            color="k",
            bins=20,
        )

        plt.xlabel(column_to_plot, size=20)
        plt.ylabel("Freq.", size=20)
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.grid()
        plt.show()

    def load_patches(
        self,
        patch_paths: str,
        patch_file_ext: str | bool | None = False,
        parent_paths: str | bool | None = False,
        parent_file_ext: str | bool | None = False,
        add_geo_info: bool | None = False,
        clear_images: bool | None = False,
    ) -> None:
        """
        Loads patch images from the given paths and adds them to the ``images``
        dictionary in the ``MapImages`` instance.

        Parameters
        ----------
        patch_paths : str
            The file path of the patches to be loaded.

            *Note: The ``patch_paths`` parameter accepts wildcards.*
        patch_file_ext : str or bool, optional
            The file extension of the patches to be loaded, ignored if file extensions are specified in ``patch_paths`` (e.g. with ``"./path/to/dir/*png"``)
            By default ``False``.
        parent_paths : str or bool, optional
            The file path of the parent images to be loaded. If set to
            ``False``, no parents are loaded. Default is ``False``.

            *Note: The ``parent_paths`` parameter accepts wildcards.*
        parent_file_ext : str or bool, optional
            The file extension of the parent images, ignored if file extensions are specified in ``parent_paths`` (e.g. with ``"./path/to/dir/*png"``)
            By default ``False``.
        add_geo_info : bool, optional
            If ``True``, adds geographic information to the parent image.
            Default is ``False``.
        clear_images : bool, optional
            If ``True``, clears the images from the ``images`` dictionary
            before loading. Default is ``False``.

        Returns
        -------
        None
        """
        patch_files = self._resolve_file_path(patch_paths, patch_file_ext)

        if clear_images:
            self.images = {"parent": {}, "patch": {}}
            self.parents = {}  # are these needed?
            self.patches = {}  # are these needed?

        if parent_paths:
            # Add parents
            self.load_parents(
                parent_paths=parent_paths,
                parent_file_ext=parent_file_ext,
                overwrite=False,
                add_geo_info=add_geo_info,
            )

        for patch_file in tqdm(patch_files):
            if not os.path.isfile(patch_file):
                print(f"[WARNING] File does not exist: {patch_file}")
                continue

            self._check_image_mode(patch_file)

            # patch ID is set to the basename
            patch_id = os.path.basename(patch_file)

            # Parent ID and border can be detected using patch_id
            try:
                parent_id = self.detect_parent_id_from_path(patch_id)
                pixel_bounds = self.detect_pixel_bounds_from_path(patch_id)
            except:
                parent_id = None
                pixel_bounds = None

            # Add patch
            if not self.patches.get(patch_id, False):
                self.patches[patch_id] = {}
            self.patches[patch_id]["parent_id"] = parent_id
            self.patches[patch_id]["image_path"] = patch_file
            self.patches[patch_id]["pixel_bounds"] = pixel_bounds

            # Add patches to the parent
            self._add_patch_to_parent(patch_id)

    @staticmethod
    def detect_parent_id_from_path(
        image_id: int | str, parent_delimiter: str | None = "#"
    ) -> str:
        """
        Detect parent IDs from ``image_id``.

        Parameters
        ----------
        image_id : int or str
            ID of patch.
        parent_delimiter : str, optional
            Delimiter used to separate parent ID when naming patch, by
            default ``"#"``.

        Returns
        -------
        str
            Parent ID.
        """
        return image_id.split(parent_delimiter)[1]

    @staticmethod
    def detect_pixel_bounds_from_path(
        image_id: int | str,
        # border_delimiter="-" # <-- not in use in this method
    ) -> tuple[int, int, int, int]:
        """
        Detects borders from the path assuming patch is named using the
        following format: ``...-min_x-min_y-max_x-max_y-...``

        Parameters
        ----------
        image_id : int or str
            ID of image

        ..
            border_delimiter : str, optional
                Delimiter used to separate border values when naming patch
                image, by default ``"-"``.

        Returns
        -------
        tuple of min_x, min_y, max_x, max_y
            Border (min_x, min_y, max_x, max_y) of image
        """

        split_path = image_id.split("-")
        return (
            int(split_path[1]),
            int(split_path[2]),
            int(split_path[3]),
            int(split_path[4]),
        )

    def load_parents(
        self,
        parent_paths: str | bool | None = False,
        parent_ids: list[str] | (str | bool) | None = False,
        parent_file_ext: str | bool | None = False,
        overwrite: bool | None = False,
        add_geo_info: bool | None = False,
    ) -> None:
        """
        Load parent images from file paths (``parent_paths``).

        If ``parent_paths`` is not given, only ``parent_ids``, no image path
        will be added to the images.

        Parameters
        ----------
        parent_paths : str or bool, optional
            Path to parent images, by default ``False``.
        parent_ids : list, str or bool, optional
            ID(s) of parent images. Ignored if ``parent_paths`` are specified.
            By default ``False``.
        parent_file_ext : str or bool, optional
            The file extension of the parent images, ignored if file extensions are specified in ``parent_paths`` (e.g. with ``"./path/to/dir/*png"``)
            By default ``False``.
        overwrite : bool, optional
            If ``True``, current parents will be overwritten, by default
            ``False``.
        add_geo_info : bool, optional
            If ``True``, geographical info will be added to parents, by
            default ``False``.

        Returns
        -------
        None
        """

        if parent_paths:
            files = self._resolve_file_path(parent_paths, parent_file_ext)

            if overwrite:
                self.parents = {}

            for file in tqdm(files):
                if not os.path.isfile(file):
                    print(f"[WARNING] File does not exist: {file}")
                    continue

                self._check_image_mode(file)

                parent_id = os.path.basename(file)

                if not self.parents.get(parent_id, False):
                    self.parents[parent_id] = {}
                self.parents[parent_id]["parent_id"] = None
                self.parents[parent_id]["image_path"] = (
                    os.path.abspath(file) if os.path.isfile(file) else None
                )
                if add_geo_info:
                    self.add_geo_info()

        elif parent_ids:
            if not isinstance(parent_ids, list):
                parent_ids = [parent_ids]

            for parent_id in parent_ids:
                if not self.parents.get(parent_id, False):
                    self.parents[parent_id] = {}
                self.parents[parent_id]["parent_id"] = None
                self.parents[parent_id]["image_path"] = None

        else:
            raise ValueError(
                "[ERROR] Please pass one of ``parent_paths`` or ``parent_ids``."
            )

    def load_df(
        self,
        parent_df: pd.DataFrame | None = None,
        patch_df: pd.DataFrame | None = None,
        clear_images: bool | None = True,
    ) -> None:
        """
        Create ``MapImages`` instance by loading data from pandas DataFrame(s).

        Parameters
        ----------
        parent_df : pandas.DataFrame, optional
            DataFrame containing parents or path to parents, by default
            ``None``.
        patch_df : pandas.DataFrame, optional
            DataFrame containing patches, by default ``None``.
        clear_images : bool, optional
            If ``True``, clear images before reading the dataframes, by
            default ``True``.

        Returns
        -------
        None
        """

        if clear_images:
            self.images = {"parent": {}, "patch": {}}

        if isinstance(parent_df, pd.DataFrame):
            self.parents.update(parent_df.to_dict(orient="index"))

        if isinstance(patch_df, pd.DataFrame):
            self.patches.update(patch_df.to_dict(orient="index"))

        for patch_id in self.list_patches():
            self._add_patch_to_parent(patch_id)

    def load_csv(
        self,
        parent_path: str | None = None,
        patch_path: str | None = None,
        clear_images: bool | None = False,
        index_col_patch: int | None = 0,
        index_col_parent: int | None = 0,
        delimiter: str | None = ",",
    ) -> None:
        """
        Load CSV files containing information about parent and patches,
        and update the ``images`` attribute of the ``MapImages`` instance with
        the loaded data.

        Parameters
        ----------
        parent_path : str, optional
            Path to the CSV file containing parent image information.
        patch_path : str, optional
            Path to the CSV file containing patch information.
        clear_images : bool, optional
            If True, clear all previously loaded image information before
            loading new information. Default is ``False``.
        index_col_patch : int, optional
            Column to set as index for the patch DataFrame, by default ``0``.
        index_col_parent : int, optional
            Column to set as index for the parent DataFrame, by default ``0``.
        delimiter : str, optional
            The delimiter to use when reading the dataframe. By default ``","``.

        Returns
        -------
        None
        """
        if clear_images:
            self.images = {"parent": {}, "patch": {}}

        if not isinstance(parent_path, str):
            raise ValueError("[ERROR] Please pass ``parent_path`` as string.")
        if not isinstance(patch_path, str):
            raise ValueError("[ERROR] Please pass ``patch_path`` as string.")

        if os.path.isfile(parent_path):
            parent_df = pd.read_csv(
                parent_path, index_col=index_col_parent, sep=delimiter
            )
        else:
            raise ValueError(f"[ERROR] {parent_path} cannot be found.")

        if os.path.isfile(patch_path):
            patch_df = pd.read_csv(patch_path, index_col=index_col_patch, sep=delimiter)
        else:
            raise ValueError(f"[ERROR] {patch_path} cannot be found.")

        self.load_df(parent_df=parent_df, patch_df=patch_df, clear_images=clear_images)

    def add_geo_info(
        self,
        target_crs: str | None = "EPSG:4326",
        verbose: bool | None = True,
    ) -> None:
        """
        Add coordinates (reprojected to EPSG:4326) to all parents images using image metadata.

        Parameters
        ----------
        target_crs : str, optional
            Projection to convert coordinates into, by default ``"EPSG:4326"``.
        verbose : bool, optional
            Whether to print verbose output, by default ``True``

        Returns
        -------
        None

        Notes
        -----
        For each image in the parents dictionary, this method calls ``_add_geo_info_id`` and coordinates (if present) to the image in the ``parent`` dictionary.
        """
        image_ids = list(self.parents.keys())

        for image_id in image_ids:
            self._add_geo_info_id(image_id, target_crs)

    def _add_geo_info_id(
        self,
        image_id: str,
        target_crs: str | None = "EPSG:4326",
        verbose: bool | None = True,
    ) -> None:
        """
        Add coordinates (reprojected to EPSG:4326) to an image.

        Parameters
        ----------
        image_id : str
            The ID of the image to add geographic information to
        target_crs : str, optional
            Projection to convert coordinates into, by default ``"EPSG:4326"``.
        verbose : bool, optional
            Whether to print verbose output, by default ``True``

        Returns
        -------
        None

        Notes
        ------
        This method reads the image files specified in the ``image_path`` key
        of each dictionary in the ``parent`` dictionary.

        It then checks if the image has geographic coordinates in its metadata,
        if not it prints a warning message and skips to the next image.

        If coordinates are present, this method converts them to the specified
        projection ``target_crs``.

        These are then added to the dictionary in the ``parent`` dictionary corresponding to each image.
        """

        image_path = self.parents[image_id]["image_path"]

        # Read the image using rasterio
        tiff_src = rasterio.open(image_path)

        # Check whether coordinates are present
        if isinstance(tiff_src.crs, type(None)):
            self._print_if_verbose(
                f"No coordinates found in {image_id}. Try add_metadata instead.",
                verbose,
            )  # noqa
            return

        else:
            # Get coordinates as string
            tiff_proj = tiff_src.crs.to_string()
            # Coordinate transformation: proj1 ---> proj2
            # tiff is "lat, lon" instead of "x, y"
            transformer = Transformer.from_crs(tiff_proj, target_crs, always_xy=True)
            coords = transformer.transform_bounds(*tiff_src.bounds)
            self.parents[image_id]["coordinates"] = coords
            self.parents[image_id]["crs"] = target_crs

    @staticmethod
    def _print_if_verbose(msg: str, verbose: bool) -> None:
        """
        Print message if verbose is True.
        """
        if verbose:
            print(msg)

    def _get_tree_level(self, image_id: str) -> str:
        """Identify tree level of an image from image_id.

        Parameters
        ----------
        image_id : str
            The ID of the image to identify tree level for.

        Returns
        -------
        str
            The tree level of the image.
        """
        tree_level = "parent" if bool(self.parents.get(image_id)) else "patch"
        return tree_level

    def save_parents_as_geotiffs(
        self,
        rewrite: bool = False,
        verbose: bool = False,
        crs: str | None = None,
    ) -> None:
        """Save all parents in MapImages instance as geotiffs.

        Parameters
        ----------
        rewrite : bool, optional
            Whether to rewrite files if they already exist, by default False
        verbose : bool, optional
            Whether to print verbose outputs, by default False
        crs : str, optional
            The CRS of the coordinates.
            If None, the method will first look for ``crs`` in the parents dictionary and use those. If ``crs`` cannot be found in the dictionary, the method will use "EPSG:4326".
            By default None.
        """

        parents_list = self.list_parents()

        for parent_id in tqdm(parents_list):
            self._save_parent_as_geotiff(parent_id, rewrite, verbose, crs)

    def _save_parent_as_geotiff(
        self,
        parent_id: str,
        rewrite: bool = False,
        verbose: bool = False,
        crs: str | None = None,
    ) -> None:
        """Save a parent image as a geotiff.

        Parameters
        ----------
        parent_id : str
            The ID of the parent to write.
        rewrite : bool, optional
            Whether to rewrite files if they already exist, by default False
        verbose : bool, optional
            Whether to print verbose outputs, by default False
        crs : Optional[str], optional
            The CRS of the coordinates.
            If None, the method will first look for ``crs`` in the parents dictionary and use those. If ``crs`` cannot be found in the dictionary, the method will use "EPSG:4326".
            By default None.

        Raises
        ------
        ValueError
            If parent directory does not exist.
        """

        parent_path = self.parents[parent_id]["image_path"]
        parent_dir = os.path.dirname(parent_path)

        if not os.path.exists(parent_dir):
            raise ValueError(f'[ERROR] Parent directory "{parent_dir}" does not exist.')

        parent_id_no_ext = os.path.splitext(parent_id)[0]
        geotiff_path = f"{parent_dir}/{parent_id_no_ext}.tif"

        self.parents[parent_id]["geotiff_path"] = geotiff_path

        if os.path.isfile(f"{geotiff_path}"):
            if not rewrite:
                self._print_if_verbose(
                    f"[INFO] File already exists: {geotiff_path}.", verbose
                )
                return

        self._print_if_verbose(
            f"[INFO] Creating: {geotiff_path}.",
            verbose,
        )

        if "shape" not in self.parents[parent_id].keys():
            self._add_shape_id(parent_id)
        height, width, channels = self.parents[parent_id]["shape"]

        if "coordinates" not in self.parents[parent_id].keys():
            print(self.parents[parent_id].keys())
            raise ValueError(f"[ERROR] Cannot locate coordinates for {parent_id}")
        coords = self.parents[parent_id]["coordinates"]

        if not crs:
            crs = self.parents[parent_id].get("crs", "EPSG:4326")

        parent_affine = rasterio.transform.from_bounds(*coords, width, height)
        parent = Image.open(parent_path)

        with rasterio.open(
            f"{geotiff_path}",
            "w",
            driver="GTiff",
            height=parent.height,
            width=parent.width,
            count=channels,
            transform=parent_affine,
            dtype="uint8",
            nodata=0,
            crs=crs,
        ) as dst:
            if len(parent.getbands()) == 1:
                parent_array = np.array(parent)
                dst.write(parent_array, indexes=1)
            else:
                parent_array = reshape_as_raster(parent)
                dst.write(parent_array)

    def save_patches_as_geotiffs(
        self,
        rewrite: bool | None = False,
        verbose: bool | None = False,
        crs: str | None = None,
    ) -> None:
        """Save all patches in MapImages instance as geotiffs.

        Parameters
        ----------
        rewrite : bool, optional
            Whether to rewrite files if they already exist, by default False
        verbose : bool, optional
            Whether to print verbose outputs, by default False
        crs : str, optional
            The CRS of the coordinates.
            If None, the method will first look for ``crs`` in the patches dictionary and use those. If ``crs`` cannot be found in the dictionary, the method will use "EPSG:4326".
            By default None.
        """

        patches_list = self.list_patches()

        for patch_id in tqdm(patches_list):
            self._save_patch_as_geotiff(patch_id, rewrite, verbose, crs)

    def _save_patch_as_geotiff(
        self,
        patch_id: str,
        rewrite: bool | None = False,
        verbose: bool | None = False,
        crs: str | None = None,
    ) -> None:
        """Save a patch as a geotiff.

        Parameters
        ----------
        patch_id : str
            The ID of the patch to write.
        rewrite : bool, optional
            Whether to rewrite files if they already exist, by default False
        verbose : bool, optional
            Whether to print verbose outputs, by default False
        crs : Optional[str], optional
            The CRS of the coordinates.
            If None, the method will first look for ``crs`` in the patches dictionary and use those. If ``crs`` cannot be found in the dictionary, the method will use "EPSG:4326".
            By default None.

        Raises
        ------
        ValueError
            If patch directory does not exist.
        """

        patch_path = self.patches[patch_id]["image_path"]
        patch_dir = os.path.dirname(patch_path)
        patch = Image.open(patch_path)

        if not os.path.exists(patch_dir):
            raise ValueError(f'[ERROR] Patch directory "{patch_dir}" does not exist.')

        patch_id_no_ext = os.path.splitext(patch_id)[0]
        geotiff_path = f"{patch_dir}/{patch_id_no_ext}.tif"

        self.patches[patch_id]["geotiff_path"] = geotiff_path

        if os.path.isfile(f"{geotiff_path}"):
            if not rewrite:
                self._print_if_verbose(
                    f"[INFO] File already exists: {geotiff_path}.", verbose
                )
                return

        self._print_if_verbose(
            f"[INFO] Creating: {geotiff_path}.",
            verbose,
        )

        # get shape
        if "shape" not in self.patches[patch_id].keys():
            self._add_shape_id(patch_id)
        height, width, channels = self.patches[patch_id]["shape"]

        # get coords
        if "coordinates" not in self.patches[patch_id].keys():
            self._add_patch_coords_id(patch_id)
        coords = self.patches[patch_id]["coordinates"]

        if not crs:
            crs = self.patches[patch_id].get("crs", "EPSG:4326")

        # for edge patches, crop the patch to the correct size first
        min_x, min_y, max_x, max_y = self.patches[patch_id]["pixel_bounds"]
        if width != max_x - min_x:
            width = max_x - min_x
            patch = patch.crop((0, 0, width, height))
        if height != max_y - min_y:
            height = max_y - min_y
            patch = patch.crop((0, 0, width, height))

        patch_affine = rasterio.transform.from_bounds(*coords, width, height)

        with rasterio.open(
            f"{geotiff_path}",
            "w",
            driver="GTiff",
            height=patch.height,
            width=patch.width,
            count=channels,
            transform=patch_affine,
            dtype="uint8",
            nodata=0,
            crs=crs,
        ) as dst:
            if len(patch.getbands()) == 1:
                patch_array = np.array(patch)
                dst.write(patch_array, indexes=1)
            else:
                patch_array = reshape_as_raster(patch)
                dst.write(patch_array)

    def save_patches_to_geojson(
        self,
        geojson_fname: str | None = "patches.geojson",
        rewrite: bool | None = False,
        crs: str | None = None,
    ) -> None:
        """Saves patches to a geojson file.

        Parameters
        ----------
        geojson_fname : Optional[str], optional
            The name of the geojson file, by default "patches.geojson"
        rewrite : Optional[bool], optional
            Whether to overwrite an existing file, by default False.
        crs : Optional[str], optional
            The CRS to use when writing the geojson.
            If None, the method will look for "crs" in the patches dictionary and, if found, will use that. Otherwise it will set the crs to the default value of "EPSG:4326".
            By default None
        """
        if os.path.isfile(geojson_fname):
            if not rewrite:
                print(
                    f"[WARNING] File already exists: {geojson_fname}. Use ``rewrite=True`` to overwrite."
                )
                return

        _, patch_df = self.convert_images()

        if "polygon" not in patch_df.columns:
            self.add_patch_polygons()
            _, patch_df = self.convert_images()

        patch_df["polygon"] = patch_df["polygon"].apply(
            lambda x: x if isinstance(x, Polygon) else wkt.loads(x)
        )

        if not crs:
            if "crs" in patch_df.columns:
                if len(patch_df["crs"].unique()) == 1:
                    crs = patch_df["crs"].unique()[0]
            else:
                crs = "EPSG:4326"

        if "image_id" in patch_df.columns:
            patch_df.drop(columns=["image_id"], inplace=True)
        patch_df.reset_index(names="image_id", inplace=True)

        # drop pixel stats columns
        patch_df.drop(columns=patch_df.filter(like="pixel", axis=1), inplace=True)
        # change tuple columns to strings
        for col in patch_df.columns:
            if isinstance(patch_df[col][0], tuple):
                patch_df[col] = patch_df[col].apply(str)

        geo_patch_df = geopd.GeoDataFrame(patch_df, geometry="polygon", crs=crs)
        geo_patch_df.to_file(geojson_fname, driver="GeoJSON")

    '''
    def readPatches(self,
                  patch_paths,
                  parent_paths,
                  metadata=None,
                  metadata_fmt="dataframe",
                  metadata_cols2add=[],
                  metadata_index_column="image_id",
                  clear_images=False):
        """read patches from files (patch_paths) and add parents if
           parent_paths is provided
            Arguments:
            patch_paths {str, wildcard accepted} -- path to patches
            parent_paths {False or str, wildcard accepted} -- path to parents
            Keyword Arguments:
            clear_images {bool} -- clear images variable before reading
                    patches (default: {False})
        """
        patch_paths = glob(os.path.abspath(patch_paths))

        if clear_images:
            self.images = {}
            self.parents = {}
            self.patches = {}
            # XXX check
        if not isinstance(metadata, type(None)):
            include_metadata = True
            if metadata_fmt in ["dataframe"]:
                metadata_df = metadata
            elif metadata_fmt.lower() in ["csv"]:
                try:
                    metadata_df = pd.read_csv(metadata)
                except:
                    print(f"[WARNING] could not find metadata file: {metadata}")  # noqa
            else:
                print(f"format cannot be recognized: {metadata_fmt}")
                include_metadata = False
            if include_metadata:
                metadata_df['rd_index_id'] = metadata_df[metadata_index_column].apply(lambda x: os.path.basename(x))
        else:
            include_metadata = False
            for tpath in patch_paths:
            tpath = os.path.abspath(tpath)
            if not os.path.isfile(tpath):
                raise ValueError(f"patch_paths should point to actual files. Current patch_paths: {patch_paths}")
            # patch ID is set to the basename
            patch_id = os.path.basename(tpath)
            # XXXX
            if include_metadata and (not patch_id in list(metadata['rd_index_id'])):
                continue
            # Parent ID and border can be detected using patch_id
            parent_id = self.detect_parent_id_from_path(patch_id)
            min_x, min_y, max_x, max_y = self.detect_border_from_path(patch_id)

            # Add patch
            if not self.patches.get(patch_id, False):
                self.patches[patch_id] = {}
            self.patches[patch_id]["parent_id"] = parent_id
            self.patches[patch_id]["image_path"] = tpath
            self.patches[patch_id]["min_x"] = min_x
            self.patches[patch_id]["min_y"] = min_y
            self.patches[patch_id]["max_x"] = max_x
            self.patches[patch_id]["max_y"] = max_y

        # XXX check
        if include_metadata:
            # metadata_cols = set(metadata_df.columns) - set(['rd_index_id'])
            for one_row in metadata_df.iterrows():
                for one_col in list(metadata_cols2add):
                    self.patches[one_row[1]['rd_index_id']][one_col] = one_row[1][one_col]

        if parent_paths:
            # Add parents
            self.readParents(parent_paths=parent_paths)
            # Add patches to the parent
            self._add_patch_to_parent()

    def process(self, tree_level="parent", update_paths=True,
                save_preproc_dir="./test_preproc"):
        """Process images using process.py module

        Args:
            tree_level (str, optional): "parent" or "patch" paths will be used. Defaults to "parent".
            update_paths (bool, optional): XXX. Defaults to True.
            save_preproc_dir (str, optional): Path to store preprocessed images. Defaults to "./test_preproc".
        """
            from mapreader import process
        # Collect paths and store them self.process_paths
        self.getProcessPaths(tree_level=tree_level)

        saved_paths = process.preprocess_all(self.process_paths,
                                             save_preproc_dir=save_preproc_dir)
        if update_paths:
            self.readParents(saved_paths, update=True)

    def getProcessPaths(self, tree_level="parent"):
        """Create a list of paths to be processed

        Args:
            tree_level (str, optional): "parent" or "patch" paths will be used. Defaults to "parent".
        """
        process_paths = []
        for one_img in self.images[tree_level].keys():
            process_paths.append(self.images[tree_level][one_img]["image_path"])
        self.process_paths = process_paths

    def prepare4inference(self, fmt="dataframe"):
        """Convert images to the specified format (fmt)
            Keyword Arguments:
            fmt {str} -- convert images variable to this format (default: {"dataframe"})
        """
        if fmt in ["pandas", "dataframe"]:
            patches = pd.DataFrame.from_dict(self.patches, orient="index")
            patches.reset_index(inplace=True)
            if len(patches) > 0:
                patches.rename(columns={"image_path": "image_id"}, inplace=True)
                patches.drop(columns=["index", "parent_id"], inplace=True)
                patches["label"] = -1

            parents = pd.DataFrame.from_dict(self.parents, orient="index")
            parents.reset_index(inplace=True)
            if len(parents) > 0:
                parents.rename(columns={"image_path": "image_id"}, inplace=True)
                parents.drop(columns=["index", "parent_id"], inplace=True)
                parents["label"] = -1

            return parents, patches
        else:
            raise ValueError(f"Format {fmt} is not supported!")
    '''
