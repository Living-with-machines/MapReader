from __future__ import annotations

try:
    from geopy.distance import geodesic, great_circle
except ImportError:
    pass

import os
import pathlib
import random
import re
import warnings
from ast import literal_eval
from glob import glob
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import rasterio
import xyzservices as xyz
from PIL import Image, ImageOps, ImageStat
from pyproj import Transformer
from rasterio.plot import reshape_as_raster
from shapely.geometry import box
from tqdm.auto import tqdm

from mapreader.download.data_structures import GridBoundingBox, GridIndex
from mapreader.download.downloader_utils import get_polygon_from_grid_bb
from mapreader.utils.load_frames import (
    get_geodataframe,
    load_from_csv,
    load_from_excel,
    load_from_geojson,
)

os.environ[
    "USE_PYGEOS"
] = "0"  # see here https://github.com/geopandas/geopandas/issues/2691
import geopandas as gpd  # noqa: E402

# Ignore warnings
warnings.filterwarnings("ignore")


class MapImages:
    """
    Class to manage a collection of image paths and construct image objects.

    Parameters
    ----------
    path_images : str or None, optional
        Path to the directory containing images (accepts wildcards).
        By default, ``None`` (no images will be loaded).
    file_ext : str or None, optional
        The file extension of the image files to be loaded, ignored if file types are specified in ``path_images`` (e.g. with ``"./path/to/dir/*png"``).
        By default ``None``.
    tree_level : str, optional
        Level of the image hierarchy to construct. The value can be
        ``"parent"`` (default) and ``"patch"``.
    parent_path : str or None, optional
        Path to parent images (if applicable), by default ``None``.
    **kwargs : dict, optional
        Keyword arguments to pass to the
        :meth:`~.load.images.MapImages._images_constructor` method.

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
        file_ext: str | None = None,
        tree_level: str = "parent",
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
        self.georeferenced = False

        for image_path in tqdm(self.path_images):
            self._images_constructor(
                image_path=image_path,
                parent_path=parent_path,
                tree_level=tree_level,
                **kwargs,
            )

        self.check_georeferencing()

    def check_georeferencing(self):
        if all(
            "coordinates" in self.parents[parent_id].keys()
            for parent_id in self.list_parents()
        ):
            self.add_parent_polygons(tqdm_kwargs={"disable": True})
            self.add_patch_coords(tqdm)
            self.add_patch_polygons(tqdm_kwargs={"disable": True})
            self.georeferenced = True
        else:
            try:
                self.add_coords_from_grid_bb(
                    tqdm_kwargs={"disable": True}
                )  # try to add parent coords using grid_bb
            except ValueError:
                try:
                    self.infer_parent_coords_from_patches(
                        tqdm_kwargs={"disable": True}
                    )  # if not, try to add parent coords using patch coords
                except ValueError:
                    self.georeferenced = False  # give up
                    return
            self.check_georeferencing()  # check again

    @staticmethod
    def _resolve_file_path(file_path: str, file_ext: str | None = None):
        """Resolves file path to list of files.

        Parameters
        ----------
        file_path : str
            Path to the file(s) or directory containing files. Can contain wildcards.
        file_ext : str or None, optional
            The file extension of the images to be loaded, by default ``None``. Ignored if file types are specified in ``file_path`` (e.g. with ``"./path/to/dir/*png"``).

        Returns
        -------
        list
            List of file paths as strings.

        Notes
        -----
        Valid inputs for file path are:

        - Path to a directory containing images (e.g. ``"./path/to/dir"``).
        - Path to a specific image file (e.g. ``"./path/to/image.png"``).
        - Path to multiple image files (e.g. ``"./path/to/dir/*png"``).

        If a directory is provided, the method will search for files with the specified extension (if provided) in the directory. Else it will search for all files in the directory.
        """
        if pathlib.Path(file_path).is_dir():
            if file_ext:
                files = [*pathlib.Path(file_path).glob(f"*.{file_ext}")]
            else:
                files = [*pathlib.Path(file_path).glob("*.*")]
            files = [str(file) for file in files]  # convert to string

        else:
            files = glob(
                file_path
            )  # if not a directory, assume it's a file path or contains wildcards

        if len(files) == 0:
            raise ValueError("[ERROR] No files found!")

        valid_file_exts = r"png$|jpg$|jpeg$|tif$|tiff$"
        if any(re.search(valid_file_exts, file) is None for file in files):
            raise ValueError(
                "[ERROR] Non-image file types detected - please specify a file extension. Supported file types include: png, jpg, jpeg, tif, tiff."
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
        tree_level: str = "parent",
        **kwargs: dict,
    ) -> None:
        """
        Constructs image data from the given image path and parent path and
        adds it to the :class:`~.load.images.MapImages` instance's ``images``
        attribute.

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
        This method assumes that the ``images`` attribute has been initialized
        on the :class:`~.load.images.MapImages` instance as a dictionary with
        two levels of hierarchy, ``"parent"`` and ``"patch"``. The image data
        is added to the corresponding level based on the value of
        ``tree_level``.
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
        Convert an image path into an absolute path and find basename and
        directory name.

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
        metadata: str | pathlib.Path | pd.DataFrame | gpd.GeoDataFrame,
        index_col: int | str | None = 0,
        delimiter: str | None = ",",
        usecols: list[str] | None = None,
        tree_level: str = "parent",
        ignore_mismatch: bool = False,
    ) -> None:
        """
        Add metadata information to the ``images`` dictionary property.

        Parameters
        ----------
        metadata : str or pathlib.Path or pandas.DataFrame or geopandas.GeoDataFrame
            Path to a CSV/TSV/etc, Excel or JSON/GeoJSON file or a pandas DataFrame or geopandas GeoDataFrame.
        index_col : int or str, optional
            Column to use as the index when reading the file and converting into a panda.DataFrame.
            Accepts column indices or column names.
            By default ``0`` (first column).

            Only used if a CSV/TSV file path is provided as the ``metadata`` parameter.
            Ignored if ``usecols`` parameter is passed.
        delimiter : str, optional
            Delimiter used in the ``csv`` file, by default ``","``.

            Only used if a ``csv`` file path is provided as
            the ``metadata`` parameter.
        usecols : list, optional
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
            If metadata is not a valid file path, pandas DataFrame or geopandas GeoDataFrame.

            If 'name' or 'image_id' is not one of the columns in the metadata.

        Returns
        -------
        None

        Notes
        ------
        Your metadata file must contain an column which contains the image IDs
        (filenames) of your images. This should have a column name of either
        ``name`` or ``image_id``.

        Existing information in your :class:`~.load.images.MapImages` object
        will be overwritten if there are overlapping column headings in your
        metadata file/dataframe.
        """

        if isinstance(metadata, pd.DataFrame):
            if usecols:
                metadata_df = metadata[usecols].copy(deep=True)
            else:
                metadata_df = metadata.copy(deep=True)

        elif isinstance(metadata, (str, pathlib.Path)):
            if re.search(r"\.xls.*$", str(metadata)):  # xls or xlsx
                if usecols:
                    metadata_df = load_from_excel(
                        metadata,
                        usecols=usecols,
                    )
                else:
                    metadata_df = load_from_excel(
                        metadata,
                        index_col=index_col,
                    )

            elif re.search(r"\..?sv$", str(metadata)):  # csv, tsv, etc
                print("[INFO] Loading metadata from CSV/TSV/etc file.")
                if usecols:
                    metadata_df = load_from_csv(
                        metadata, usecols=usecols, delimiter=delimiter
                    )
                else:
                    metadata_df = load_from_csv(
                        metadata, index_col=index_col, delimiter=delimiter
                    )

            elif re.search(r"\..*?json$", str(metadata)):  # json or geojson
                print("[INFO] Loading metadata from JSON/GeoJSON file.")
                metadata_df = load_from_geojson(metadata)

            else:
                raise ValueError(
                    "[ERROR] Metadata should be a CSV/TSV/etc, Excel or JSON/GeoJSON file."
                )

        else:  # if not a string, pathlib.Path, pd.DataFrame or gpd.GeoDataFrame
            raise ValueError(
                "[ERROR] ``metadata`` should either be the path to a ``csv`` (or similar), ``xls`` or ``xlsx`` file or a pandas DataFrame or a geopandas GeoDataFrame."  # noqa
            )

        # if index_col is "name" or "image_id", add this as a column too
        if metadata_df.index.name in ["name", "image_id"]:
            metadata_df[metadata_df.index.name] = metadata_df.index

        columns = metadata_df.columns
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

        if "polygon" in metadata_df.columns:
            print("[INFO] Renaming 'polygon' column to 'geometry'")
            metadata_df = metadata_df.rename(columns={"polygon": "geometry"})

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
                    f"""
                    [ERROR] Metadata is missing information for: {[*missing_metadata]}.
                    [ERROR] Metadata contains information about non-existent images: {[*extra_metadata]}
                    """
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

        self.check_georeferencing()

    def show_sample(
        self,
        num_samples: int = 6,
        tree_level: str = "patch",
        random_seed: int = 42,
        **kwargs: dict,
    ) -> None:
        """
        Display a sample of images from a particular level in the image
        hierarchy.

        Parameters
        ----------
        num_samples : int, optional
            The number of images to display. Default is ``6``.
        tree_level : str, optional
            The level of the hierarchy to display images from, which can be
            ``"patch"`` or ``"parent"``. By default "patch".
        random_seed : int, optional
            The random seed to use for reproducibility. Default is ``42``.
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

    def add_shape(
        self, tree_level: str = "parent", tqdm_kwargs: dict | None = None
    ) -> None:
        """
        Add a shape to each image in the specified level of the image
        hierarchy.

        Parameters
        ----------
        tree_level : str, optional
            The level of the hierarchy to add shapes to, either ``"parent"``
            (default) or ``"patch"``.
        tqdm_kwargs : dict, optional
            Additional keyword arguments to pass to the ``tqdm`` progress bar. By default, ``None``.

        Returns
        -------
        None

        Notes
        -----
        The method runs :meth:`~.load.images.MapImages._add_shape_id`
        for each image present at the ``tree_level`` provided.
        """
        print(f"[INFO] Add shape, tree level: {tree_level}")

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        image_ids = list(self.images[tree_level].keys(), **tqdm_kwargs)
        for image_id in image_ids:
            self._add_shape_id(image_id=image_id)

    def add_coords_from_grid_bb(
        self, verbose: bool = False, tqdm_kwargs: dict | None = None
    ) -> None:
        """Add coordinates to parent images using grid bounding boxes.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose outputs. By default, ``False``.
        tqdm_kwargs : dict, optional
            Additional keyword arguments to pass to the ``tqdm`` progress bar. By default, ``None``.

        Raises
        ------
        ValueError
            If no grid bounding box found for a parent image.
        """
        parent_list = self.list_parents()

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        for parent_id in tqdm(parent_list, **tqdm_kwargs):
            if "grid_bb" not in self.parents[parent_id].keys():
                raise ValueError(
                    f"[ERROR] No grid bounding box found for {parent_id}. Suggestion: run `add_metadata` or `add_geo_info`."
                )

            self._add_coords_from_grid_bb_id(image_id=parent_id, verbose=verbose)

    def infer_parent_coords_from_patches(
        self, verbose: bool = False, tqdm_kwargs: dict | None = None
    ) -> None:
        """
        Infers parent coordinates from patches.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose outputs.
            By default, ``False``
        tqdm_kwargs : dict, optional
            Additional keyword arguments to pass to the ``tqdm`` progress bar. By default, ``None``.
        """
        parent_list = self.list_parents()

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        for parent_id in tqdm(parent_list, **tqdm_kwargs):
            if "coordinates" not in self.parents[parent_id].keys():
                self._infer_parent_coords_from_patches_id(parent_id, verbose)

    def add_coord_increments(
        self, verbose: bool = False, tqdm_kwargs: dict | None = None
    ) -> None:
        """
        Adds coordinate increments to each image at the parent level.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose outputs, by default ``False``.
        tqdm_kwargs : dict, optional
            Additional keyword arguments to pass to the ``tqdm`` progress bar. By default, ``None``.

        Returns
        -------
        None

        Notes
        -----
        The method runs
        :meth:`~.load.images.MapImages._add_coord_increments_id`
        for each image present at the parent level, which calculates
        pixel-wise delta longitude (``dlon``) and delta latitude (``dlat``)
        for the image and adds the data to it.
        """
        print("[INFO] Add coord-increments, tree level: parent")

        parent_list = self.list_parents()

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        for parent_id in tqdm(parent_list, **tqdm_kwargs):
            if "coordinates" not in self.parents[parent_id].keys():
                raise ValueError(
                    f"[ERROR] No coordinates found for {parent_id}. Suggestion: run `add_metadata` or `add_geo_info`."  # noqa
                )

            self._add_coord_increments_id(image_id=parent_id, verbose=verbose)

    def add_patch_coords(
        self, verbose: bool = False, tqdm_kwargs: dict | None = None
    ) -> None:
        """Add coordinates to all patches in patches dictionary.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose outputs.
            By default, ``False``
        tqdm_kwargs : dict, optional
            Additional keyword arguments to pass to the ``tqdm`` progress bar. By default, ``None``.
        """
        patch_list = self.list_patches()

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        for patch_id in tqdm(patch_list, **tqdm_kwargs):
            self._add_patch_coords_id(patch_id, verbose)

    def add_patch_polygons(
        self, verbose: bool = False, tqdm_kwargs: dict | None = None
    ) -> None:
        """Add polygon to all patches in patches dictionary.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose outputs.
            By default, ``False``
        tqdm_kwargs : dict, optional
            Additional keyword arguments to pass to the ``tqdm`` progress bar. By default, ``None``.
        """
        patch_list = self.list_patches()

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        for patch_id in tqdm(patch_list, **tqdm_kwargs):
            self._add_patch_polygons_id(patch_id, verbose)

    def add_center_coord(
        self,
        tree_level: str = "patch",
        verbose: bool = False,
        tqdm_kwargs: dict | None = None,
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
        tqdm_kwargs : dict, optional
            Additional keyword arguments to pass to the ``tqdm`` progress bar. By default, ``None``.

        Returns
        -------
        None

        Notes
        -----
        The method runs
        :meth:`~.load.images.MapImages._add_center_coord_id`
        for each image present at the ``tree_level`` provided, which calculates
        central longitude and latitude (``center_lon`` and ``center_lat``) for
        the image and adds the data to it.
        """
        print(f"[INFO] Add center coordinates, tree level: {tree_level}")

        image_ids = list(self.images[tree_level].keys())

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        for image_id in tqdm(image_ids, **tqdm_kwargs):
            if tree_level == "parent":
                if "coordinates" not in self.parents[image_id].keys():
                    raise ValueError(
                        f"[ERROR] 'coordinates' could not be found in {image_id}. Suggestion: run `add_metadata` or `add_geo_info`"  # noqa
                    )

            if tree_level == "patch":
                parent_id = self.patches[image_id]["parent_id"]

                if "coordinates" not in self.parents[parent_id].keys():
                    raise ValueError(
                        f"[ERROR] 'coordinates' could not be found in {parent_id} so center coordinates cannot be calculated for it's patches. Suggestion: run `add_metadata` or `add_geo_info`."  # noqa
                    )

            self._add_center_coord_id(image_id=image_id, verbose=verbose)

    def add_parent_polygons(
        self, verbose: bool = False, tqdm_kwargs: dict | None = None
    ) -> None:
        """Add polygon to all parents in parents dictionary.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose outputs.
            By default, ``False``
        tqdm_kwargs : dict, optional
            Additional keyword arguments to pass to the ``tqdm`` progress bar. By default, ``None``.
        """
        parent_list = self.list_parents()

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        for parent_id in tqdm(parent_list, **tqdm_kwargs):
            self._add_parent_polygons_id(parent_id, verbose)

    def _add_shape_id(
        self,
        image_id: str,
    ) -> None:
        """
        Add shape (image_height, image_width, image_channels) of the image
        with specified ``image_id`` to the metadata.

        Parameters
        ----------
        image_id : str
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

    def _infer_parent_coords_from_patches_id(
        self, image_id: str, verbose: bool = False
    ) -> None:
        if "patches" not in self.parents[image_id].keys():
            raise ValueError(
                f"[ERROR] No patches found for {image_id}. Unable to infer parent coordinates."
            )

        _, patch_df = self.convert_images()
        patch_df = patch_df[patch_df["parent_id"] == image_id]

        for patch_id in patch_df.index:
            if "coordinates" not in self.patches[patch_id].keys():
                self._add_patch_coords_id(patch_id, verbose)

        if isinstance(patch_df, gpd.GeoDataFrame):
            parent_polygon = patch_df.unary_union
            parent_coords = parent_polygon.bounds

            self.parents[image_id]["coordinates"] = parent_coords
            self.parents[image_id]["crs"] = patch_df.crs.to_string()
            self.parents[image_id]["geometry"] = parent_polygon
        else:
            raise ValueError(
                f"[ERROR] No coordinates info found for patches of {image_id}. Unable to infer parent coordinates."
            )

    def _add_coord_increments_id(
        self, image_id: str, verbose: bool | None = False
    ) -> None:
        """
        Add pixel-wise delta longitude (``dlon``) and delta latitude
        (``dlat``) to the metadata of the image with the specified ``image_id``
        in the parent tree level.

        Parameters
        ----------
        image_id : str
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

        ``lon_min``, ``lat_min``, ``lon_max`` and ``lat_max`` are the
        coordinate bounds of the image, and ``image_height`` and
        ``image_width`` are the height and width of the image in pixels
        respectively.

        This method assumes that the coordinate and shape metadata of the image
        have already been added to the metadata.

        If the coordinate metadata cannot be found, a warning message will be
        printed if ``verbose=True``.

        If the shape metadata cannot be found, this method will call the
        :meth:`~.load.images.MapImages._add_shape_id` method to add it.
        """

        if "coordinates" not in self.parents[image_id].keys():
            self._print_if_verbose(
                f"[WARNING] 'coordinates' could not be found in {image_id}. Suggestion: run `add_metadata` or `add_geo_info`.",
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
                f"[WARNING] No coordinates found in  {parent_id} (parent of {image_id}). Suggestion: run `add_metadata` or `add_geo_info`.",
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
            self.patches[image_id]["geometry"] = box(*coords)

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
                    f"[WARNING] No coordinates found for {image_id}. Suggestion: run `add_metadata` or `add_geo_info`.",
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

    def _add_parent_polygons_id(self, image_id: str, verbose: bool = False) -> None:
        """Create polygon from a parent image and save to parent dictionary.

        Parameters
        ----------
        image_id : str
            The ID of the parent image
        verbose : bool, optional
            Whether to print verbose outputs.
            By default, ``False``.

        Return
        -------
        None
        """
        if "coordinates" not in self.parents[image_id].keys():
            raise ValueError(
                "[ERROR] No coordinate information found. Suggestion: run `add_metadata` or `add_geo_info`."
            )

        coords = self.parents[image_id]["coordinates"]
        self.parents[image_id]["geometry"] = box(*coords)

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
        with either the :meth:`~.load.images.MapImages.add_metadata`
        or :meth:`~.load.images.MapImages.add_geo_info` methods.

        The calculations are performed using the ``geopy.distance.geodesic``
        and ``geopy.distance.great_circle`` methods. Thus, the method requires
        the ``geopy`` package to be installed.
        """

        if "coordinates" not in self.parents[parent_id].keys():
            print(
                f"[WARNING] 'coordinates' could not be found in {parent_id}. Suggestion: run `add_metadata` or `add_geo_info`."  # noqa
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
        overlap: int = 0,
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
        overlap : int, optional
            Fractional overlap between patches, by default ``0``.

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
                    overlap=overlap,
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
        overlap: int | None = 0,
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
        overlap : int, optional
            Fractional overlap between patches, by default ``0``.
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

        x = 0
        while x < width:
            y = 0
            while y < height:
                max_x = min(x + patch_size, width)
                max_y = min(y + patch_size, height)

                patch_id = f"patch-{x}-{y}-{max_x}-{max_y}-#{image_id}#.{output_format}"
                patch_path = os.path.join(path_save, patch_id)
                patch_path = os.path.abspath(patch_path)

                if os.path.isfile(patch_path) and not rewrite:
                    self._print_if_verbose(
                        f"[INFO] File already exists: {patch_path}.", verbose
                    )

                else:
                    patch = img.crop((x, y, max_x, max_y))
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
                        pixel_bounds=(x, y, max_x, max_y),
                    )
                    self._add_patch_coords_id(patch_id)
                    self._add_patch_polygons_id(patch_id)

                overlap_pixels = int(patch_size * overlap)
                y = y + patch_size - overlap_pixels
            x = x + patch_size - overlap_pixels

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
        Convert the :class:`~.load.images.MapImages` instance's ``images``
        dictionary into pandas DataFrames (or geopandas GeoDataFrames if geo-referenced) for easy manipulation.

        Parameters
        ----------

        save : bool, optional
            Whether to save the dataframes as files. By default ``False``.
        save_format : str, optional
            If ``save = True``, the file format to use when saving the dataframes.
            Options of csv ("csv"), excel ("excel" or "xlsx") or geojson ("geojson").
            By default, "csv".
        delimiter : str, optional
            The delimiter to use when saving the dataframe. By default ``","``.

        Returns
        -------
        tuple of two pandas DataFrames or geopandas GeoDataFrames
            The method returns a tuple of two DataFrames/GeoDataFrames: One for the
            ``parent`` images and one for the ``patch`` images.
        """
        parent_df = pd.DataFrame.from_dict(self.parents, orient="index")
        patch_df = pd.DataFrame.from_dict(self.patches, orient="index")

        # set index name
        parent_df.index.set_names("image_id", inplace=True)
        patch_df.index.set_names("image_id", inplace=True)

        # convert to GeoDataFrames if coordinates are present
        if len(parent_df):
            parent_df = get_geodataframe(parent_df)
        if len(patch_df):
            patch_df = get_geodataframe(patch_df)

        if save:
            if save_format == "csv":
                if len(parent_df):
                    parent_df.to_csv("parent_df.csv", sep=delimiter)
                    print('[INFO] Saved parent dataframe as "parent_df.csv"')
                if len(patch_df):
                    patch_df.to_csv("patch_df.csv", sep=delimiter)
                    print('[INFO] Saved patch dataframe as "patch_df.csv"')
            elif save_format in ["excel", "xlsx"]:
                if len(parent_df):
                    parent_df.to_excel("parent_df.xlsx")
                    print('[INFO] Saved parent dataframe as "parent_df.xlsx"')
                if len(patch_df):
                    patch_df.to_excel("patch_df.xlsx")
                    print('[INFO] Saved patch dataframe as "patch_df.xslx"')

            # save as geojson (only if georeferenced)
            elif save_format == "geojson":
                if not self.georeferenced:
                    raise ValueError(
                        "[ERROR] Cannot save as GeoJSON as no coordinate information found."
                    )

                if isinstance(parent_df, gpd.GeoDataFrame):
                    parent_df.to_file(
                        "parent_df.geojson", driver="GeoJSON", engine="pyogrio"
                    )
                    print('[INFO] Saved parent dataframe as "parent_df.geojson"')

                if isinstance(patch_df, gpd.GeoDataFrame):
                    patch_df.to_file(
                        "patch_df.geojson", driver="GeoJSON", engine="pyogrio"
                    )
                    print('[INFO] Saved patch dataframe as "patch_df.geojson"')

            else:
                raise ValueError(
                    f'[ERROR] ``save_format`` should be one of "csv", "excel" or "xlsx" or "geojson". Not {save_format}.'
                )

        return parent_df, patch_df

    def explore_patches(
        self,
        parent_id: str,
        column_to_plot: str | None = None,
        xyz_url: str | None = None,
        categorical: bool = False,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        style_kwargs: dict | None = None,
    ):
        """
        Explore patches of a parent image. This method only works with georeferenced images.

        Parameters
        ----------
        parent_id : str
            The ID of the parent image to explore.
        column_to_plot : str | None, optional
            The column values to plot on patches. If None, plot just the patch bounding boxes.
            By default None.
        xyz_url : str | None, optional
            The XYZ URL of the tilelayer to use as a baselayer for the map. If None, will default to OpenStreetMap.Mapnik. By default None.
        categorical : bool, optional
            Whether the column to plot is categorical or not. By default False.
        cmap : str, optional
            The colormap to use when plotting column values. By default "viridis".
        vmin : float | None, optional
            The minimum value of the colormap. If `None`, will use the minimum value in the data. By default `None`.
        vmax : float | None, optional
            The maximum value of the colormap. If `None`, will use the minimum value in the data. By default `None`.
        style_kwargs : dict | None, optional
            A dictionary of style keyword arguments to pass to the folium style_function. By default None.

        Returns
        -------
        folium.Map
            The folium map object with the patches plotted.
        """
        if parent_id not in self.list_parents():
            raise ValueError(f"[ERROR] {parent_id} not found in parent list.")

        if style_kwargs is None:
            style_kwargs = {}

        if self.georeferenced:
            if xyz_url:
                tiles = xyz.TileProvider(name=xyz_url, url=xyz_url, attribution=xyz_url)
            else:
                tiles = xyz.providers.OpenStreetMap.Mapnik

            _, patch_df = self.convert_images()

            if "image_id" in patch_df.columns:
                patch_df.drop(columns=["image_id"], inplace=True)

            if column_to_plot:  # plot column values
                return patch_df[patch_df["parent_id"] == parent_id].explore(
                    column=column_to_plot,
                    tiles=tiles,
                    categorical=categorical,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    style_kwds=style_kwargs,
                )
            else:  # plot patches (i.e. bounding boxes)
                return patch_df[patch_df["parent_id"] == parent_id].explore(
                    tiles=tiles,
                    style_kwds=style_kwargs,
                )

        else:
            raise NotImplementedError(
                "[ERROR] This method only works with georeferenced images. Either add coordinate information or use the `show_patches` method."
            )

    def show_patches(
        self,
        parent_id: str,
        column_to_plot: str | None = None,
        figsize: tuple = (10, 10),
        alpha: float = 0.5,
        categorical: bool = False,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        style_kwargs: dict | None = None,
    ):
        """Plot patches of a parent image using matplotlib. This method works for both georeferenced and non-georeferenced images.

        Parameters
        ----------
        parent_id : str
            The ID of the parent image to plot.
        column_to_plot : str | None, optional
            The column values to plot on patches. If None, plot just the patch bounding boxes.
        figsize : tuple, optional
            The size of the figure to be plotted. By default, (10,10).
        alpha : float, optional
            Transparency level for plotting patches, by default 0.5.
        categorical : bool, optional
            Whether the column to plot is categorical or not. By default False.
        cmap : str, optional
            The colormap to use when plotting column values. By default "viridis".
        vmin : float | None, optional
            The minimum value of the colormap. If `None`, will use the minimum value in the data. By default `None`.
        vmax : float | None, optional
            The maximum value of the colormap. If `None`, will use the minimum value in the data. By default `None`.
        style_kwargs : dict | None, optional
            A dictionary of style keyword arguments to pass to matplotlib.pyplot.plot. By default None.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object with the patches plotted.

        Notes
        -----
        Users with georeferenced images may wish to use the `explore_patches` method instead.
        """
        image_path = self.parents[parent_id]["image_path"]
        img = Image.open(image_path)

        if style_kwargs is None:
            style_kwargs = {}
        style_kwargs.pop("alpha", None)  # remove alpha from style_kwargs, if present

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        # check if grayscale
        if len(img.getbands()) == 1:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, zorder=1)
        else:
            ax.imshow(img, zorder=1)
        ax.set_title(parent_id)

        _, patch_df = self.convert_images()

        if self.georeferenced:
            patch_df = pd.DataFrame(patch_df)  # convert back to pd.DataFrame
            patch_df.drop(columns=["geometry", "crs"], inplace=True)
        patch_df["pixel_geometry"] = patch_df["pixel_bounds"].apply(lambda x: box(*x))
        patch_df = gpd.GeoDataFrame(patch_df, geometry="pixel_geometry")

        if column_to_plot:  # plot column values
            patch_df[patch_df["parent_id"] == parent_id].plot(
                column=column_to_plot,
                ax=ax,
                legend=True,
                categorical=categorical,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                **style_kwargs,
            )
        else:  # plot patches (i.e. bounding boxes)
            patch_df[patch_df["parent_id"] == parent_id].boundary.plot(
                ax=ax, alpha=alpha, **style_kwargs
            )

        fig.show()

    def load_patches(
        self,
        patch_paths: str,
        parent_paths: str | bool | None = False,
        patch_file_ext: str | bool | None = False,
        parent_file_ext: str | bool | None = False,
        add_geo_info: bool | None = False,
        clear_images: bool | None = False,
    ) -> None:
        """
        Loads patch images from the given paths and adds them to the ``images``
        dictionary in the :class:`~.load.images.MapImages` instance.

        Parameters
        ----------
        patch_paths : str
            The file path of the patches to be loaded.

            *Note: The ``patch_paths`` parameter accepts wildcards.*
        parent_paths : str or bool, optional
            The file path of the parent images to be loaded. If set to
            ``False``, no parents are loaded. Default is ``False``.

            *Note: The ``parent_paths`` parameter accepts wildcards.*
        patch_file_ext : str or bool, optional
            The file extension of the patches to be loaded, ignored if file extensions are specified in ``patch_paths`` (e.g. with ``"./path/to/dir/*png"``)
            By default ``False``.
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
            self.parents = {}
            self.patches = {}

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

        self.check_georeferencing()

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

        if add_geo_info:
            self.add_geo_info()

        self.check_georeferencing()

    def load_df(
        self,
        parent_df: pd.DataFrame | gpd.GeoDataFrame | None = None,
        patch_df: pd.DataFrame | gpd.GeoDataFrame | None = None,
        clear_images: bool | None = True,
    ) -> None:
        """
        Create :class:`~.load.images.MapImages` instance by loading data from
        pandas DataFrame(s).

        Parameters
        ----------
        parent_df : pandas.DataFrame or gpd.GeoDataFrame or None, optional
            DataFrame containing parents or path to parents, by default
            ``None``.
        patch_df : pandas.DataFrame or gpd.GeoDataFrame or None, optional
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
            self.parents = {}
            self.patches = {}

        if isinstance(parent_df, pd.DataFrame):
            if "polygon" in parent_df.columns:
                print("[INFO] Renaming 'polygon' to 'geometry' for parent_df.")
                parent_df = parent_df.rename(columns={"polygon": "geometry"})
            self.parents.update(parent_df.to_dict(orient="index"))

        if isinstance(patch_df, pd.DataFrame):
            if "polygon" in patch_df.columns:
                print("[INFO] Renaming 'polygon' to 'geometry' for patch_df.")
                patch_df = patch_df.rename(columns={"polygon": "geometry"})
            self.patches.update(patch_df.to_dict(orient="index"))

        for patch_id in self.list_patches():
            self._add_patch_to_parent(patch_id)

        self.check_georeferencing()

    def load_csv(
        self,
        parent_path: str | pathlib.Path | None = None,
        patch_path: str | pathlib.Path | None = None,
        clear_images: bool = False,
        index_col_patch: int | str | None = 0,
        index_col_parent: int | str | None = 0,
        delimiter: str = ",",
    ) -> None:
        """
        Load CSV files containing information about parent and patches,
        and update the ``images`` attribute of the
        :class:`~.load.images.MapImages` instance with the loaded data.

        Parameters
        ----------
        parent_path : str or pathlib.Path or None
            Path to the CSV file containing parent image information. By default, ``None``.
        patch_path : str or pathlib.Path or None
            Path to the CSV file containing patch information. By default, ``None``.
        clear_images : bool, optional
            If True, clear all previously loaded image information before
            loading new information. Default is ``False``.
        index_col_patch : int or str or None, optional
            Column to set as index for the patch DataFrame, by default ``0``.
        index_col_parent : int or str or None, optional
            Column to set as index for the parent DataFrame, by default ``0``.
        delimiter : str, optional
            The delimiter to use when reading the dataframe. By default ``","``.

        Returns
        -------
        None
        """
        if clear_images:
            self.images = {"parent": {}, "patch": {}}
            self.parents = {}
            self.patches = {}

        if isinstance(parent_path, (str, pathlib.Path)):
            parent_df = load_from_csv(
                parent_path, index_col=index_col_parent, delimiter=delimiter
            )
        else:
            parent_df = None

        if isinstance(patch_path, (str, pathlib.Path)):
            patch_df = load_from_csv(
                patch_path, index_col=index_col_patch, delimiter=delimiter
            )
        else:
            patch_df = None

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
                f"No coordinates found in {image_id}. Try `add_metadata` instead.",
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
        """
        Save all parents in :class:`~.load.images.MapImages` instance as
        geotiffs.

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
        """
        Save all patches in :class:`~.load.images.MapImages` instance as
        geotiffs.

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
        if len(self.patches.keys()) == 0:
            raise ValueError(
                "[ERROR] No patches found. Please patchify your maps first!"
            )

        if os.path.isfile(geojson_fname):
            if not rewrite:
                print(
                    f"[WARNING] File already exists: {geojson_fname}. Use ``rewrite=True`` to overwrite."
                )
                return

        self.check_georeferencing()
        if not self.georeferenced:
            raise ValueError(
                "[ERROR] No geographic information found. Please run `add_geo_info`  or `add_metadata` first."
            )

        _, patch_df = self.convert_images()

        if crs and crs != patch_df.crs:
            print(
                f"[INFO] Reprojecting patches to {crs}. Note: This will not update coordinates column."
            )
            patch_df = patch_df.to_crs(crs)

        if not crs:
            crs = patch_df.crs

        if "image_id" in patch_df.columns:
            patch_df.drop(columns=["image_id"], inplace=True)
        patch_df.reset_index(names="image_id", inplace=True)

        # drop pixel stats columns
        patch_df.drop(columns=patch_df.filter(like="pixel", axis=1), inplace=True)

        # save
        patch_df.to_file(geojson_fname, driver="GeoJSON", engine="pyogrio")
