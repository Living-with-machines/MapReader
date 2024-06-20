from __future__ import annotations

import os
import pathlib
from itertools import combinations

import geopandas as geopd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from shapely import Polygon
from tqdm.auto import tqdm


class Runner:
    def __init__() -> None:
        """Initialise the Runner class."""
        # empty in the base class

    def run_all(
        self,
        patch_df: pd.DataFrame = None,
        return_dataframe: bool = False,
        min_ioa: float = 0.7,
    ) -> dict | pd.DataFrame:
        """Run the model on all images in the patch dataframe.

        Parameters
        ----------
        patch_df : pd.DataFrame, optional
            Dataframe containing patch information, by default None.
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        min_ioa : float, optional
            The minimum intersection over area to consider two polygons the same, by default 0.7

        Returns
        -------
        dict or pd.DataFrame
            A dictionary of predictions for each patch image or a DataFrame if `as_dataframe` is True.
        """
        if patch_df is None:
            if self.patch_df is not None:
                patch_df = self.patch_df
            else:
                raise ValueError("[ERROR] Please provide a `patch_df`")
        img_paths = patch_df["image_path"].to_list()

        patch_predictions = self.run_on_images(
            img_paths, return_dataframe=return_dataframe, min_ioa=min_ioa
        )
        return patch_predictions

    def run_on_images(
        self,
        img_paths: str | pathlib.Path | list,
        return_dataframe: bool = False,
        min_ioa: float = 0.7,
    ) -> dict | pd.DataFrame:
        """Run the model on a list of images.

        Parameters
        ----------
        img_paths : str, pathlib.Path or list
            A list of image paths to run the model on.
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        min_ioa : float, optional
            The minimum intersection over area to consider two polygons the same, by default 0.7

        Returns
        -------
        dict or pd.DataFrame
            A dictionary of predictions for each image or a DataFrame if `as_dataframe` is True.
        """

        if isinstance(img_paths, (str, pathlib.Path)):
            img_paths = [img_paths]

        for img_path in tqdm(img_paths):
            _ = self.run_on_image(img_path, return_outputs=False, min_ioa=min_ioa)

        if return_dataframe:
            return self._dict_to_dataframe(self.patch_predictions, geo=False)
        return self.patch_predictions

    def run_on_image(
        self,
        img_path: str | pathlib.Path,
        return_outputs=False,
        return_dataframe: bool = False,
        min_ioa: float = 0.7,
    ) -> dict | pd.DataFrame:
        """Run the model on a single image.

        Parameters
        ----------
        img_path : str or pathlib.Path
            The path to the image to run the model on.
        return_outputs : bool, optional
            Whether to return the outputs direct from the model, by default False
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        min_ioa : float, optional
            The minimum intersection over area to consider two polygons the same, by default 0.7

        Returns
        -------
        dict or pd.DataFrame
            The predictions for the image or the outputs from the model if `return_outputs` is True.
        """
        # load image
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # run inference
        outputs = self.predictor(img_array)
        outputs["image_id"] = os.path.basename(img_path)
        outputs["img_path"] = img_path

        if return_outputs:
            return outputs

        self.get_patch_predictions(outputs, min_ioa=min_ioa)

        if return_dataframe:
            return self._dict_to_dataframe(self.patch_predictions, geo=False)
        return self.patch_predictions

    def _deduplicate(self, image_id, min_ioa=0.7):
        polygons = [
            instance[0].buffer(0) for instance in self.patch_predictions[image_id]
        ]

        def calc_ioa(i, j):
            return polygons[i].intersection(polygons[j]).area / polygons[i].area

        i_matrix = np.zeros((len(polygons), len(polygons)))

        for i, j in combinations(range(len(polygons)), 2):
            if polygons[i].intersects(polygons[j]):
                ioa_i = calc_ioa(i, j)
                ioa_j = calc_ioa(j, i)
                if ioa_i > ioa_j and ioa_i > min_ioa:
                    i_matrix[i, j] = ioa_i
                elif ioa_j > ioa_i and ioa_j > min_ioa:
                    i_matrix[j, i] = ioa_j

        for i, row in enumerate(i_matrix):
            if any([ioa != 0 for ioa in row]):
                self.patch_predictions[image_id][i] = None
                i_matrix[:, i] = 0

        self.patch_predictions[image_id] = [
            prediction
            for prediction in self.patch_predictions[image_id]
            if prediction is not None
        ]

    def convert_to_parent_pixel_bounds(
        self,
        patch_df: pd.DataFrame = None,
        return_dataframe: bool = False,
        deduplicate: bool = False,
    ) -> dict | pd.DataFrame:
        """Convert the patch predictions to parent predictions by converting pixel bounds.

        Parameters
        ----------
        patch_df : pd.DataFrame, optional
            Dataframe containing patch information, by default None
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        deduplicate : bool, optional
            Whether to deduplicate the parent predictions, by default False.
            Depending on size of parent images, this can be slow.

        Returns
        -------
        dict or pd.DataFrame
            A dictionary of predictions for each parent image or a DataFrame if `as_dataframe` is True.

        Raises
        ------
        ValueError
            If `patch_df` is not available.
        """
        if patch_df is None:
            if self.patch_df is not None:
                patch_df = self.patch_df
            else:
                raise ValueError("[ERROR] Please provide a `patch_df`")

        for image_id, prediction in self.patch_predictions.items():
            parent_id = patch_df.loc[image_id, "parent_id"]
            if parent_id not in self.parent_predictions.keys():
                self.parent_predictions[parent_id] = []

            for instance in prediction:
                polygon = instance[0]

                xx, yy = (np.array(i) for i in polygon.exterior.xy)
                xx = xx + patch_df.loc[image_id, "pixel_bounds"][0]  # add min_x
                yy = yy + patch_df.loc[image_id, "pixel_bounds"][1]  # add min_y

                parent_polygon = Polygon(zip(xx, yy))
                self.parent_predictions[parent_id].append(
                    [parent_polygon, *instance[1:]]
                )

            if deduplicate:
                self._deduplicate_parent_level(parent_id)

        if return_dataframe:
            return self._dict_to_dataframe(self.parent_predictions, geo=False)
        return self.parent_predictions

    def _deduplicate_parent_level(self, image_id, min_ioa=0.7):
        polygons = [
            instance[0].buffer(0) for instance in self.parent_predictions[image_id]
        ]

        def calc_ioa(i, j):
            return polygons[i].intersection(polygons[j]).area / polygons[i].area

        i_matrix = np.zeros((len(polygons), len(polygons)))

        for i, j in combinations(range(len(polygons)), 2):
            if polygons[i].intersects(polygons[j]):
                ioa_i = calc_ioa(i, j)
                ioa_j = calc_ioa(j, i)
                if ioa_i > ioa_j and ioa_i > min_ioa:
                    i_matrix[i, j] = ioa_i
                elif ioa_j > ioa_i and ioa_j > min_ioa:
                    i_matrix[j, i] = ioa_j

        for i, row in enumerate(i_matrix):
            if any([ioa != 0 for ioa in row]):
                self.parent_predictions[image_id][i] = None
                i_matrix[:, i] = 0

        self.parent_predictions[image_id] = [
            prediction
            for prediction in self.parent_predictions[image_id]
            if prediction is not None
        ]

    def convert_to_coords(
        self,
        parent_df: pd.DataFrame = None,
        return_dataframe: bool = False,
    ) -> dict | pd.DataFrame:
        """Convert the parent predictions to georeferenced predictions by converting pixel bounds to coordinates.

        Parameters
        ----------
        parent_df : pd.DataFrame, optional
            Dataframe containing parent image information, by default None
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False

        Returns
        -------
        dict or pd.DataFrame
            A dictionary of predictions for each parent image or a DataFrame if `as_dataframe` is True.

        Raises
        ------
        ValueError
            If `parent_df` is not available.
        """
        if parent_df is None:
            if self.parent_df is not None:
                parent_df = self.parent_df
            else:
                raise ValueError("[ERROR] Please provide a `parent_df`")

        if self.parent_predictions == {}:
            print("[INFO] Converting patch pixel bounds to parent pixel bounds.")
            _ = self.convert_to_parent_pixel_bounds()

        for parent_id, prediction in self.parent_predictions.items():
            if parent_id not in self.geo_predictions.keys():
                self.geo_predictions[parent_id] = []

                for instance in prediction:
                    polygon = instance[0]

                    xx, yy = (np.array(i) for i in polygon.exterior.xy)
                    xx = (
                        xx * parent_df.loc[parent_id, "dlon"]
                        + parent_df.loc[parent_id, "coordinates"][0]
                    )
                    yy = (
                        parent_df.loc[parent_id, "coordinates"][3]
                        - yy * parent_df.loc[parent_id, "dlat"]
                    )

                    crs = parent_df.loc[parent_id, "crs"]

                    parent_polygon_geo = Polygon(zip(xx, yy))
                    self.geo_predictions[parent_id].append(
                        [parent_polygon_geo, crs, *instance[1:]]
                    )

        if return_dataframe:
            return self._dict_to_dataframe(self.geo_predictions, geo=True)
        return self.geo_predictions

    def save_to_geojson(
        self,
        save_path: str | pathlib.Path,
    ) -> None:
        """Save the georeferenced predictions to a GeoJSON file.

        Parameters
        ----------
        save_path : str | pathlib.Path, optional
            Path to save the GeoJSON file
        """

        geo_df = self._dict_to_dataframe(self.geo_predictions, geo=True)

        # get the crs (should be the same for all polygons)
        assert geo_df["crs"].nunique() == 1
        crs = geo_df["crs"].unique()[0]

        geo_df = geopd.GeoDataFrame(geo_df, geometry="polygon", crs=crs)
        geo_df.to_file(save_path, driver="GeoJSON")

    def show(
        self,
        image_id: str,
        figsize: tuple | None = (10, 10),
        border_color: str | None = "r",
        text_color: str | None = "b",
        image_width_resolution: int | None = None,
        return_fig: bool = False,
    ) -> None:
        """Show the predictions on an image.

        Parameters
        ----------
        image_id : str
            The image ID to show the predictions on.
        figsize : tuple | None, optional
            The size of the figure, by default (10, 10)
        border_color : str | None, optional
            The color of the border of the polygons, by default "r"
        text_color : str | None, optional
            The color of the text, by default "b"
        image_width_resolution : int | None, optional
            The maximum resolution of the image width, by default None
        return_fig : bool, optional
            Whether to return the figure, by default False

        Returns
        -------
        fig
            The matplotlib figure if `return_fig` is True.

        Raises
        ------
        ValueError
            If the image ID is not found in the patch or parent predictions.
        """

        if image_id in self.patch_predictions.keys():
            preds = self.patch_predictions
            image_path = self.patch_df.loc[image_id, "image_path"]

        elif image_id in self.parent_predictions.keys():
            preds = self.parent_predictions
            image_path = self.parent_df.loc[image_id, "image_path"]

        else:
            raise ValueError(
                f"[ERROR] {image_id} not found in patch or parent predictions."
            )

        img = Image.open(image_path)

        # if image_width_resolution is specified, resize the image
        if image_width_resolution:
            new_width = int(image_width_resolution)
            rescale_factor = new_width / img.width
            new_height = int(img.height * rescale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

        # check if grayscale
        if len(img.getbands()) == 1:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, zorder=1)
        else:
            ax.imshow(img, zorder=1)

        for instance in preds[image_id]:
            polygon = np.array(instance[0].exterior.coords.xy)
            center = instance[0].centroid.coords.xy
            patch = patches.Polygon(polygon.T, edgecolor=border_color, facecolor="none")
            ax.add_patch(patch)
            ax.text(
                center[0][0], center[1][0], instance[1], fontsize=8, color=text_color
            )

        plt.axis("off")
        plt.title(image_id)

        if return_fig:
            return fig
