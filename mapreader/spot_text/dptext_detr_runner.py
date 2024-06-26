from __future__ import annotations

import logging
import os
import pathlib

try:
    import adet
except ImportError:
    raise ImportError(
        "Please install DPText-DETR from the following link: https://github.com/rwood-97/DPText-DETR"
    )

import geopandas as geopd
import numpy as np
import pandas as pd
from adet.config import get_cfg

try:
    from detectron2.engine import DefaultPredictor
except ImportError:
    raise ImportError("Please install Detectron2")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from shapely import Polygon

# first assert we are using the dptext detr version of adet
if adet.__version__ != "0.2.0-dptext-detr":
    raise ImportError(
        "Please install DPText-DETR from the following link: https://github.com/rwood-97/DPText-DETR"
    )

# Set up logging
logger = logging.getLogger(__name__)


class DPTextDETRRunner:
    def __init__(
        self,
        patch_df: pd.DataFrame = None,
        parent_df: pd.DataFrame = None,
        cfg_file: str
        | pathlib.Path = "./DPText-DETR/configs/DPText_DETR/ArT/R_50_poly.yaml",
        weights_file: str | pathlib.Path = "./art_final.pth",
        device: str = "cpu",
    ) -> None:
        # first assert we are using the deep solo version of adet
        assert adet.__version__ == "0.2.0-dptext-detr"

        # setup the dataframes
        self.patch_df = patch_df
        self.parent_df = parent_df

        # set up predictions as dictionaries
        self.patch_predictions = {}
        self.parent_predictions = {}
        self.geo_predictions = {}

        # setup the config
        cfg = get_cfg()  # get a fresh new config
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.WEIGHTS = weights_file
        cfg.MODEL.DEVICE = device

        # setup the predictor
        self.predictor = DefaultPredictor(cfg)

    def run_all(
        self,
        patch_df: pd.DataFrame = None,
        return_dataframe: bool = False,
    ) -> dict | pd.DataFrame:
        """Run the model on all images in the patch dataframe.

        Parameters
        ----------
        patch_df : pd.DataFrame, optional
            Dataframe containing patch information, by default None.
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False

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
            img_paths, return_dataframe=return_dataframe
        )
        return patch_predictions

    def run_on_images(
        self,
        img_paths: str | pathlib.Path | list,
        return_dataframe: bool = False,
    ) -> dict | pd.DataFrame:
        """Run the model on a list of images.

        Parameters
        ----------
        img_paths : str, pathlib.Path or list
            A list of image paths to run the model on.
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False

        Returns
        -------
        dict or pd.DataFrame
            A dictionary of predictions for each image or a DataFrame if `as_dataframe` is True.
        """

        if isinstance(img_paths, (str, pathlib.Path)):
            img_paths = [img_paths]

        for img_path in img_paths:
            _ = self.run_on_image(img_path, return_outputs=False)

        if return_dataframe:
            return self._dict_to_dataframe(self.patch_predictions, geo=False)
        return self.patch_predictions

    def run_on_image(
        self,
        img_path: str | pathlib.Path,
        return_outputs=False,
        return_dataframe: bool = False,
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

        self.get_patch_predictions(outputs)

        if return_dataframe:
            return self._dict_to_dataframe(self.patch_predictions, patch=True)
        return self.patch_predictions

    def get_patch_predictions(
        self,
        outputs: dict,
        return_dataframe: bool = False,
    ) -> dict | pd.DataFrame:
        """Post process the model outputs to get patch predictions.

        Parameters
        ----------
        outputs : dict
            The outputs from the model.
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False

        Returns
        -------
        dict or pd.DataFrame
            A dictionary containing the patch predictions or a DataFrame if `as_dataframe` is True.
        """
        # key for predictions
        image_id = outputs["image_id"]
        self.patch_predictions[image_id] = []

        # get instances
        instances = outputs["instances"].to("cpu")
        scores = instances.scores.tolist()
        pred_classes = instances.pred_classes.tolist()
        bd_pts = np.asarray(instances.polygons)

        self._post_process(image_id, scores, pred_classes, bd_pts)

        if return_dataframe:
            return self._dict_to_dataframe(self.patch_predictions, geo=False)
        return self.patch_predictions

    def _post_process(self, image_id, scores, pred_classes, bd_pnts):
        for score, _pred_class, bd in zip(scores, pred_classes, bd_pnts):
            # draw polygons
            if bd is not None:
                bd = bd.reshape(-1, 2)
                polygon = Polygon(bd)

            score = f"{score:.2f}"

            self.patch_predictions[image_id].append([polygon, score])

    def convert_to_parent_pixel_bounds(
        self,
        patch_df: pd.DataFrame = None,
        return_dataframe: bool = False,
    ) -> dict | pd.DataFrame:
        """Convert the patch predictions to parent predictions by converting pixel bounds.

        Parameters
        ----------
        patch_df : pd.DataFrame, optional
            Dataframe containing patch information, by default None
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False

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
                self.parent_predictions[parent_id].append([parent_polygon, instance[1]])

        if return_dataframe:
            return self._dict_to_dataframe(self.parent_predictions, geo=False)
        return self.parent_predictions

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
            logger.info("Converting patch pixel bounds to parent pixel bounds.")
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
                        [parent_polygon_geo, crs, instance[1]]
                    )

        if return_dataframe:
            return self._dict_to_dataframe(self.geo_predictions, geo=True)
        return self.geo_predictions

    @staticmethod
    def _dict_to_dataframe(
        preds: dict,
        geo: bool = False,
    ) -> pd.DataFrame:
        """Convert the predictions dictionary to a pandas DataFrame.

        Parameters
        ----------
        preds : dict
            A dictionary of predictions.
        geo : bool, optional
            Whether the dictionary is georeferenced coords (or pixel bounds), by default True

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the predictions.
        """
        if geo:
            columns = ["polygon", "crs", "score"]
        else:
            columns = ["polygon", "score"]

        preds_df = pd.concat(
            pd.DataFrame(
                preds[k],
                index=np.full(len(preds[k]), k),
                columns=columns,
            )
            for k in preds.keys()
        )
        preds_df.index.name = "image_id"
        preds_df.reset_index(inplace=True)
        return preds_df

    def save_to_geojson(
        self,
        save_path: str | pathlib.Path = "./dptext-detr_text_outputs.geojson",
    ) -> None:
        """Save the georeferenced predictions to a GeoJSON file.

        Parameters
        ----------
        save_path : str | pathlib.Path, optional
            Path to save the GeoJSON file, by default "./deepsolo_text_outputs.geojson"
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
            plt.imshow(img, cmap="gray", vmin=0, vmax=255, zorder=1)
        else:
            plt.imshow(img, zorder=1)

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
