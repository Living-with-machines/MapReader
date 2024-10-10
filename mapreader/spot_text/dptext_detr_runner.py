from __future__ import annotations

import pathlib

try:
    import adet
except ImportError:
    raise ImportError(
        "[ERROR] Please install DPText-DETR from the following link: https://github.com/rwood-97/DPText-DETR"
    )

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from adet.config import get_cfg

try:
    from detectron2.engine import DefaultPredictor
except ImportError:
    raise ImportError("[ERROR] Please install Detectron2")

from shapely import MultiPolygon, Polygon

# first assert we are using the dptext detr version of adet
if adet.__version__ != "0.2.0-dptext-detr":
    raise ImportError(
        "[ERROR] Please install DPText-DETR from the following link: https://github.com/rwood-97/DPText-DETR"
    )

from .runner_base import Runner


class DPTextDETRRunner(Runner):
    def __init__(
        self,
        patch_df: pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path,
        parent_df: pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path | None = None,
        cfg_file: str
        | pathlib.Path = "./DPText-DETR/configs/DPText_DETR/ArT/R_50_poly.yaml",
        weights_file: str | pathlib.Path = "./art_final.pth",
        device: str = "default",
        delimiter: str = ",",
    ) -> None:
        """Initialise the DPText-DETR runner.

        Parameters
        ----------
        patch_df : pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path,
            The dataframe containing the patch information. If a string/path, it should be a path to a CSV/TSV/etc or geojson file.
        parent_df : pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path,
            The dataframe containing the parent information. If a string/path, it should be a path to a CSV/TSV/etc or geojson file, by default None.
        cfg_file : str | pathlib.Path, optional
            The path to the config file (yaml), by default "./DPText-DETR/configs/DPText_DETR/ArT/R_50_poly.yaml"
        weights_file : str | pathlib.Path, optional
            The path to the weights file (.pth), by default "./art_final.pth"
        device : str, optional
            The device to use for the model, by default "default". If default, the device will be set to cuda if available, otherwise cpu.
        delimiter : str, optional
            The delimiter to use if loading dataframes from CSV files, by default ",".
        """
        # setup the dataframes
        self._load_df(patch_df, parent_df, delimiter)

        # set up predictions as dictionaries
        self.patch_predictions = {}
        self.parent_predictions = {}
        self.geo_predictions = {}

        # setup the config
        cfg = get_cfg()  # get a fresh new config
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.WEIGHTS = weights_file
        if device == "default":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.DEVICE = device

        # setup the predictor
        self.predictor = DefaultPredictor(cfg)

    def get_patch_predictions(
        self,
        outputs: dict,
        return_dataframe: bool = False,
        min_ioa: float = 0.7,
    ) -> dict | pd.DataFrame:
        """Post process the model outputs to get patch predictions.

        Parameters
        ----------
        outputs : dict
            The outputs from the model.
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        min_ioa : float, optional
            The minimum intersection over area to consider two polygons the same, by default 0.7

        Returns
        -------
        dict or pd.DataFrame
            A dictionary containing the patch predictions or a DataFrame if `return_dataframe` is True.
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
        self._deduplicate(image_id, min_ioa=min_ioa)

        if return_dataframe:
            return self._dict_to_dataframe(self.patch_predictions, geo=False)
        return self.patch_predictions

    def _post_process(self, image_id, scores, pred_classes, bd_pnts):
        for score, _pred_class, bd in zip(scores, pred_classes, bd_pnts):
            # draw polygons
            if bd is not None:
                bd = bd.reshape(-1, 2)
                polygon = Polygon(bd).buffer(0)

                if isinstance(polygon, MultiPolygon):
                    polygon = polygon.convex_hull

            score = f"{score:.2f}"

            self.patch_predictions[image_id].append([polygon, score])

    @staticmethod
    def _dict_to_dataframe(
        preds: dict,
        geo: bool = False,
        parent: bool = False,
    ) -> pd.DataFrame:
        """Convert the predictions dictionary to a pandas DataFrame.

        Parameters
        ----------
        preds : dict
            A dictionary of predictions.
        geo : bool, optional
            Whether the dictionary is georeferenced coords (or pixel bounds), by default True
        parent : bool, optional
            Whether the dictionary is at the parent level, by default False

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the predictions.
        """
        if geo:
            columns = ["geometry", "crs", "score"]
        else:
            columns = ["geometry", "score"]

        if parent:
            columns.append("patch_id")

        preds_df = pd.concat(
            pd.DataFrame(
                preds[k],
                index=np.full(len(preds[k]), k),
                columns=columns,
            )
            for k in preds.keys()
        )

        if geo:
            # get the crs (should be the same for all)
            if not preds_df["crs"].nunique() == 1:
                raise ValueError("[ERROR] Multiple crs found in the predictions.")
            crs = preds_df["crs"].unique()[0]

            preds_df = gpd.GeoDataFrame(
                preds_df,
                geometry="geometry",
                crs=crs,
            )

        preds_df.index.name = "image_id"
        preds_df.reset_index(inplace=True)
        return preds_df
