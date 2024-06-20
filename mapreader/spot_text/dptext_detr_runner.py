from __future__ import annotations

import pathlib

try:
    import adet
except ImportError:
    raise ImportError(
        "[ERROR] Please install DPText-DETR from the following link: https://github.com/rwood-97/DPText-DETR"
    )

import numpy as np
import pandas as pd
from adet.config import get_cfg

try:
    from detectron2.engine import DefaultPredictor
except ImportError:
    raise ImportError("[ERROR] Please install Detectron2")

from shapely import Polygon

# first assert we are using the dptext detr version of adet
if adet.__version__ != "0.2.0-dptext-detr":
    raise ImportError(
        "[ERROR] Please install DPText-DETR from the following link: https://github.com/rwood-97/DPText-DETR"
    )

from .runner_base import Runner


class DPTextDETRRunner(Runner):
    def __init__(
        self,
        patch_df: pd.DataFrame = None,
        parent_df: pd.DataFrame = None,
        cfg_file: str
        | pathlib.Path = "./DPText-DETR/configs/DPText_DETR/ArT/R_50_poly.yaml",
        weights_file: str | pathlib.Path = "./art_final.pth",
        device: str = "cpu",
    ) -> None:
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
        self._deduplicate(image_id, min_ioa=min_ioa)

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
