from __future__ import annotations

import pathlib
import re

import geopandas as gpd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xyzservices as xyz
from PIL import Image
from shapely import LineString, MultiPolygon, Polygon

from .runner_base import Runner


class RecRunner(Runner):
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
        ctrl_pnts = instances.ctrl_points.numpy()
        scores = instances.scores.tolist()
        recs = instances.recs
        bd_pts = np.asarray(instances.bd)

        self._post_process(image_id, ctrl_pnts, scores, recs, bd_pts)
        self._deduplicate(image_id, min_ioa=min_ioa)

        if return_dataframe:
            return self._dict_to_dataframe(self.patch_predictions, geo=False)
        return self.patch_predictions

    def _process_ctrl_pnt(self, pnt):
        points = pnt.reshape(-1, 2)
        return points

    def _post_process(self, image_id, ctrl_pnts, scores, recs, bd_pnts, alpha=0.4):
        for ctrl_pnt, score, rec, bd in zip(ctrl_pnts, scores, recs, bd_pnts):
            # draw polygons
            if bd is not None:
                bd = np.hsplit(bd, 2)
                bd = np.vstack([bd[0], bd[1][::-1]])
                polygon = Polygon(bd).buffer(0)

                if isinstance(polygon, MultiPolygon):
                    polygon = polygon.convex_hull

            # draw center lines
            line = self._process_ctrl_pnt(ctrl_pnt)
            line = LineString(line)

            # draw text
            text = self._ctc_decode_recognition(rec)
            if self.voc_size == 37:
                text = text.upper()
            # text = "{:.2f}: {}".format(score, text)
            text = f"{text}"
            score = f"{score:.2f}"

            self.patch_predictions[image_id].append([polygon, text, score])

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
            Whether the dictionary is at parent level, by default False

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the predictions.
        """
        if geo:
            columns = ["geometry", "crs", "text", "score"]
        else:
            columns = ["geometry", "text", "score"]

        if parent:
            columns.append("patch_id")

        if len(preds.keys()):
            preds_df = pd.concat(
                pd.DataFrame(
                    preds[k],
                    index=np.full(len(preds[k]), k),
                    columns=columns,
                )
                for k in preds.keys()
            )
        else:
            preds_df = pd.DataFrame(columns=columns)  # empty dataframe

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
        preds_df.reset_index(inplace=True)  # reset index to get image_id as a column
        return preds_df

    def search_preds(
        self, search_text: str, ignore_case: bool = True, return_dataframe: bool = False
    ) -> dict | pd.DataFrame:
        """Search the predictions for specific text. Accepts regex.

        Parameters
        ----------
        search_text : str
            The text to search for. Can be a regex pattern.
        ignore_case : bool, optional
            Whether to ignore case when searching, by default True.
        return_dataframe : bool, optional
            Whether to return the results as a pandas DataFrame, by default False.

        Returns
        -------
        dict | pd.DataFrame
            A dictionary containing the search results or a DataFrame if `return_dataframe` is True.

        Raises
        ------
        ValueError
            If no parent predictions are found.
        """
        # reset the search results
        self.search_results = {}

        # whether to ignore case
        kwargs = {"flags": re.IGNORECASE} if ignore_case else {}

        if self.parent_predictions == {}:
            raise ValueError(
                "[ERROR] No parent predictions found. You may need to run `convert_to_parent_pixel_bounds()`."
            )

        for image_id, preds in self.parent_predictions.items():
            for instance in preds:
                # ["geometry", "text", "score"]
                if re.search(search_text, instance[1], **kwargs):
                    if image_id in self.search_results:
                        self.search_results[image_id].append(instance)
                    else:
                        self.search_results[image_id] = [instance]

        if return_dataframe:
            return self._dict_to_dataframe(self.search_results, parent=True)
        return self.search_results

    def show_search_results(
        self,
        parent_id: str,
        figsize: tuple | None = (10, 10),
        border_color: str | None = "r",
        text_color: str | None = "b",
    ) -> None:
        """Show the search results on an image.

        Parameters
        ----------
        parent_id : str
            The image ID to show the predictions on (must be parent level).
        figsize : tuple | None, optional
            The size of the figure, by default (10, 10)
        border_color : str | None, optional
            The color of the border of the polygons, by default "r"
        text_color : str | None, optional
            The color of the text, by default "b".

        Raises
        ------
        ValueError
            If the image ID is not found in the patch or parent predictions.
        """
        if parent_id in self.parent_predictions.keys():
            image_path = self.parent_df.loc[parent_id, "image_path"]
        else:
            raise ValueError(f"[ERROR] {parent_id} not found in parent predictions.")

        img = Image.open(image_path)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        # check if grayscale
        if len(img.getbands()) == 1:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, zorder=1)
        else:
            ax.imshow(img, zorder=1)
        ax.set_title(parent_id)

        preds = self.search_results

        for instance in preds[parent_id]:
            # Instance is:
            # - [geometry, text, score] for det/rec
            polygon = np.array(instance[0].exterior.coords.xy)
            center = instance[0].centroid.coords.xy
            patch = patches.Polygon(polygon.T, edgecolor=border_color, facecolor="none")
            ax.add_patch(patch)
            ax.text(
                center[0][0], center[1][0], instance[1], fontsize=8, color=text_color
            )

        fig.show()

    def _get_geo_search_results(self):
        """Convert search results to georeferenced search results.

        Returns
        -------
        dict
            Dictionary containing georeferenced search results.
        """
        self.check_georeferencing()
        if not self.georeferenced:
            raise ValueError(
                "[ERROR] Cannot convert to coordinates as parent_df does not have 'coordinates' column."
            )

        geo_search_results = {}

        for parent_id, prediction in self.search_results.items():
            if parent_id not in geo_search_results.keys():
                geo_search_results[parent_id] = []

                for instance in prediction:
                    polygon = instance[0]

                    xx, yy = (np.array(i) for i in polygon.exterior.xy)
                    xx = (
                        xx * self.parent_df.loc[parent_id, "dlon"]
                        + self.parent_df.loc[parent_id, "coordinates"][0]
                    )
                    yy = (
                        self.parent_df.loc[parent_id, "coordinates"][3]
                        - yy * self.parent_df.loc[parent_id, "dlat"]
                    )

                    crs = self.parent_df.loc[parent_id, "crs"]

                    parent_polygon_geo = Polygon(zip(xx, yy)).buffer(0)
                    geo_search_results[parent_id].append(
                        [parent_polygon_geo, crs, *instance[1:]]
                    )

        return geo_search_results

    def explore_search_results(
        self,
        parent_id: str,
        xyz_url: str | None = None,
        style_kwargs: dict | None = None,
    ):
        self.check_georeferencing()
        if not self.georeferenced:
            raise ValueError(
                "[ERROR] This method only works for georeferenced results. Please ensure parent_df has 'coordinates' column and run `convert_to_coords` first."
            )

        if parent_id not in self.geo_predictions.keys():
            raise ValueError(f"[ERROR] {parent_id} not found in geo predictions.")

        if style_kwargs is None:
            style_kwargs = {"fillOpacity": 0.2}

        if xyz_url:
            tiles = xyz.TileProvider(name=xyz_url, url=xyz_url, attribution=xyz_url)
        else:
            tiles = xyz.providers.OpenStreetMap.Mapnik

        geo_search_results = self._get_geo_search_results()
        geo_df = self._dict_to_dataframe(geo_search_results, geo=True, parent=True)

        return geo_df[geo_df["image_id"] == parent_id].explore(
            tiles=tiles,
            style_kwds=style_kwargs,
        )

    def save_search_results_to_geojson(
        self,
        save_path: str | pathlib.Path,
    ) -> None:
        """Convert the search results to georeferenced search results and save them to a GeoJSON file.

        Parameters
        ----------
        save_path : str | pathlib.Path
            The path to save the GeoJSON file.

        Raises
        ------
        ValueError
            If no search results are found.
        """
        if self.search_results == {}:
            raise ValueError("[ERROR] No results to save!")

        geo_search_results = self._get_geo_search_results()

        geo_df = self._dict_to_dataframe(geo_search_results, geo=True, parent=True)
        geo_df.to_file(save_path, driver="GeoJSON", engine="pyogrio")
