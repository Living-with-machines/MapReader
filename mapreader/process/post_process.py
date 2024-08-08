#!/usr/bin/env python
from __future__ import annotations

import logging
from ast import literal_eval
from itertools import product

import pandas as pd
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)


class PostProcessor:
    """A class for post-processing predictions on patches using the surrounding context.

    Parameters
    ----------
    patch_df : pd.DataFrame
        the DataFrame containing patches and predictions
    labels_map : dict
        the dictionary mapping label indices to their labels.
        e.g. `{0: "no", 1: "railspace"}`.
    """

    def __init__(
        self,
        patch_df: pd.DataFrame,
        labels_map: dict,
    ):
        required_columns = [
            "parent_id",
            "pixel_bounds",
            "pred",
            "predicted_label",
            "conf",
        ]
        if not all([col in patch_df.columns for col in required_columns]):
            raise ValueError(
                f"[ERROR] Your dataframe must contain the following columns: {required_columns}."
            )

        # ensure lists/tuples are evaluated as such
        for col in patch_df.columns:
            try:
                patch_df[col] = patch_df[col].apply(literal_eval)
            except (ValueError, TypeError, SyntaxError):
                pass

        if patch_df.index.name != "image_id" and "image_id" in patch_df.columns:
            patch_df.set_index("image_id", drop=True, inplace=True)

        if all(
            [col in patch_df.columns for col in ["min_x", "min_y", "max_x", "max_y"]]
        ):
            logger.info(
                "Using existing pixel bounds columns (min_x, min_y, max_x, max_y)."
            )
        else:
            patch_df[["min_x", "min_y", "max_x", "max_y"]] = [*patch_df["pixel_bounds"]]

        # set the patch_df attribute
        self.patch_df = patch_df

        self.labels_map = labels_map
        self._label_patches = None
        self.context = {}

    def __len__(self):
        return len(self.patch_df)

    def get_context(
        self,
        labels: str | list,
    ):
        """Get the context of the patches with the specified labels.

        Parameters
        ----------
        labels : str | list
            The label(s) to get context for.
        """
        if isinstance(labels, str):
            labels = [labels]
        self._label_patches = self.patch_df[
            self.patch_df["predicted_label"].isin(labels)
        ]

        for id in tqdm(self._label_patches.index):
            if id not in self.context:
                context_list = self._get_context_id(id)
                # only add context if all surrounding patches are found
                if len(context_list) == 9:
                    self.context[id] = context_list

    def _get_context_id(
        self,
        id,
    ):
        """Get the context of the patch with the specified index."""
        parent_id = self.patch_df.loc[id, "parent_id"]
        min_x = self.patch_df.loc[id, "min_x"]
        min_y = self.patch_df.loc[id, "min_y"]
        max_x = self.patch_df.loc[id, "max_x"]
        max_y = self.patch_df.loc[id, "max_y"]

        context_grid = [
            *product(
                [
                    (self.patch_df["min_x"], min_x),
                    (min_x, max_x),
                    (max_x, self.patch_df["max_x"]),
                ],
                [
                    (self.patch_df["min_y"], min_y),
                    (min_y, max_y),
                    (max_y, self.patch_df["max_y"]),
                ],
            )
        ]
        # reshape to min_x, min_y, max_x, max_y
        context_grid = [(x[0][0], x[1][0], x[0][1], x[1][1]) for x in context_grid]

        context_list = [
            self.patch_df[
                (self.patch_df["min_x"] == context_loc[0])
                & (self.patch_df["min_y"] == context_loc[1])
                & (self.patch_df["max_x"] == context_loc[2])
                & (self.patch_df["max_y"] == context_loc[3])
                & (self.patch_df["parent_id"] == parent_id)
            ]
            for context_loc in context_grid
        ]
        context_list = [x.index[0] for x in context_list if len(x)]
        return context_list

    def update_preds(self, remap: dict, conf: float = 0.7, inplace: bool = False):
        """Update the predictions of the chosen patches based on their context.

        Parameters
        ----------
        remap : dict
            A dictionary mapping the old labels to the new labels.
        conf : float, optional
            Patches with confidence scores below this value will be relabelled, by default 0.7.
        inplace : bool, optional
            Whether to relabel inplace or create new dataframe columns, by default False
        """
        if self._label_patches is None:
            raise ValueError("[ERROR] You must run `get_context` first.")
        if len(self.context) == 0:
            raise ValueError(
                "[ERROR] No patches to update. Try changing which labels you are updating."
            )

        labels = self._label_patches["predicted_label"].unique()
        if any([label not in remap.keys() for label in labels]):
            raise ValueError(
                f"[ERROR] You must specify a remap for each label in {labels}."
            )

        # add new label to labels_map if not already present (assume label index is next in sequence)
        for new_label in remap.values():
            if new_label not in self.labels_map.values():
                logger.info(
                    f"Adding {new_label} to labels_map at index {len(self.labels_map)}."
                )
                self.labels_map[len(self.labels_map)] = new_label

        for id in tqdm(self.context):
            self._update_preds_id(
                id, labels=labels, remap=remap, conf=conf, inplace=inplace
            )

    def _update_preds_id(
        self, id, labels: str | list, remap: dict, conf: float, inplace: bool = False
    ):
        """Update the predictions of the patch with the specified index."""
        context_list = self.context[id]

        context_df = self.patch_df[self.patch_df.index.isin(context_list)].copy(
            deep=True
        )

        # drop central patch from context
        context_df.drop(index=id, inplace=True)

        # reverse the labels_map dict
        label_index_dict = {v: k for k, v in self.labels_map.items()}

        prefix = "" if inplace else "new_"
        if (not any(context_df["predicted_label"].isin(labels))) & (
            self.patch_df.loc[id, "conf"] < conf
        ):
            self.patch_df.loc[id, f"{prefix}predicted_label"] = remap[
                self.patch_df.loc[id, "predicted_label"]
            ]
            self.patch_df.loc[id, f"{prefix}pred"] = label_index_dict[
                self.patch_df.loc[id, f"{prefix}predicted_label"]
            ]
