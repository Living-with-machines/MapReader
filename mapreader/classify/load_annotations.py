#!/usr/bin/env python
from __future__ import annotations

import os
import pathlib
import re
from decimal import Decimal
from typing import Callable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torchvision.transforms import Compose

from mapreader.utils.load_frames import load_from_csv, load_from_geojson

from .datasets import PatchContextDataset, PatchDataset


class AnnotationsLoader:
    """
    A class for loading annotations and preparing datasets and dataloaders for
    use in training/validation of a model.
    """

    def __init__(self):
        self.annotations = pd.DataFrame()
        self.labels_map = {}
        self.reviewed = pd.DataFrame()
        self.patch_paths_col = None
        self.label_col = None
        self.datasets = None

    def load(
        self,
        annotations: str | pathlib.Path | pd.DataFrame | gpd.GeoDataFrame,
        labels_map: dict | None = None,
        delimiter: str = ",",
        images_dir: str | None = None,
        remove_broken: bool = True,
        ignore_broken: bool = False,
        patch_paths_col: str | None = "image_path",
        label_col: str | None = "label",
        append: bool = True,
        scramble_frame: bool = False,
        reset_index: bool = False,
    ):
        """
        Loads annotations from a CSV/TSV/geojson file, a pandas DataFrame or a geopandas GeoDataFrame.
        Sets the ``patch_paths_col`` and ``label_col`` attributes.

        Parameters
        ----------
        annotations : str | pathlib.Path | pd.DataFrame | gpd.GeoDataFrame
            The annotations.
            Can either be the path to a CSV/TSV/geojson file, a pandas DataFrame or a geopandas GeoDataFrame.
        labels_map : Optional[dict], optional
            A dictionary mapping labels to indices. If not provided, labels will be mapped to indices based on the order in which they appear in the annotations dataframe. By default None.
        delimiter : str, optional
            The delimiter to use when loading the csv file as a DataFrame, by default ",".
        images_dir : str, optional
            The path to the directory in which patches are stored.
            This argument should be passed if image paths are different from the path saved in existing annotations.
            If None, no updates will be made to the image paths in the annotations DataFrame/csv.
            By default None.
        remove_broken : bool, optional
            Whether to remove annotations with broken image paths.
            If False, annotations with broken paths will remain in annotations DataFrame and may cause issues!
            By default True.
        ignore_broken : bool, optional
            Whether to ignore broken image paths (only valid if remove_broken=False).
            If True, annotations with broken paths will remain in annotations DataFrame and no error will be raised. This may cause issues!
            If False, annotations with broken paths will raise error. By default, False.
        patch_paths_col : str, optional
            The name of the column containing the image paths, by default "image_path".
        label_col : str, optional
            The name of the column containing the image labels, by default "label".
        append : bool, optional
            Whether to append the annotations to a pre-existing ``annotations`` DataFrame.
            If False, existing DataFrame will be overwritten.
            By default True.
        scramble_frame : bool, optional
            Whether to shuffle the rows of the DataFrame, by default False.
        reset_index : bool, optional
            Whether to reset the index of the DataFrame (e.g. after shuffling), by default False.

        Raises
        ------
        ValueError
            If ``annotations`` is passed as something other than a string or pd.DataFrame.
        """

        if not self.patch_paths_col:
            self.patch_paths_col = patch_paths_col
        elif self.patch_paths_col != patch_paths_col:
            print(
                f'[WARNING] Patch paths column was previously "{self.patch_paths_col}, but will now be set to {patch_paths_col}.'
            )
            self.patch_paths_col = patch_paths_col

        if not self.label_col:
            self.label_col = label_col
        elif self.label_col != label_col:
            print(
                f'[WARNING] Label column was previously "{self.label_col}, but will now be set to {label_col}.'
            )
            self.label_col = label_col

        if isinstance(annotations, (str, pathlib.Path)):
            annotations = self._load_annotations_file(
                annotations, delimiter, scramble_frame, reset_index
            )
        if not isinstance(annotations, pd.DataFrame):
            raise ValueError(
                "[ERROR] Please pass ``annotations`` as a path to a CSV/TSV/geojson file, a pandas DataFrame or a geopandas GeoDataFrame."
            )

        if images_dir:
            abs_images_dir = os.path.abspath(images_dir)
            annotations[self.patch_paths_col] = annotations.index.map(
                lambda x: os.path.join(abs_images_dir, x)
            )

        annotations = annotations.astype(
            {self.label_col: str}
        )  # ensure labels are interpreted as strings

        if append:
            self.annotations = pd.concat([self.annotations, annotations])
        else:
            self.annotations = annotations

        self._check_patch_paths(
            remove_broken=remove_broken, ignore_broken=ignore_broken
        )

        self.unique_labels = self.annotations[self.label_col].unique().tolist()

        # if labels_map is explicitly provided
        if labels_map:
            self.labels_map = dict(
                sorted(labels_map.items())
            )  # sort labels_map by keys
            if not set(self.unique_labels).issubset(set(labels_map.values())):
                raise ValueError(
                    "[ERROR] There are label(s) in the annotations that are not in the labels map. Please check the labels_map."
                )
        # if inferring labels_map
        else:
            if append:
                for label in self.unique_labels:
                    if label not in self.labels_map.values():
                        self.labels_map[len(self.labels_map)] = label
            else:
                # reset labels map
                labels_map = {i: label for i, label in enumerate(self.unique_labels)}
                self.labels_map = labels_map

        self.annotations["label_index"] = self.annotations[self.label_col].apply(
            self._get_label_index
        )

        print(self)

    def _load_annotations_file(
        self,
        annotations: str | pathlib.Path,
        delimiter: str = ",",
        scramble_frame: bool = False,
        reset_index: bool = False,
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """Loads annotations from a CSV/TSV/geojson file.

        Parameters
        ----------
        annotations : str or pathlib.Path
            The path to the annotations file.
        delimiter : str, optional
            The delimiter to use when loading the file as a DataFrame, by default ",".
        scramble_frame : bool, optional
            Whether to shuffle the rows of the DataFrame, by default False.
        reset_index : bool, optional
            Whether to reset the index of the DataFrame (e.g. after shuffling), by default False.

        Returns
        -------
        pd.DataFrame or gpd.GeoDataFrame
            DataFrame containing the annotations.

        Raises
        ------
        ValueError
            If ``annotations`` is passed as something other than a string, pathlib.Path, pd.DataFrame or gpd.GeoDataFrame.
        """

        if re.search(r"\..?sv$", str(annotations)):
            print(f'[INFO] Reading "{annotations}"')
            annotations = load_from_csv(
                annotations,
                delimiter=delimiter,
            )
        elif re.search(r"\..*?json$", str(annotations)):
            annotations = load_from_geojson(annotations)
        else:
            raise ValueError(
                "[ERROR] ``annotations`` must be a path to a CSV/TSV/etc or geojson file, a pandas DataFrame or a geopandas GeoDataFrame."
            )

        if scramble_frame:
            annotations = annotations.sample(frac=1)
        if reset_index:
            annotations.reset_index(drop=False, inplace=True)

        annotations = annotations[
            ~annotations.index.duplicated(keep="first")
        ]  # remove duplicates
        return annotations

    def _check_patch_paths(
        self,
        remove_broken: bool | None = True,
        ignore_broken: bool | None = False,
    ) -> None:
        """
        Checks the file paths of annotations and manages broken paths.

        Parameters
        ----------
        remove_broken : Optional[bool], optional
            Whether to remove annotations with broken image paths.
            If False, annotations with broken paths will remain in annotations DataFrame and may cause issues!
            By default True.
        ignore_broken : Optional[bool], optional
            Whether to ignore broken image paths (only valid if remove_broken=False).
            If True, annotations with broken paths will remain in annotations DataFrame and no error will be raised. This may cause issues!
        """

        if len(self.annotations) == 0:
            return

        broken_paths = []
        for i, patch_path in self.annotations[self.patch_paths_col].items():
            if not os.path.exists(patch_path):
                broken_paths.append(patch_path)
                if remove_broken:
                    self.annotations.drop(i, inplace=True)

        if len(broken_paths) != 0:  # write broken paths to text file
            with open("broken_files.txt", "w") as f:
                for broken_path in broken_paths:
                    f.write(f"{broken_path}\n")

            print(
                f"[WARNING] {len(broken_paths)} files cannot be found.\n\
Check '{os.path.abspath('broken_paths.txt')}' for more details and, if possible, update your file paths using the 'images_dir' argument."
            )

            if remove_broken:
                if len(self.annotations) == 0:
                    raise ValueError(
                        "[ERROR] No annotations remaining. \
Please check your files exist and, if possible, update your file paths using the 'images_dir' argument."
                    )
                else:
                    print(
                        f"[INFO] Annotations with broken file paths have been removed.\n\
Number of annotations remaining: {len(self.annotations)}"
                    )

            else:  # raise error for 'remove_broken=False'
                if ignore_broken:
                    print(
                        f"[WARNING] Continuing with {len(broken_paths)} broken file paths."
                    )
                else:
                    raise ValueError(
                        f"[ERROR] {len(broken_paths)} files cannot be found."
                    )

    def show_patch(self, patch_id: str) -> None:
        """
        Display a patch and its label.

        Parameters
        ----------
        patch_id : str
            The image ID of the patch to show.

        Returns
        -------
        None
        """

        if len(self.annotations) == 0:
            raise ValueError("[ERROR] No annotations loaded.")

        patch_path = self.annotations.loc[patch_id, self.patch_paths_col]
        patch_label = self.annotations.loc[patch_id, self.label_col]
        try:
            img = Image.open(patch_path)
        except FileNotFoundError as e:
            e.add_note(
                f'[ERROR] File could not be found: "{patch_path}".\n\n\
Please check your image paths in your annonations.csv file and update them if necessary.'
            )

        plt.imshow(img)
        plt.axis("off")
        plt.title(patch_label)
        plt.show()

    def print_unique_labels(self) -> None:
        """Prints unique labels

        Raises
        ------
        ValueError
            If no annotations are found.
        """
        if len(self.annotations) == 0:
            raise ValueError("[ERROR] No annotations loaded.")

        print(f"[INFO] Unique labels: {self.unique_labels}")

    def review_labels(
        self,
        label_to_review: str | None = None,
        chunks: int = 8 * 3,
        num_cols: int = 8,
        exclude_df: pd.DataFrame | gpd.GeoDataFrame | None = None,
        include_df: pd.DataFrame | gpd.GeoDataFrame | None = None,
        deduplicate_col: str = "image_id",
    ) -> None:
        """
        Perform image review on annotations and update labels for a given
        label or all labels.

        Parameters
        ----------
        label_to_review : str, optional
            The target label to review. If not provided, all labels will be
            reviewed, by default ``None``.
        chunks : int, optional
            The number of images to display at a time, by default ``24``.
        num_cols : int, optional
            The number of columns in the display, by default ``8``.
        exclude_df : pandas.DataFrame or gpd.GeoDataFrame or None, optional
            A DataFrame of images to exclude from review, by default ``None``.
        include_df : pandas.DataFrame or gpd.GeoDataFrame or None, optional
            A DataFrame of images to include for review, by default ``None``.
        deduplicate_col : str, optional
            The column to use for deduplicating reviewed images, by default
            ``"image_id"``.

        Returns
        -------
        None

        Notes
        ------
        This method reviews images with their corresponding labels and allows
        the user to change the label for each image.

        Updated labels are saved in
        :attr:`~.classify.load_annotations.AnnotationsLoader.annotations`
        and in a newly created
        :attr:`~.classify.load_annotations.AnnotationsLoader.reviewed`
        DataFrame.

        If ``exclude_df`` is provided, images found in this df are skipped in the review process.

        If ``include_df`` is provided, only images found in this df are reviewed.

        The :attr:`~.classify.load_annotations.AnnotationsLoader.reviewed`
        DataFrame is deduplicated based on the ``deduplicate_col``.
        """
        if len(self.annotations) == 0:
            raise ValueError("[ERROR] No annotations loaded.")

        if label_to_review:
            annots2review = self.annotations[
                self.annotations[self.label_col] == label_to_review
            ]
            annots2review.reset_index(inplace=True, drop=False)
        else:
            annots2review = self.annotations
            annots2review.reset_index(inplace=True, drop=False)

        if exclude_df is not None:
            if isinstance(exclude_df, pd.DataFrame):
                merged_df = pd.merge(
                    annots2review, exclude_df, how="left", indicator=True
                )
                annots2review = merged_df[merged_df["_merge"] == "left_only"].drop(
                    columns="_merge"
                )
                annots2review.reset_index(inplace=True, drop=True)
            else:
                raise ValueError("[ERROR] ``exclude_df`` must be a pandas DataFrame.")

        if include_df is not None:
            if isinstance(include_df, pd.DataFrame):
                annots2review = pd.merge(annots2review, include_df, how="right")
                annots2review.reset_index(inplace=True, drop=True)
            else:
                raise ValueError("[ERROR] ``include_df`` must be a pandas DataFrame.")

        image_idx = 0
        while image_idx < len(annots2review):
            print('[INFO] Type "exit", "end" or "stop" to exit.')
            print(
                f"[INFO] Showing {image_idx}-{image_idx+chunks} out of {len(annots2review)}."  # noqa
            )
            plt.figure(figsize=(num_cols * 3, (chunks // num_cols) * 3))
            counter = 1
            iter_ids = []
            while (counter <= chunks) and (image_idx < len(annots2review)):
                # The first term is just a ceiling division, equivalent to:
                # from math import ceil
                # int(ceil(chunks / num_cols))
                plt.subplot((chunks // num_cols), num_cols, counter)
                patch_path = annots2review.iloc[image_idx][self.patch_paths_col]
                try:
                    img = Image.open(patch_path)
                except FileNotFoundError as e:
                    e.add_note(
                        f'[ERROR] File could not be found: "{patch_path}".\n\n\
Please check your image paths and update them if necessary.'
                    )
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                plt.title(
                    f"{annots2review.iloc[image_idx][self.label_col]} | id: {annots2review.iloc[image_idx].name}"  # noqa
                )
                iter_ids.append(annots2review.iloc[image_idx].name)
                # Add to reviewed
                self.reviewed = pd.concat(
                    [self.reviewed, annots2review.iloc[image_idx : image_idx + 1]]
                )
                try:
                    self.reviewed.drop_duplicates(subset=[deduplicate_col])
                except Exception:
                    pass
                counter += 1
                image_idx += 1
            plt.show()

            print(f"[INFO] IDs of current patches: {iter_ids}")
            q = "\nEnter IDs, comma separated (or press enter to continue): "
            user_input_ids = input(q)

            while user_input_ids.strip().lower() not in [
                "",
                "exit",
                "end",
                "stop",
            ]:
                list_input_ids = user_input_ids.split(",")
                print(
                    f"[INFO] Options for labels:{list(self.annotations[self.label_col].unique())}"
                )
                input_label = input("Enter new label:  ")
                if input_label not in list(self.annotations[self.label_col].unique()):
                    print(
                        f'[ERROR] Label "{input_label}" not found in the annotations. Please enter a valid label.'
                    )
                    continue

                for input_id in list_input_ids:
                    input_id = int(input_id)
                    # Change both annotations and reviewed
                    self.annotations.loc[input_id, self.label_col] = input_label
                    self.reviewed.loc[input_id, self.label_col] = input_label
                    # Update label indices
                    self.annotations.loc[
                        input_id, "label_index"
                    ] = self._get_label_index(input_label)
                    self.reviewed.loc[input_id, "label_index"] = self._get_label_index(
                        input_label
                    )
                    if not (
                        self.annotations[self.label_col].value_counts().tolist()
                        == self.annotations["label_index"].value_counts().tolist()
                    ):
                        raise RuntimeError(
                            f"[ERROR] Label indices do not match label counts. Please check the label indices for label '{input_label}'."
                        )
                    print(
                        f'[INFO] Image {input_id} has been relabelled as "{input_label}"'
                    )

                user_input_ids = input(q)

            if user_input_ids.lower() in ["exit", "end", "stop"]:
                break

        print("[INFO] Exited.")

    def show_sample(self, label_to_show: str, num_samples: int | None = 9) -> None:
        """Show a random sample of images with the specified label (tar_label).

        Parameters
        ----------
        label_to_show : str, optional
            The label of the images to show.
        num_sample : int, optional
            The number of images to show.
            If ``None``, all images with the specified label will be shown. Default is ``9``.

        Returns
        -------
        None
        """
        if len(self.annotations) == 0:
            raise ValueError("[ERROR] No annotations loaded.")

        annot2plot = self.annotations[self.annotations[self.label_col] == label_to_show]

        if len(annot2plot) == 0:
            print(f'[INFO] No annotations found for label "{label_to_show}".')
            return

        annot2plot = annot2plot.sample(frac=1)
        annot2plot.reset_index(drop=False, inplace=True)

        num_samples = min(len(annot2plot), num_samples)

        plt.figure(figsize=(8, num_samples))
        for i in range(num_samples):
            plt.subplot(int(num_samples / 2.0), 3, i + 1)
            patch_path = annot2plot.iloc[i][self.patch_paths_col]
            try:
                img = Image.open(patch_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f'[ERROR] File could not be found: "{patch_path}".\n\n\
Please check your image paths and update them if necessary.'
                )
            plt.imshow(img)
            plt.axis("off")
            plt.title(annot2plot.iloc[i][self.label_col])
        plt.show()

    def create_datasets(
        self,
        frac_train: float = 0.70,
        frac_val: float = 0.15,
        frac_test: float = 0.15,
        random_state: int = 1364,
        train_transform: str | (Compose | Callable) = "train",
        val_transform: str | (Compose | Callable) = "val",
        test_transform: str | (Compose | Callable) = "test",
        context_datasets: bool = False,
        context_df: str | pathlib.Path | pd.DataFrame | gpd.GeoDataFrame | None = None,
    ) -> None:
        """
        Splits the dataset into three subsets: training, validation, and test sets (DataFrames) and saves them as a dictionary in ``self.datasets``.

        Parameters
        ----------
        frac_train : float, optional
            Fraction of the dataset to be used for training.
            By default ``0.70``.
        frac_val : float, optional
            Fraction of the dataset to be used for validation.
            By default ``0.15``.
        frac_test : float, optional
            Fraction of the dataset to be used for testing.
            By default ``0.15``.
        random_state : int, optional
            Random seed to ensure reproducibility. The default is ``1364``.
        train_transform: str, tochvision.transforms.Compose or Callable, optional
            The transform to use on the training dataset images.
            Options are "train", "test" or "val" or, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose).
            By default "train".
        val_transform: str, tochvision.transforms.Compose or Callable, optional
            The transform to use on the validation dataset images.
            Options are "train", "test" or "val" or, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose).
            By default "val".
        test_transform: str, tochvision.transforms.Compose or Callable, optional
            The transform to use on the test dataset images.
            Options are "train", "test" or "val" or, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose).
            By default "test".
        context_datasets: bool, optional
            Whether to create context datasets or not. By default False.
        context_df: str or or pathlib.Path or pandas.DataFrame or gpd.GeoDataFrame, optional
            The DataFrame containing all patches if using context datasets.
            Used to create context images. By default None.

        Raises
        ------
        ValueError
            If the sum of fractions of training, validation and test sets does
            not add up to 1.

        Returns
        -------
        None

        Notes
        -----
        This method saves the split datasets as a dictionary in ``self.datasets``.

        Following fractional ratios provided by the user, where each subset is
        stratified by the values in a specific column (that is, each subset has
        the same relative frequency of the values in the column). It performs
        this splitting by running ``train_test_split()`` twice.

        See ``PatchDataset`` for more information on transforms.
        """
        if len(self.annotations) == 0:
            raise ValueError("[ERROR] No annotations loaded.")

        frac_train = Decimal(str(frac_train))
        frac_val = Decimal(str(frac_val))
        frac_test = Decimal(str(frac_test))

        if sum([frac_train + frac_val + frac_test]) != 1:
            raise ValueError(
                f"[ERROR] ``frac_train`` ({frac_train}), ``frac_val`` ({frac_val}) and ``frac_test`` ({frac_test}) do not add up to 1."
            )  # noqa

        labels = self.annotations[self.label_col]

        # Split original dataframe into train and temp (val+test) dataframes.
        df_train, df_temp, _, labels_temp = train_test_split(
            self.annotations,
            labels,
            stratify=labels,
            test_size=float(1 - frac_train),
            random_state=random_state,
        )

        if frac_test != 0:
            # Split the temp dataframe into val and test dataframes.
            relative_frac_test = Decimal(frac_test / (frac_val + frac_test))
            relative_frac_test = relative_frac_test.quantize(Decimal("0.001"))
            df_val, df_test, _, _ = train_test_split(
                df_temp,
                labels_temp,
                stratify=labels_temp,
                test_size=float(relative_frac_test),
                random_state=random_state,
            )
            if not len(self.annotations) == len(df_train) + len(df_val) + len(df_test):
                raise ValueError(
                    "[ERROR] Number of annotations in the split DataFrames does not match the number of annotations in the original DataFrame."
                )

        else:
            df_val = df_temp
            df_test = None
            if not len(self.annotations) == len(df_train) + len(df_val):
                raise ValueError(
                    "[ERROR] Number of annotations in the split DataFrames does not match the number of annotations in the original DataFrame."
                )

        if context_datasets:
            datasets = self.create_patch_context_datasets(
                context_df,
                train_transform,
                val_transform,
                test_transform,
                df_train,
                df_val,
                df_test,
            )
        else:
            datasets = self.create_patch_datasets(
                train_transform,
                val_transform,
                test_transform,
                df_train,
                df_val,
                df_test,
            )

        dataset_sizes = {
            set_name: len(datasets[set_name]) for set_name in datasets.keys()
        }

        self.datasets = datasets
        self.dataset_sizes = dataset_sizes

        print("[INFO] Number of annotations in each set:")
        for set_name in datasets.keys():
            print(f"    - {set_name}:   {dataset_sizes[set_name]}")

    def create_patch_datasets(
        self, train_transform, val_transform, test_transform, df_train, df_val, df_test
    ):
        train_dataset = PatchDataset(
            df_train,
            train_transform,
            patch_paths_col=self.patch_paths_col,
            label_col=self.label_col,
            label_index_col="label_index",
        )
        val_dataset = PatchDataset(
            df_val,
            val_transform,
            patch_paths_col=self.patch_paths_col,
            label_col=self.label_col,
            label_index_col="label_index",
        )
        if df_test is not None:
            test_dataset = PatchDataset(
                df_test,
                test_transform,
                patch_paths_col=self.patch_paths_col,
                label_col=self.label_col,
                label_index_col="label_index",
            )
            datasets = {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
            }

        else:
            datasets = {"train": train_dataset, "val": val_dataset}

        return datasets

    def create_patch_context_datasets(
        self,
        context_df,
        train_transform,
        val_transform,
        test_transform,
        df_train,
        df_val,
        df_test,
    ):
        train_dataset = PatchContextDataset(
            df_train,
            context_df,
            transform=train_transform,
            patch_paths_col=self.patch_paths_col,
            label_col=self.label_col,
            label_index_col="label_index",
            create_context=True,
        )
        val_dataset = PatchContextDataset(
            df_val,
            context_df,
            transform=val_transform,
            patch_paths_col=self.patch_paths_col,
            label_col=self.label_col,
            label_index_col="label_index",
            create_context=True,
        )
        if df_test is not None:
            test_dataset = PatchContextDataset(
                df_test,
                context_df,
                transform=test_transform,
                patch_paths_col=self.patch_paths_col,
                label_col=self.label_col,
                label_index_col="label_index",
                create_context=True,
            )
            datasets = {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
            }

        else:
            datasets = {"train": train_dataset, "val": val_dataset}

        return datasets

    def create_dataloaders(
        self,
        batch_size: int | None = 16,
        sampler: Sampler | (str | None) | None = "default",
        shuffle: bool | None = False,
        num_workers: int | None = 0,
        **kwargs,
    ) -> None:
        """Creates a dictionary containing PyTorch dataloaders
        saves it to as ``self.dataloaders`` and returns it.

        Parameters
        ----------
        batch_size : int, optional
            The batch size to use for the dataloader. By default ``16``.
        sampler : Sampler, str or None, optional
            The sampler to use when creating batches from the training dataset.
        shuffle : bool, optional
            Whether to shuffle the dataset during training. By default ``False``.
        num_workers : int, optional
            The number of worker threads to use for loading data. By default ``0``.
        **kwds :
            Additional keyword arguments to pass to PyTorch's ``DataLoader`` constructor.

        Returns
        --------
        Dict
            Dictionary containing dataloaders.

        Notes
        -----
        ``sampler`` will only be applied to the training dataset (datasets["train"]).
        """
        if not self.datasets:
            print(
                "[INFO] Creating datasets using default train/val/test split of 0.7:0.15:0.15 and default transformations."
            )
            self.create_datasets()

        datasets = self.datasets

        if isinstance(sampler, str):
            if sampler == "default":
                print("[INFO] Using default sampler.")
                sampler = self._define_sampler()
            else:
                raise ValueError(
                    '[ERROR] ``sampler`` can only be a PyTorch sampler, ``"default"`` or ``None``.'
                )

        if sampler and shuffle:
            print("[INFO] ``sampler`` is defined so train dataset will be un-shuffled.")

        dataloaders = {
            set_name: DataLoader(
                datasets[set_name],
                batch_size=batch_size,
                sampler=sampler if set_name == "train" else None,
                shuffle=False if set_name == "train" else shuffle,
                num_workers=num_workers,
                **kwargs,
            )
            for set_name in datasets.keys()
        }

        self.dataloaders = dataloaders

        return dataloaders

    def _define_sampler(self):
        """Defines a weighted random sampler for the training dataset.
        Weighting are proportional to the reciprocal of number of instances of each label.

        Returns
        -------
        torch.utils.data.WeightedRandomSampler
            The sampler

        Raises
        ------
        ValueError
            If "train" cannot be found in ``self.datasets.keys()``.
        """
        if not self.datasets:
            self.create_datasets()

        datasets = self.datasets

        if "train" in datasets.keys():
            value_counts = (
                datasets["train"].patch_df[self.label_col].value_counts().to_list()
            )
            weights = np.reciprocal(Tensor(value_counts))
            weights = weights.double()
            sampler = WeightedRandomSampler(
                weights[datasets["train"].patch_df["label_index"].tolist()],
                num_samples=len(datasets["train"].patch_df),
            )

        else:
            raise ValueError('[ERROR] "train" should be one the dataset names.')

        return sampler

    def _get_label_index(self, label: str) -> int:
        """Gets the index of a label.

        Parameters
        ----------
        label : str
            A label from the ``label_col`` of the ``patch_df``.

        Returns
        -------
        int
            The index of the label.

        Notes
        -----
        Used to generate the ``label_index`` column.

        """
        index_map = {v: k for k, v in self.labels_map.items()}
        return index_map[label]

    def __str__(self):
        print(f"[INFO] Number of annotations:   {len(self.annotations)}\n")
        if len(self.annotations) > 0:
            value_counts = self.annotations[self.label_col].value_counts()
            print(
                f'[INFO] Number of instances of each label (from column "{self.label_col}"):'
            )
            for label, count in value_counts.items():
                print(f"    - {label}:  {count}")
        return ""
