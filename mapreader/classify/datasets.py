#!/usr/bin/env python
from __future__ import annotations

import logging
import os
from ast import literal_eval
from itertools import product
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set up logging
logger = logging.getLogger(__name__)


# Import parhugin
try:
    from parhugin import multiFunc

    parhugin_installed = True
except ImportError:
    logger.warning(
        "parhugin (https://github.com/kasra-hosseini/parhugin) is not installed, continue without it."  # noqa
    )
    parhugin_installed = False


class PatchDataset(Dataset):
    """A PyTorch Dataset class for loading image patches from a DataFrame.

    Parameters
    ----------
    patch_df : pandas.DataFrame or str
        DataFrame or path to csv file containing the paths to image patches and their labels.
    transform : Union[str, transforms.Compose, Callable]
        The transform to use on the image.
        A string can be used to call default transforms - options are "train", "test" or "val".
        Alternatively, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose) that takes in an image
        and performs image transformations can be used.
        At minimum, transform should be ``torchvision.transforms.ToTensor()``.
    delimiter : str, optional
        The delimiter to use when reading the dataframe. By default ``","``.
    patch_paths_col : str, optional
        The name of the column in the DataFrame containing the image paths. Default is "image_path".
    label_col : str, optional
        The name of the column containing the image labels. Default is None.
    label_index_col : str, optional
        The name of the column containing the indices of the image labels. Default is None.
    image_mode : str, optional
        The color format to convert the image to. Default is "RGB".

    Attributes
    ----------
    patch_df : pandas.DataFrame
        DataFrame containing the paths to image patches and their labels.
    label_col : str
        The name of the column containing the image labels.
    label_index_col : str
        The name of the column containing the labels indices.
    patch_paths_col : str
        The name of the column in the DataFrame containing the image
        paths.
    image_mode : str
        The color format to convert the image to.
    unique_labels : list
        The unique labels in the label column of the patch_df DataFrame.
    transform : callable
        A callable object (a torchvision transform) that takes in an image
        and performs image transformations.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(idx)
        Retrieves the image, its label and the index of that label at the given index in the dataset.
    return_orig_image(idx)
        Retrieves the original image at the given index in the dataset.
    _default_transform(t_type, resize2)
        Returns a transforms.Compose containing the default image transformations for the train and validation sets.

    Raises
    ------
    ValueError
        If ``label_col`` not in ``patch_df``.
    ValueError
        If ``label_index_col`` not in ``patch_df``.
    ValueError
        If ``transform`` passed as a string, but not one of "train", "test" or "val".
    """

    def __init__(
        self,
        patch_df: pd.DataFrame | str,
        transform: str | (transforms.Compose | Callable),
        delimiter: str = ",",
        patch_paths_col: str | None = "image_path",
        label_col: str | None = None,
        label_index_col: str | None = None,
        image_mode: str | None = "RGB",
    ):
        if isinstance(patch_df, pd.DataFrame):
            self.patch_df = patch_df

        elif isinstance(patch_df, str):
            if os.path.isfile(patch_df):
                logger.info(f'Reading "{patch_df}".')
                patch_df = pd.read_csv(patch_df, sep=delimiter)
                # ensure tuple/list columns are read as such
                patch_df = self._eval_df(patch_df)
                self.patch_df = patch_df
            else:
                raise ValueError(f'[ERROR] "{patch_df}" cannot be found.')

        else:
            raise ValueError(
                "[ERROR] Please pass ``patch_df`` as a string (path to csv file) or pd.DataFrame."
            )

        # force index to be image_id
        if (
            "image_id" in self.patch_df.columns
            and self.patch_df.index.name != "image_id"
        ):
            self.patch_df.set_index("image_id", drop=True, inplace=True)

        self.label_col = label_col
        self.label_index_col = label_index_col
        self.image_mode = image_mode
        self.patch_paths_col = patch_paths_col
        self.unique_labels = []

        if self.label_col:
            if self.label_col not in self.patch_df.columns:
                raise ValueError(
                    f"[ERROR] Label column ({label_col}) not in dataframe."
                )
            else:
                self.unique_labels = self.patch_df[self.label_col].unique().tolist()

        if self.label_index_col:
            if self.label_index_col not in self.patch_df.columns:
                if self.label_col:
                    logger.info(
                        f"Label index column ({label_index_col}) not in dataframe. Creating column."
                    )
                    self.patch_df[self.label_index_col] = self.patch_df[
                        self.label_col
                    ].apply(self._get_label_index)
                else:
                    raise ValueError(
                        f"[ERROR] Label index column ({label_index_col}) not in dataframe."
                    )

        if isinstance(transform, str):
            if transform in ["train", "val", "test"]:
                self.transform = self._default_transform(transform)
            else:
                raise ValueError(
                    '[ERROR] ``transform`` can only be "train", "val" or "test" or, a transform.'
                )
        else:
            self.transform = transform

    @staticmethod
    def _eval_df(df):
        for col in df.columns:
            try:
                df[col] = df[col].apply(literal_eval)
            except (ValueError, TypeError, SyntaxError):
                pass
        return df

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.patch_df)

    def __getitem__(
        self, idx: int | torch.Tensor
    ) -> tuple[tuple[torch.Tensor], str, int]:
        """
        Return the image, its label and the index of that label at the given index in the dataset.

        Parameters
        ----------
        idx : int or torch.Tensor
            Index or indices of the desired image.

        Returns
        -------
        Tuple[torch.Tensor, str, int]
            A tuple containing the transformed image, its label the index of that label.

        Notes
        ------
            The label is "" and has index -1 if it is not present in the DataFrame.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.patch_df.iloc[idx][self.patch_paths_col]

        if os.path.exists(img_path):
            img = Image.open(img_path).convert(self.image_mode)
        else:
            raise ValueError(
                f'[ERROR] "{img_path} cannot be found.\n\n\
Please check the image exists, your file paths are correct and that ``.patch_paths_col`` is set to the correct column.'
            )

        img = self.transform(img)

        if self.label_col in self.patch_df.iloc[idx].keys():
            image_label = self.patch_df.iloc[idx][self.label_col]
        else:
            image_label = ""

        if self.label_index_col in self.patch_df.iloc[idx].keys():
            image_label_index = self.patch_df.iloc[idx][self.label_index_col]
        else:
            image_label_index = -1

        return (img,), image_label, image_label_index

    def return_orig_image(self, idx: int | torch.Tensor) -> Image:
        """
        Return the original image associated with the given index.

        Parameters
        ----------
        idx : int or Tensor
            The index of the desired image, or a Tensor containing the index.

        Returns
        -------
        PIL.Image.Image
            The original image associated with the given index.

        Notes
        -----
        This method returns the original image associated with the given index
        by loading the image file using the file path stored in the
        ``patch_paths_col`` column of the ``patch_df`` DataFrame at the given
        index. The loaded image is then converted to the format specified by
        the ``image_mode`` attribute of the object. The resulting
        :class:`PIL.Image.Image` object is returned.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.patch_df.iloc[idx][self.patch_paths_col]

        if os.path.exists(img_path):
            img = Image.open(img_path).convert(self.image_mode)
        else:
            raise ValueError(
                f'[ERROR] "{img_path} cannot be found.\n\n\
Please check the image exists, your file paths are correct and that ``.patch_paths_col`` is set to the correct column.'
            )

        return img

    def _default_transform(
        self,
        t_type: str | None = "train",
        resize: int | tuple[int, int] | None = (224, 224),
    ) -> transforms.Compose:
        """
        Returns the default image transformations for the train, test and validation sets as a transforms.Compose.

        Parameters
        ----------
        t_type : str, optional
            The type of transformation to return. Either "train", "test" or "val".
            Default is "train".
        resize2 : int or Tuple[int, int], optional
            The size in pixels to resize the image to. Default is (224, 224).

        Returns
        -------
        transforms.Compose
            A torchvision.transforms.Compose containing the default image transformations for the specified type.

        Notes
        -----
        "val" and "test" are aliased by this method - both return the same transforms.
        """
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        t_type = "val" if t_type == "test" else t_type  # test and val are synonymous

        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.RandomApply(
                        [
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            # transforms.ColorJitter(brightness=0.3, contrast=0.3), # noqa
                        ],
                        p=0.5,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(normalize_mean, normalize_std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(normalize_mean, normalize_std),
                ]
            ),
        }
        return data_transforms[t_type]

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
        return self.unique_labels.index(label)

    def create_dataloaders(
        self,
        set_name: str = "infer",
        batch_size: int = 16,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        """Creates a dictionary containing a PyTorch dataloader.

        Parameters
        ----------
        set_name : str, optional
            The name to use for the dataloader.
        batch_size : int, optional
            The batch size to use for the dataloader. By default ``16``.
        shuffle : bool, optional
            Whether to shuffle the PatchDataset, by default False
        num_workers : int, optional
            The number of worker threads to use for loading data. By default ``0``.
        **kwargs :
            Additional keyword arguments to pass to PyTorch's ``DataLoader`` constructor.

        Returns
        --------
        Dict
            Dictionary containing dataloaders.
        """

        dataloaders = {
            set_name: DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                **kwargs,
            )
        }

        return dataloaders


# --- Dataset that returns an image, its context and its label
class PatchContextDataset(PatchDataset):
    """
    A PyTorch Dataset class for loading contextual information about image
    patches from a DataFrame.

    Parameters
    ----------
    patch_df : pandas.DataFrame or str
        DataFrame or path to csv file containing the paths to image patches and their labels.
    total_df : pandas.DataFrame or str
        DataFrame or path to csv file containing the paths to all images and their labels.
    transform : str
        Torchvision transform to be applied to context  images.
        Either "train" or "val".
    delimiter : str
        The delimiter to use when reading the csv file. By default ``","``.
    patch_paths_col : str, optional
        The name of the column in the DataFrame containing the image paths. Default is "image_path".
    label_col : str, optional
        The name of the column containing the image labels. Default is None.
    label_index_col : str, optional
        The name of the column containing the indices of the image labels. Default is None.
    image_mode : str, optional
        The color space of the images. Default is "RGB".
    context_dir : str, optional
        The path to context maps (or, where to save context if not created yet).
        Default is "./maps/maps_context".
    create_context : bool, optional
        Whether or not to create context maps. Default is False.
    parent_path : str, optional
        The path to the directory containing parent images. Default is
        "./maps".

    Attributes
    ----------
    patch_df : pandas.DataFrame
        A pandas DataFrame with columns representing image paths, labels,
        and object bounding boxes.
    label_col : str
        The name of the column containing the image labels.
    label_index_col : str
        The name of the column containing the labels indices.
    patch_paths_col : str
        The name of the column in the DataFrame containing the image
        paths.
    image_mode : str
        The color space of the images.
    parent_path : str
        The path to the directory containing parent images.
    create_context : bool
        Whether or not to create context maps.
    context_dir : str
        The path to context maps.
    unique_labels : list or str
        The unique labels in ``label_col``.
    """

    def __init__(
        self,
        patch_df: pd.DataFrame | str,
        total_df: pd.DataFrame | str,
        transform: str,
        delimiter: str = ",",
        patch_paths_col: str | None = "image_path",
        label_col: str | None = None,
        label_index_col: str | None = None,
        image_mode: str | None = "RGB",
        context_dir: str | None = "./maps/maps_context",
        create_context: bool = False,
        parent_path: str | None = "./maps",
    ):
        if isinstance(patch_df, pd.DataFrame):
            self.patch_df = patch_df

        elif isinstance(patch_df, str):
            if os.path.isfile(patch_df):
                logger.info(f'Reading "{patch_df}".')
                patch_df = pd.read_csv(patch_df, sep=delimiter)
                patch_df = self._eval_df(patch_df)
                self.patch_df = patch_df
            else:
                raise ValueError(f'[ERROR] "{patch_df}" cannot be found.')

        else:
            raise ValueError(
                "[ERROR] Please pass ``patch_df`` as a string (path to csv file) or pd.DataFrame."
            )

        if isinstance(total_df, pd.DataFrame):
            self.total_df = total_df

        elif isinstance(total_df, str):
            if os.path.isfile(total_df):
                total_df = pd.read_csv(total_df, sep=delimiter)
                total_df = self._eval_df(total_df)
                self.total_df = total_df
            else:
                raise ValueError(f'[ERROR] "{total_df}" cannot be found.')

        else:
            raise ValueError(
                "[ERROR] Please pass ``total_df`` as a string (path to csv file) or pd.DataFrame."
            )

        # force index to be image_id
        for df in [self.patch_df, self.total_df]:
            if "image_id" in df.columns and df.index.name != "image_id":
                df.set_index("image_id", drop=True, inplace=True)

        self.label_col = label_col
        self.label_index_col = label_index_col
        self.image_mode = image_mode
        self.patch_paths_col = patch_paths_col
        self.parent_path = parent_path
        self.create_context = create_context
        self.context_dir = os.path.abspath(context_dir)

        if self.label_col:
            if self.label_col not in self.patch_df.columns:
                raise ValueError(
                    f"[ERROR] Label column ({self.label_col}) not in dataframe."
                )
            self.unique_labels = self.patch_df[self.label_col].unique().tolist()

        if self.label_index_col:
            if self.label_index_col not in self.patch_df.columns:
                logger.info(
                    f"Label index column ({label_index_col}) not in dataframe. Creating column."
                )
                self.patch_df[self.label_index_col] = self.patch_df[
                    self.label_col
                ].apply(self._get_label_index)

        if isinstance(transform, str):
            if transform in ["train", "val", "test"]:
                self.transform = self._default_transform(transform)
            else:
                raise ValueError(
                    '[ERROR] ``transform`` can only be "train", "val" or "test" or, a transform.'
                )
        else:
            self.transform = transform

    def save_context(
        self,
        processors: int = 10,
        sleep_time: float = 0.001,
        use_parhugin: bool = True,
        overwrite: bool = False,
    ) -> None:
        """
        Save context images for all patches in the patch_df.

        Parameters
        ----------
        processors : int, optional
            The number of required processors for the job, by default 10.
        sleep_time : float, optional
            The time to wait between jobs, by default 0.001.
        use_parhugin : bool, optional
            Whether to use Parhugin to parallelize the job, by default True.
        overwrite : bool, optional
            Whether to overwrite existing parent files, by default False.

        Returns
        -------
        None

        Notes
        -----
        Parhugin is a Python package for parallelizing computations across
        multiple CPU cores. The method uses Parhugin to parallelize the
        computation of saving parent patches to disk. When Parhugin is
        installed and ``use_parhugin`` is set to True, the method parallelizes
        the calling of the ``get_context_id`` method and its corresponding
        arguments. If Parhugin is not installed or ``use_parhugin`` is set to
        False, the method executes the loop over patch indices sequentially
        instead.
        """
        if parhugin_installed and use_parhugin:
            my_proc = multiFunc(num_req_p=processors, sleep_time=sleep_time)
            list_jobs = []
            for id in self.patch_df.index:
                list_jobs.append(
                    [
                        self.get_context_id(
                            id,
                            overwrite=overwrite,
                            save_context=True,
                            return_image=False,
                        ),
                        {},
                    ]
                )

            logger.info(f"Total number of jobs: {len(list_jobs)}")
            # and then adding them to my_proc
            my_proc.add_list_jobs(list_jobs)
            my_proc.run_jobs()
        else:
            for id in self.patch_df.index:
                self.get_context_id(
                    id,
                    overwrite=overwrite,
                    save_context=True,
                    return_image=False,
                )

    @staticmethod
    def _get_empty_square(
        patch_size: tuple[int, int],
    ):
        """Get an empty square image with size (width, height) equal to `patch_size`."""
        im = Image.new(
            size=patch_size,
            mode="RGB",
            color=None,
        )
        return im

    def get_context_id(
        self,
        id,
        overwrite: bool = False,
        save_context: bool = False,
        return_image: bool = True,
    ) -> None:
        """
        Save the parents of a specific patch to the specified location.

        Parameters
        ----------
            id
                Index of the patch in the dataset.
            overwrite : bool, optional
                Whether to overwrite the existing parent files. Default is
                False.
            save_context : bool, optional
                Whether to save the context image. Default is False.
            return_image : bool, optional
                Whether to return the context image. Default is True.

        Raises
        ------
        ValueError
            If the patch is not found in the dataset.

        Returns
        -------
        None
        """
        total_df = self.total_df.copy(deep=True)

        # backwards compatibility with annotator without pixel_bounds and parent_id columns
        if "pixel_bounds" not in total_df.columns:
            total_df["pixel_bounds"] = total_df.index.map(
                lambda x: str(tuple(x.split("-")[1:5]))
            )
        if "parent_id" not in total_df.columns:
            total_df["parent_id"] = total_df.index.map(lambda x: x.split("#")[1])
        total_df = self._eval_df(total_df)

        if not all(
            [col in total_df.columns for col in ["min_x", "min_y", "max_x", "max_y"]]
        ):
            total_df[["min_x", "min_y", "max_x", "max_y"]] = [*total_df.pixel_bounds]

        patch_image = Image.open(total_df.loc[id, self.patch_paths_col]).convert(
            self.image_mode
        )
        patch_width, patch_height = (patch_image.width, patch_image.height)
        parent_id = total_df.loc[id, "parent_id"]
        min_x = total_df.loc[id, "min_x"]
        min_y = total_df.loc[id, "min_y"]
        max_x = total_df.loc[id, "max_x"]
        max_y = total_df.loc[id, "max_y"]

        # get a pixel bounds of context images
        context_grid = [
            *product(
                [
                    (total_df["min_y"], min_y),
                    (min_y, max_y),
                    (max_y, total_df["max_y"]),
                ],
                [
                    (total_df["min_x"], min_x),
                    (min_x, max_x),
                    (max_x, total_df["max_x"]),
                ],
            )
        ]
        # reshape to min_x, min_y, max_x, max_y
        context_grid = [
            (coord[1][0], coord[0][0], coord[1][1], coord[0][1])
            for coord in context_grid
        ]

        # get a list of context images
        context_list = [
            total_df[
                (total_df["min_x"] == context_loc[0])
                & (total_df["min_y"] == context_loc[1])
                & (total_df["max_x"] == context_loc[2])
                & (total_df["max_y"] == context_loc[3])
                & (total_df["parent_id"] == parent_id)
            ]
            for context_loc in context_grid
        ]
        if any([len(context_patch) > 1 for context_patch in context_list]):
            raise ValueError(f"[ERROR] Multiple context patches found for '{id}'.")
        if len(context_list) != 9:
            raise ValueError(f"[ERROR] Missing context images for '{id}'.")

        context_paths = [
            (
                context_patch[self.patch_paths_col].values[0]
                if len(context_patch)
                else None
            )
            for context_patch in context_list
        ]
        context_images = [
            (
                Image.open(context_path).convert(self.image_mode)
                if context_path is not None
                else self._get_empty_square((patch_width, patch_height))
            )
            for context_path in context_paths
        ]

        # split into rows (3x3 grid)
        context_images = [
            context_images[i : i + 3] for i in range(0, len(context_images), 3)
        ]

        total_width = 3 * patch_width
        total_height = 3 * patch_height
        context_image = Image.new(self.image_mode, (total_width, total_height))

        y_offset = 0
        for row in context_images:
            x_offset = 0
            for image in row:
                context_image.paste(image, (x_offset, y_offset))
                x_offset += patch_width
            y_offset += patch_height

        if save_context:
            os.makedirs(self.context_dir, exist_ok=True)
            context_path = os.path.join(
                self.context_dir,
                os.path.basename(total_df.loc[id, self.patch_paths_col]),
            )
            if overwrite or not os.path.exists(context_path):
                context_image.save(context_path)

        if return_image:
            return context_image
        else:
            return

    def plot_sample(self, idx: int) -> None:
        """
        Plot a sample patch and its corresponding context from the dataset.

        Parameters
        ----------
        idx : int
            The index of the sample to plot.

        Returns
        -------
        None
            Displays the plot of the sample patch and its corresponding
            context.

        Notes
        -----
        This method plots a sample patch and its corresponding context side-by-
        side in a single figure with two subplots. The figure size is set to
        10in x 5in, and the titles of the subplots are set to "Patch" and
        "Context", respectively. The resulting figure is displayed using
        the ``matplotlib`` library (required).
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(transforms.ToPILImage()(self.__getitem__(idx)[0][0]))
        plt.title("Patch", size=18)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(transforms.ToPILImage()(self.__getitem__(idx)[0][1]))
        plt.title("Context", size=18)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.show()

    def __getitem__(
        self, idx: int | torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], str, int]:
        """
        Retrieves the patch image, the context image and the label at the
        given index in the dataset (``idx``).

        Parameters
        ----------
        idx : int
            The index of the data to retrieve.

        Returns
        -------
        Tuple(torch.Tensor, torch.Tensor, str, int)
            A tuple containing the transformed image, the context image, the image label the index of that label.

        Notes
        ------
            The label is "" and has index -1 if it is not present in the DataFrame.

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.patch_df.iloc[idx].name
        img_path = self.patch_df.iloc[idx][self.patch_paths_col]

        if self.create_context:
            context_img = self.get_context_id(image_id, return_image=True)
        else:
            context_img = Image.open(
                os.path.join(self.context_dir, os.path.basename(img_path))
            ).convert(self.image_mode)

        context_img = self.transform(context_img)

        if self.label_col in self.patch_df.iloc[idx].keys():
            image_label = self.patch_df.iloc[idx][self.label_col]
        else:
            image_label = ""

        if self.label_index_col in self.patch_df.iloc[idx].keys():
            image_label_index = self.patch_df.iloc[idx][self.label_index_col]
        else:
            image_label_index = -1

        return (context_img,), image_label, image_label_index
