#!/usr/bin/env python
from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

# Import parhugin
try:
    from parhugin import multiFunc

    parhugin_installed = True
except ImportError:
    print(
        "[WARNING] parhugin (https://github.com/kasra-hosseini/parhugin) is not installed, continue without it."  # noqa
    )
    parhugin_installed = False

# PIL.Image ignore decompression bomb warnings
Image.MAX_IMAGE_PIXELS = None

class PatchDataset(Dataset):
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

        if isinstance(patch_df, pd.DataFrame):
            self.patch_df = patch_df

        elif isinstance(patch_df, str):
            if os.path.isfile(patch_df):
                print(f'[INFO] Reading "{patch_df}".')
                patch_df = pd.read_csv(patch_df, sep=delimiter)
                self.patch_df = patch_df
            else:
                raise ValueError(f'[ERROR] "{patch_df}" cannot be found.')

        else:
            raise ValueError(
                "[ERROR] Please pass ``patch_df`` as a string (path to csv file) or pd.DataFrame."
            )

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
                    print(
                        f"[INFO] Label index column ({label_index_col}) not in dataframe. Creating column."
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

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.patch_df)

    def __getitem__(self, idx: int | torch.Tensor) -> tuple[torch.Tensor, str, int]:
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

        return img, image_label, image_label_index

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
        ``PIL.Image.Image`` object is returned.
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


# --- Dataset that returns an image, its context and its label
class PatchContextDataset(PatchDataset):
    def __init__(
        self,
        patch_df: pd.DataFrame | str,
        transform1: str,
        transform2: str,
        delimiter: str = ",",
        patch_paths_col: str | None = "image_path",
        label_col: str | None = None,
        label_index_col: str | None = None,
        image_mode: str | None = "RGB",
        context_save_path: str | None = "./maps/maps_context",
        create_context: bool | None = False,
        parent_path: str | None = "./maps",
        x_offset: float | None = 1.0,
        y_offset: float | None = 1.0,
        slice_method: str | None = "scale",
    ):
        """
        A PyTorch Dataset class for loading contextual information about image
        patches from a DataFrame.

        Parameters
        ----------
        patch_df : pandas.DataFrame or str
            DataFrame or path to csv file containing the paths to image patches and their labels.
        transform1 : str
            Torchvision transform to be applied to input images.
            Either "train" or "val".
        transform2 : str
            Torchvision transform to be applied to target images.
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
        context_save_path : str, optional
            The path to save context maps to. Default is "./maps/maps_context".
        create_context : bool, optional
            Whether or not to create context maps. Default is False.
        parent_path : str, optional
            The path to the directory containing parent images. Default is
            "./maps".
        x_offset : float, optional
            The size of the horizontal offset around objects, as a fraction of
            the image width. Default is 1.0.
        y_offset : float, optional
            The size of the vertical offset around objects, as a fraction of
            the image height. Default is 1.0.
        slice_method : str, optional
            The method used to slice images. Either "scale" or "absolute".
            Default is "scale".

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
        x_offset : float
            The size of the horizontal offset around objects, as a fraction of
            the image width.
        y_offset : float
            The size of the vertical offset around objects, as a fraction of
            the image height.
        slice_method : str
            The method used to slice images.
        create_context : bool
            Whether or not to create context maps.
        context_save_path : str
            The path to save context maps to.
        unique_labels : list or str
            The unique labels in ``label_col``, or "NS" if ``label_col`` not in
            ``patch_df``.

        Methods
        ----------
        __getitem__(idx)
            Retrieves the patch image, the context image and the label at the
            given index in the dataset.
        save_parents()
            Saves parent images.
        save_parents_idx(idx)
            Saves parent image at index ``idx``.
        return_orig_image(idx)
            Return the original image associated with the given index.
        """

        if isinstance(patch_df, pd.DataFrame):
            self.patch_df = patch_df

        elif isinstance(patch_df, str):
            if os.path.isfile(patch_df):
                print(f'[INFO] Reading "{patch_df}".')
                patch_df = pd.read_csv(patch_df, sep=delimiter)
                self.patch_df = patch_df
            else:
                raise ValueError(f'[ERROR] "{patch_df}" cannot be found.')

        else:
            raise ValueError(
                "[ERROR] Please pass ``patch_df`` as a string (path to csv file) or pd.DataFrame."
            )

        self.label_col = label_col
        self.label_index_col = label_index_col
        self.image_mode = image_mode
        self.patch_paths_col = patch_paths_col
        self.parent_path = parent_path
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.slice_method = slice_method
        self.create_context = create_context
        self.context_save_path = os.path.abspath(
            context_save_path
        )  # we need this either way I think?

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
                    print(
                        f"[INFO] Label index column ({label_index_col}) not in dataframe. Creating column."
                    )
                    self.patch_df[self.label_index_col] = self.patch_df[
                        self.label_col
                    ].apply(self._get_label_index)
                else:
                    raise ValueError(
                        f"[ERROR] Label index column ({label_index_col}) not in dataframe."
                    )

        if isinstance(transform1, str):
            if transform1 in ["train", "val", "test"]:
                self.transform1 = self._default_transform(transform1)
            else:
                raise ValueError(
                    '[ERROR] ``transform`` can only be "train", "val" or "test" or, a transform.'
                )
        else:
            self.transform1 = transform1

        if isinstance(transform2, str):
            if transform2 in ["train", "val", "test"]:
                self.transform2 = self._default_transform(transform2)
            else:
                raise ValueError(
                    '[ERROR] ``transform`` can only be "train", "val" or "test" or, a transform.'
                )
        else:
            self.transform2 = transform2

    def save_parents(
        self,
        processors: int | None = 10,
        sleep_time: float | None = 0.001,
        use_parhugin: bool | None = True,
        parent_delimiter: str | None = "#",
        loc_delimiter: str | None = "-",
        overwrite: bool | None = False,
    ) -> None:
        """
        Save parent patches for all patches in the patch_df.

        Parameters
        ----------
        processors : int, optional
            The number of required processors for the job, by default 10.
        sleep_time : float, optional
            The time to wait between jobs, by default 0.001.
        use_parhugin : bool, optional
            Flag indicating whether to use Parhugin to parallelize the job, by
            default True.
        parent_delimiter : str, optional
            The delimiter used to separate parent IDs in the patch filename, by
            default "#".
        loc_delimiter : str, optional
            The delimiter used to separate patch pixel bounds in the patch
            filename, by default "-".
        overwrite : bool, optional
            Flag indicating whether to overwrite existing parent files, by
            default False.

        Returns
        -------
        None

        Notes
        -----
        Parhugin is a Python package for parallelizing computations across
        multiple CPU cores. The method uses Parhugin to parallelize the
        computation of saving parent patches to disk. When Parhugin is
        installed and ``use_parhugin`` is set to True, the method parallelizes
        the calling of the ``save_parents_idx`` method and its corresponding
        arguments. If Parhugin is not installed or ``use_parhugin`` is set to
        False, the method executes the loop over patch indices sequentially
        instead.
        """
        if parhugin_installed and use_parhugin:
            myproc = multiFunc(processors=processors, sleep_time=sleep_time)
            list_jobs = []
            for idx in range(len(self.patch_df)):
                list_jobs.append(
                    [
                        self.save_parents_idx,
                        (idx, parent_delimiter, loc_delimiter, overwrite),
                    ]
                )

            print(f"Total number of jobs: {len(list_jobs)}")
            # and then adding them to myproc
            myproc.add_list_jobs(list_jobs)
            myproc.run_jobs()
        else:
            for idx in range(len(self.patch_df)):
                self.save_parents_idx(idx)

    def save_parents_idx(
        self,
        idx: int,
        parent_delimiter: str | None = "#",
        loc_delimiter: str | None = "-",
        overwrite: bool | None = False,
        return_image: bool | None = False,
    ) -> None:
        """
        Save the parents of a specific patch to the specified location.

        Parameters
        ----------
            idx : int
                Index of the patch in the dataset.
            parent_delimiter : str, optional
                Delimiter to split the parent names in the file path. Default
                is "#".
            loc_delimiter : str, optional
                Delimiter to split the location of the patch in the file path.
                Default is "-".
            overwrite : bool, optional
                Whether to overwrite the existing parent files. Default is
                False.

        Raises
        ------
        ValueError
            If the patch is not found in the dataset.

        Returns
        -------
        None
        """
        img_path = self.patch_df.iloc[idx][self.patch_paths_col]

        if os.path.exists(img_path):
            img = Image.open(img_path).convert(self.image_mode)
        else:
            raise ValueError(
                f'[ERROR] "{img_path} cannot be found.\n\n\
Please check the image exists, your file paths are correct and that ``.patch_paths_col`` is set to the correct column.'
            )

        if not return_image:
            os.makedirs(self.context_save_path, exist_ok=True)

            path2save_context = os.path.join(
                self.context_save_path, os.path.basename(img_path)
            )

            if os.path.isfile(path2save_context) and (not overwrite):
                return

        if self.slice_method in ["scale"]:
            # size: (width, height)
            tar_y_offset = int(img.size[1] * self.y_offset)
            tar_x_offset = int(img.size[0] * self.x_offset)
        else:
            tar_y_offset = self.y_offset
            tar_x_offset = self.x_offset

        par_name = os.path.basename(img_path).split(parent_delimiter)[1]
        split_path = os.path.basename(img_path).split(loc_delimiter)
        min_x, min_y, max_x, max_y = (
            int(split_path[1]),
            int(split_path[2]),
            int(split_path[3]),
            int(split_path[4]),
        )

        if self.parent_path in ["dynamic"]:
            parent_path2read = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(img_path))),
                par_name,
            )
        else:
            parent_path2read = os.path.join(os.path.abspath(self.parent_path), par_name)

        par_img = Image.open(parent_path2read).convert(self.image_mode)

        min_y_par = max(0, min_y - tar_y_offset)
        min_x_par = max(0, min_x - tar_x_offset)
        max_x_par = min(max_x + tar_x_offset, np.shape(par_img)[1])
        max_y_par = min(max_y + tar_y_offset, np.shape(par_img)[0])

        pad_activate = False
        top_pad = left_pad = right_pad = bottom_pad = 0
        if (min_y - tar_y_offset) < 0:
            top_pad = abs(min_y - tar_y_offset)
            pad_activate = True
        if (min_x - tar_x_offset) < 0:
            left_pad = abs(min_x - tar_x_offset)
            pad_activate = True
        if (max_x + tar_x_offset) > np.shape(par_img)[1]:
            right_pad = max_x + tar_x_offset - np.shape(par_img)[1]
            pad_activate = True
        if (max_y + tar_y_offset) > np.shape(par_img)[0]:
            bottom_pad = max_y + tar_y_offset - np.shape(par_img)[0]
            pad_activate = True

        # par_img = par_img[min_y_par:max_y_par, min_x_par:max_x_par]
        par_img = par_img.crop((min_x_par, min_y_par, max_x_par, max_y_par))

        if pad_activate:
            padding = (left_pad, top_pad, right_pad, bottom_pad)
            par_img = ImageOps.expand(par_img, padding)

        if return_image:
            return par_img
        elif not os.path.isfile(path2save_context):
            par_img.save(path2save_context)

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
        plt.imshow(transforms.ToPILImage()(self.__getitem__(idx)[0]))
        plt.title("Patch", size=18)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(transforms.ToPILImage()(self.__getitem__(idx)[1]))
        plt.title("Context", size=18)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.show()

    def __getitem__(
        self, idx: int | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, str, int]:
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

        img_path = self.patch_df.iloc[idx][self.patch_paths_col]

        if os.path.exists(img_path):
            img = Image.open(img_path).convert(self.image_mode)
        else:
            raise ValueError(
                f'[ERROR] "{img_path} cannot be found.\n\n\
Please check the image exists, your file paths are correct and that ``.patch_paths_col`` is set to the correct column.'
            )

        if self.create_context:
            context_img = self.save_parents_idx(idx, return_image=True)
        else:
            context_img = Image.open(
                os.path.join(self.context_save_path, os.path.basename(img_path))
            ).convert(self.image_mode)

        img = self.transform1(img)
        context_img = self.transform2(context_img)

        if self.label_col in self.patch_df.iloc[idx].keys():
            image_label = self.patch_df.iloc[idx][self.label_col]
        else:
            image_label = ""

        if self.label_index_col in self.patch_df.iloc[idx].keys():
            image_label_index = self.patch_df.iloc[idx][self.label_index_col]
        else:
            image_label_index = -1

        return img, context_img, image_label, image_label_index
