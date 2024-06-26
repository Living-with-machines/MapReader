from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import random
import string
import warnings
from ast import literal_eval
from itertools import product
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from numpy import array_split
from PIL import Image, ImageOps

from ..load.loader import load_patches

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

_CENTER_LAYOUT = widgets.Layout(
    display="flex", flex_flow="column", align_items="center"
)

# Set up logging
logger = logging.getLogger(__name__)

class Annotator:
    """
    Annotator class for annotating patches with labels.

    Parameters
    ----------
    patch_df : str or pd.DataFrame or None, optional
        Path to a CSV file or a pandas DataFrame containing patch data, by default None
    parent_df : str or pd.DataFrame or None, optional
        Path to a CSV file or a pandas DataFrame containing parent data, by default None
    labels : list, optional
        List of labels for annotation, by default None
    patch_paths : str or None, optional
        Path to patch images, by default None
        Ignored if patch_df is provided.
    parent_paths : str or None, optional
        Path to parent images, by default None
        Ignored if parent_df is provided.
    metadata_path : str or None, optional
        Path to metadata CSV file, by default None
    annotations_dir : str, optional
        Directory to store annotations, by default "./annotations"
    patch_paths_col : str, optional
        Name of the column in which image paths are stored in patch DataFrame, by default "image_path"
    label_col : str, optional
        Name of the column in which labels are stored in patch DataFrame, by default "label"
    show_context : bool, optional
        Whether to show context when loading patches, by default False
    auto_save : bool, optional
        Whether to automatically save annotations, by default True
    delimiter : str, optional
        Delimiter used in CSV files, by default ","
    sortby : str or None, optional
        Name of the column to use to sort the patch DataFrame, by default None.
        Default sort order is ``ascending=True``. Pass ``ascending=False`` keyword argument to sort in descending order.
    ascending : bool, optional
        Whether to sort the DataFrame in ascending order when using the ``sortby`` argument, by default True.
    username : str or None, optional
        Username to use when saving annotations file, by default None.
        If not provided, a random string is generated.
    task_name : str or None, optional
        Name of the annotation task, by default None.
    min_values : dict, optional
        A dictionary consisting of column names (keys) and minimum values as floating point values (values), by default None.
    max_values : dict, optional
        A dictionary consisting of column names (keys) and maximum values as floating point values (values), by default None.
    filter_for : dict, optional
        A dictionary consisting of column names (keys) and values to filter for (values), by default None.
    surrounding : int, optional
        The number of surrounding images to show for context, by default 1.
    max_size : int, optional
        The size in pixels for the longest side to which constrain each patch image, by default 1000.
    resize_to : int or None, optional
        The size in pixels for the longest side to which resize each patch image, by default None.

    Raises
    ------
    FileNotFoundError
        If the provided patch_df or parent_df file path does not exist
    ValueError
        If patch_df or parent_df is not a valid path to a CSV file or a pandas DataFrame
        If patch_df or patch_paths is not provided
        If the DataFrame does not have the required columns
        If sortby is not a string or None
        If labels provided are not in the form of a list
    SyntaxError
        If labels provided are not in the form of a list
    """

    def __init__(
        self,
        patch_df: str | pd.DataFrame | None = None,
        parent_df: str | pd.DataFrame | None = None,
        labels: list = None,
        patch_paths: str | None = None,
        parent_paths: str | None = None,
        metadata_path: str | None = None,
        annotations_dir: str = "./annotations",
        patch_paths_col: str = "image_path",
        label_col: str = "label",
        show_context: bool = False,
        auto_save: bool = True,
        delimiter: str = ",",
        sortby: str | None = None,
        ascending: bool = True,
        username: str | None = None,
        task_name: str | None = None,
        min_values: dict | None = None,
        max_values: dict | None = None,
        filter_for: dict | None = None,
        surrounding: int = 1,
        max_size: int = 1000,
        resize_to: int | None = None,
    ):
        if labels is None:
            labels = []
        if patch_df is not None:
            if isinstance(patch_df, str):
                if os.path.exists(patch_df):
                    patch_df = pd.read_csv(
                        patch_df,
                        index_col=0,
                        sep=delimiter,
                    )
                else:
                    raise FileNotFoundError(f"[ERROR] Could not find {patch_df}.")
            if not isinstance(patch_df, pd.DataFrame):
                raise ValueError(
                    "[ERROR] ``patch_df`` must be a path to a csv or a pandas DataFrame."
                )
            patch_df = self._eval_df(patch_df)  # eval tuples/lists in df

        if parent_df is not None:
            if isinstance(parent_df, str):
                if os.path.exists(parent_df):
                    parent_df = pd.read_csv(
                        parent_df,
                        index_col=0,
                        sep=delimiter,
                    )
                else:
                    raise FileNotFoundError(f"[ERROR] Could not find {parent_df}.")
            if not isinstance(parent_df, pd.DataFrame):
                raise ValueError(
                    "[ERROR] ``parent_df`` must be a path to a csv or a pandas DataFrame."
                )
            parent_df = self._eval_df(parent_df)  # eval tuples/lists in df

        if patch_df is None:
            # If we don't get patch data provided, we'll use the patches and parents to create the dataframes
            if patch_paths:
                parent_paths_df, patch_df = self._load_dataframes(
                    patch_paths=patch_paths,
                    parent_paths=parent_paths,
                    metadata_path=metadata_path,
                    delimiter=delimiter,
                )

                # only take this dataframe if parent_df is None
                if parent_df is None:
                    parent_df = parent_paths_df
            else:
                raise ValueError(
                    "[ERROR] Please specify one of ``patch_df`` or ``patch_paths``."
                )

        # Check for metadata + data
        if not isinstance(patch_df, pd.DataFrame):
            raise ValueError("[ERROR] No patch data available.")
        if not isinstance(parent_df, pd.DataFrame):
            raise ValueError("[ERROR] No metadata (parent data) available.")

        # Check for url column and add to patch dataframe
        if "url" in parent_df.columns:
            patch_df = patch_df.join(parent_df["url"], on="parent_id")

        # Add label column if not present
        if label_col not in patch_df.columns:
            patch_df[label_col] = None

        # Check for image paths column
        if patch_paths_col not in patch_df.columns:
            raise ValueError(
                f"[ERROR] Your DataFrame does not have the image paths column: {patch_paths_col}."
            )

        image_list = json.dumps(
            sorted(patch_df[patch_paths_col].to_list()), sort_keys=True
        )

        # Set up annotations file
        if not username:
            username = "".join(
                [random.choice(string.ascii_letters + string.digits) for n in range(30)]
            )
        if not task_name:
            task_name = "task"
        id = hashlib.md5(image_list.encode("utf-8")).hexdigest()

        annotations_file = task_name.replace(" ", "_") + f"_#{username}#-{id}.csv"
        annotations_file = os.path.join(annotations_dir, annotations_file)

        # Ensure labels are of type list
        if not isinstance(labels, list):
            raise SyntaxError("[ERROR] Labels provided must be as a list")

        # Ensure unique values in list
        labels = sorted(set(labels), key=labels.index)

        # Test for existing patch annotation file
        if os.path.exists(annotations_file):
            logger.info("Loading existing patch annotations.")
            patch_df = self._load_annotations(
                patch_df=patch_df,
                annotations_file=annotations_file,
                labels=labels,
                label_col=label_col,
                delimiter=delimiter,
            )

        ## pixel_bounds = x0, y0, x1, y1
        patch_df["min_x"] = patch_df["pixel_bounds"].apply(lambda x: x[0])
        patch_df["min_y"] = patch_df["pixel_bounds"].apply(lambda x: x[1])
        patch_df["max_x"] = patch_df["pixel_bounds"].apply(lambda x: x[2])
        patch_df["max_y"] = patch_df["pixel_bounds"].apply(lambda x: x[3])

        # Sort by sortby column if provided
        if isinstance(sortby, str):
            if sortby in patch_df.columns:
                self._sortby = sortby
                self._ascending = ascending
            else:
                raise ValueError(
                    f"[ERROR] {sortby} is not a column in the patch DataFrame."
                )
        elif sortby is not None:
            raise ValueError("[ERROR] ``sortby`` must be a string or None.")
        else:
            self._sortby = None
            self._ascending = True

        self.patch_df = patch_df

        self._labels = labels
        self.label_col = label_col
        self.patch_paths_col = patch_paths_col
        self.annotations_file = annotations_file
        self.show_context = show_context
        self.auto_save = auto_save
        self.username = username
        self.task_name = task_name

        # set up for the annotator
        self._min_values = min_values or {}
        self._max_values = max_values or {}
        self._filter_for = filter_for

        # Create annotations_dir
        Path(annotations_dir).mkdir(parents=True, exist_ok=True)

        # Set up standards for context display
        self.surrounding = surrounding
        self.max_size = max_size
        self.resize_to = resize_to

        # set up buttons
        self._buttons = []

        # Set max buttons
        if (len(self._labels) % 2) == 0:
            if len(self._labels) > 4:
                self.buttons_per_row = 4
            else:
                self.buttons_per_row = 2
        else:
            if len(self._labels) == 3:
                self.buttons_per_row = 3
            else:
                self.buttons_per_row = 5

        # Set indices
        self.current_index = -1
        self.previous_index = 0

        # Setup buttons
        self._setup_buttons()

        # Setup box for buttons
        self._setup_box()

        # Setup queue
        self._queue = []

    def __len__(self):
        return len(self.patch_df)

    @staticmethod
    def _load_dataframes(
        patch_paths: str | None = None,
        parent_paths: str | None = None,
        metadata_path: str | None = None,
        delimiter: str = ",",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load parent and patch dataframes by loading images from file paths.

        Parameters
        ----------
        patch_paths : str | None, optional
            Path to the patches, by default None
        parent_paths : str | None, optional
            Path to the parent images, by default None
        metadata_path : str | None, optional
            Path to the parent metadata file, by default None
        delimiter : str, optional
            Delimiter used in CSV files, by default ","

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the parent dataframe and patch dataframe.
        """
        if patch_paths:
            print(f"[INFO] Loading patches from {patch_paths}.")
        if parent_paths:
            print(f"[INFO] Loading parents from {parent_paths}.")

        maps = load_patches(patch_paths=patch_paths, parent_paths=parent_paths)
        # Add pixel stats
        maps.calc_pixel_stats()

        try:
            maps.add_metadata(metadata_path, delimiter=delimiter)
            print(f"[INFO] Adding metadata from {metadata_path}.")
        except ValueError:
            raise FileNotFoundError(
                f"[INFO] Metadata file at {metadata_path} not found. Please specify the correct file path using the ``metadata_path`` argument."
            )

        parent_df, patch_df = maps.convert_images()

        return parent_df, patch_df

    @staticmethod
    def _eval_df(df):
        for col in df.columns:
            try:
                df[col] = df[col].apply(literal_eval)
            except (ValueError, TypeError, SyntaxError):
                pass
        return df

    @staticmethod
    def _load_annotations(
        patch_df: pd.DataFrame,
        annotations_file: str,
        labels: list,
        label_col: str,
        delimiter: str,
    ):
        """Load existing annotations from file.

        Parameters
        ----------
        patch_df : pd.DataFrame
            Current patch dataframe.
        annotations_file : str
            Name of the annotations file
        labels : list
            List of labels for annotation.
        label_col : str
            Name of the column in which labels are stored in annotations file
        delimiter : str
            Delimiter used in CSV files

        """
        existing_annotations = pd.read_csv(annotations_file, index_col=0, sep=delimiter)

        if label_col not in existing_annotations.columns:
            raise ValueError(
                f"[ERROR] Your existing annotations do not have the label column: {label_col}."
            )

        if existing_annotations[label_col].dtype == int:
            # convert label indices (ints) to labels (strings)
            # this is to convert old annotations format to new annotations format
            existing_annotations[label_col] = existing_annotations[label_col].apply(
                lambda x: labels[x]
            )

        patch_df = patch_df.join(
            existing_annotations[label_col], how="left", rsuffix="_existing"
        )
        if f"{label_col}_existing" in patch_df.columns:
            patch_df[label_col].fillna(patch_df[f"{label_col}_existing"], inplace=True)
            patch_df.drop(columns=f"{label_col}_existing", inplace=True)

        return patch_df

    def _setup_buttons(self) -> None:
        """
        Set up buttons for each label to be annotated.
        """
        for label in self._labels:
            btn = widgets.Button(
                description=label,
                button_style="info",
                layout=widgets.Layout(flex="1 1 0%", width="auto"),
            )
            btn.style.button_color = "#9B6F98"

            def on_click(lbl, *_, **__):
                self._add_annotation(lbl)

            btn.on_click(functools.partial(on_click, label))
            self._buttons.append(btn)

    def _setup_box(self) -> None:
        """
        Set up the box which holds all the buttons.
        """
        if len(self._buttons) > self.buttons_per_row:
            self.box = widgets.VBox(
                [
                    widgets.HBox(self._buttons[x : x + self.buttons_per_row])
                    for x in range(0, len(self._buttons), self.buttons_per_row)
                ]
            )
        else:
            self.box = widgets.HBox(self._buttons)

        # back button
        prev_btn = widgets.Button(
            description="« previous", layout=widgets.Layout(flex="1 1 0%", width="auto")
        )
        prev_btn.on_click(self._prev_example)

        # next button
        next_btn = widgets.Button(
            description="next »", layout=widgets.Layout(flex="1 1 0%", width="auto")
        )
        next_btn.on_click(self._next_example)

        self.navbox = widgets.VBox([widgets.HBox([prev_btn, next_btn])])

    def get_queue(
        self, as_type: str | None = "list"
    ) -> list[int] | (pd.Index | pd.Series):
        """
        Gets the indices of rows which are eligible for annotation.

        Parameters
        ----------
        as_type : str, optional
            The format in which to return the indices. Options: "list",
            "index". Default is "list". If any other value is provided, it
            returns a pandas.Series.

        Returns
        -------
        List[int] or pandas.Index or pandas.Series
            Depending on "as_type", returns either a list of indices, a
            pd.Index object, or a pd.Series of legible rows.
        """

        def check_eligibility(row):
            if row.label not in [np.NaN, None]:
                return False

            if self._filter_for is None:
                test = [
                    row[col] >= min_value for col, min_value in self._min_values.items()
                ] + [
                    row[col] <= max_value for col, max_value in self._max_values.items()
                ]
            else:
                test = (
                    [
                        row[col] >= min_value
                        for col, min_value in self._min_values.items()
                    ]
                    + [
                        row[col] <= max_value
                        for col, max_value in self._max_values.items()
                    ]
                    + [
                        row[col] == filter_for
                        for col, filter_for in self._filter_for.items()
                    ]
                )

            if not all(test):
                return False

            return True

        queue_df = self.patch_df.copy(deep=True)
        queue_df = queue_df[queue_df[self.label_col].isna()]  # only unlabelled
        queue_df["eligible"] = queue_df.apply(check_eligibility, axis=1)

        if self._sortby is not None:
            queue_df.sort_values(self._sortby, ascending=self._ascending, inplace=True)
            queue_df = queue_df[queue_df.eligible]
        else:
            queue_df = queue_df[queue_df.eligible].sample(frac=1)  # shuffle

        indices = queue_df.index
        if as_type == "list":
            return list(indices)
        if as_type == "index":
            return indices
        return queue_df

    def get_context(self):
        """
        Provides the surrounding context for the patch to be annotated.

        Returns
        -------
        ipywidgets.VBox
            An IPython VBox widget containing the surrounding patches for
            context.
        """

        def get_path(image_path, dim=True):
            # Resize the image
            im = Image.open(image_path)

            # Dim the image
            if dim in [True, "True"]:
                im_array = np.array(im)
                im_array = 256 - (256 - im_array) * 0.4  # lighten image
                im = Image.fromarray(im_array.astype(np.uint8))
            return im

        def get_empty_square(patch_size: tuple[int, int]):
            """Generates an empty square image.

            Parameters
            ----------
            patch_size : tuple[int, int]
                Patch size in pixels as tuple of `(width, height)`.
            """
            im = Image.new(
                size=patch_size,
                mode="RGB",
                color="white",
            )
            return im

        if self.surrounding > 3:
            display(
                widgets.HTML(
                    """<p style="color:red;"><b>Warning: More than 3 surrounding tiles may crowd the display and not display correctly.</b></p>"""
                )
            )

        ix = self._queue[self.current_index]

        min_x = self.patch_df.at[ix, "min_x"]
        min_y = self.patch_df.at[ix, "min_y"]

        # cannot assume all patches are same size
        try:
            height, width, _ = self.patch_df.at[ix, "shape"]
        except KeyError:
            im_path = self.patch_df.at[ix, self.patch_paths_col]
            im = Image.open(im_path)
            height = im.height
            width = im.width

        current_parent = self.patch_df.at[ix, "parent_id"]
        parent_frame = self.patch_df.query(f"parent_id=='{current_parent}'")

        deltas = list(range(-self.surrounding, self.surrounding + 1))
        y_and_x = list(
            product(
                [min_y + y_delta * height for y_delta in deltas],
                [min_x + x_delta * width for x_delta in deltas],
            )
        )
        queries = [f"min_x == {x} & min_y == {y}" for y, x in y_and_x]
        items = [parent_frame.query(query) for query in queries]

        # derive ids from items
        ids = [x.index[0] if len(x.index) == 1 else None for x in items]
        ids = [x != ix for x in ids]

        # derive images from items
        image_paths = [
            x.at[x.index[0], "image_path"] if len(x.index) == 1 else None for x in items
        ]

        # zip them
        image_list = list(zip(image_paths, ids))

        # split them into rows
        per_row = len(deltas)
        images = [
            [
                get_path(x[0], dim=x[1]) if x[0] else get_empty_square((width, height))
                for x in lst
            ]
            for lst in array_split(image_list, per_row)
        ]

        total_width = (2 * self.surrounding + 1) * width
        total_height = (2 * self.surrounding + 1) * height

        context_image = Image.new("RGB", (total_width, total_height))

        y_offset = 0
        for row in images:
            x_offset = 0
            for image in row:
                context_image.paste(image, (x_offset, y_offset))
                x_offset += width
            y_offset += height

        if self.resize_to is not None:
            context_image = ImageOps.contain(
                context_image, (self.resize_to, self.resize_to)
            )
        # only constrain to max size if not resize_to
        elif max(context_image.size) > self.max_size:
            context_image = ImageOps.contain(
                context_image, (self.max_size, self.max_size)
            )

        return context_image

    def annotate(
        self,
        show_context: bool | None = None,
        sortby: str | None = None,
        ascending: bool | None = None,
        min_values: dict | None = None,
        max_values: dict | None = None,
        surrounding: int | None = None,
        resize_to: int | None = None,
        max_size: int | None = None,
    ) -> None:
        """Annotate at the patch-level of the current patch.
        Renders the annotation interface for the first image.

        Parameters
        ----------
        show_context : bool or None, optional
            Whether or not to display the surrounding context for each image.
            Default is None.
        sortby : str or None, optional
            Name of the column to use to sort the patch DataFrame, by default None.
            Default sort order is ``ascending=True``. Pass ``ascending=False`` keyword argument to sort in descending order.
        ascending : bool, optional
            Whether to sort the DataFrame in ascending order when using the ``sortby`` argument, by default True.
        min_values : dict or None, optional
            Minimum values for each property to filter images for annotation.
            It should be provided as a dictionary consisting of column names
            (keys) and minimum values as floating point values (values).
            Default is None.
        max_values : dict or None, optional
            Maximum values for each property to filter images for annotation.
            It should be provided as a dictionary consisting of column names
            (keys) and minimum values as floating point values (values).
            Default is None
        surrounding : int or None, optional
            The number of surrounding images to show for context. Default: 1.
        max_size : int or None, optional
            The size in pixels for the longest side to which constrain each
            patch image. Default: 100.

        Notes
        -----
        This method is a wrapper for the ``_annotate`` method.
        """
        if sortby is not None:
            self._sortby = sortby
        if ascending is not None:
            self._ascending = ascending

        if min_values is not None:
            self._min_values = min_values
        if max_values is not None:
            self._max_values = max_values

        # re-set up queue using new min/max values
        self._queue = self.get_queue()

        self._annotate(
            show_context=show_context,
            surrounding=surrounding,
            resize_to=resize_to,
            max_size=max_size,
        )

    def _annotate(
        self,
        show_context: bool | None = None,
        surrounding: int | None = None,
        resize_to: int | None = None,
        max_size: int | None = None,
    ):
        """
        Renders the annotation interface for the first image.

        Parameters
        ----------
        show_context : bool or None, optional
            Whether or not to display the surrounding context for each image.
            Default is None.
        surrounding : int or None, optional
            The number of surrounding images to show for context. Default: 1.
        max_size : int or None, optional
            The size in pixels for the longest side to which constrain each
            patch image. Default: 100.

        Returns
        -------
        None
        """

        self.current_index = -1
        for button in self._buttons:
            button.disabled = False

        if show_context is not None:
            self.show_context = show_context
        if surrounding is not None:
            self.surrounding = surrounding
        if resize_to is not None:
            self.resize_to = resize_to
        if max_size is not None:
            self.max_size = max_size

        # re-set up queue
        self._queue = self.get_queue()

        if self._filter_for is not None:
            print(f"[INFO] Filtering for: {self._filter_for}")

        self.out = widgets.Output(layout=_CENTER_LAYOUT)
        display(self.box)
        display(self.navbox)
        display(self.out)

        # self.get_current_index()
        # TODO: Does not pick the correct NEXT...
        self._next_example()

    def _next_example(self, *_) -> tuple[int, int, str]:
        """
        Advances the annotation interface to the next image.

        Returns
        -------
        Tuple[int, int, str]
            Previous index, current index, and path of the current image.
        """
        if self.current_index == len(self._queue):
            self.render_complete()
            return

        self.previous_index = self.current_index
        self.current_index += 1

        ix = self._queue[self.current_index]

        img_path = self.patch_df.at[ix, self.patch_paths_col]

        self.render()
        return self.previous_index, self.current_index, img_path

    def _prev_example(self, *_) -> tuple[int, int, str]:
        """
        Moves the annotation interface to the previous image.

        Returns
        -------
        Tuple[int, int, str]
            Previous index, current index, and path of the current image.
        """
        if self.current_index == len(self._queue):
            self.render_complete()
            return

        if self.current_index > 0:
            self.previous_index = self.current_index
            self.current_index -= 1

        ix = self._queue[self.current_index]

        img_path = self.patch_df.at[ix, self.patch_paths_col]

        self.render()
        return self.previous_index, self.current_index, img_path

    def render(self) -> None:
        """
        Displays the image at the current index in the annotation interface.

        If the current index is greater than or equal to the length of the
        dataframe, the method disables the "next" button and saves the data.

        Returns
        -------
        None
        """
        # Check whether we have reached the end
        if self.current_index >= len(self) - 1:
            self.render_complete()
            return

        ix = self._queue[self.current_index]

        # render buttons
        for button in self._buttons:
            if button.description == "prev":
                # disable previous button when at first example
                button.disabled = self.current_index <= 0
            elif button.description == "next":
                # disable skip button when at last example
                button.disabled = self.current_index >= len(self) - 1
            elif button.description != "submit":
                if self.patch_df.at[ix, self.label_col] == button.description:
                    button.icon = "check"
                else:
                    button.icon = ""

        # display new example
        with self.out:
            clear_output(wait=True)
            image = self.get_patch_image(ix)
            if self.show_context:
                context = self.get_context()
                self._context_image = context
                display(context.convert("RGB"))
            else:
                display(image.convert("RGB"))
            add_ins = []
            if "url" in self.patch_df.loc[ix].keys():
                url = self.patch_df.at[ix, "url"]
                text = f'<p><a href="{url}" target="_blank">Click to see entire map.</a></p>'
                add_ins += [widgets.HTML(text)]

            value = self.current_index + 1 if self.current_index else 1
            description = f"{value} / {len(self._queue)}"
            add_ins += [
                widgets.IntProgress(
                    value=value,
                    min=0,
                    max=len(self._queue),
                    step=1,
                    description=description,
                    orientation="horizontal",
                    barstyle="success",
                )
            ]
            display(
                widgets.VBox(
                    add_ins,
                    layout=_CENTER_LAYOUT,
                )
            )

    def get_patch_image(self, ix) -> Image:
        """
        Returns the image at the given index.

        Parameters
        ----------
        ix : int | str
            The index of the image in the dataframe.

        Returns
        -------
        PIL.Image
            A PIL.Image object of the image at the given index.
        """
        image_path = self.patch_df.at[ix, self.patch_paths_col]
        image = Image.open(image_path)

        if self.resize_to is not None:
            image = ImageOps.contain(image, (self.resize_to, self.resize_to))
        # only constrain to max size if not resize_to
        elif max(image.size) > self.max_size:
            image = ImageOps.contain(image, (self.max_size, self.max_size))

        return image

    def _add_annotation(self, annotation: str) -> None:
        """
        Adds the provided annotation to the current image.

        Parameters
        ----------
        annotation : str
            The label to add to the current image.

        Returns
        -------
        None
        """
        # ix = self.iloc[self.current_index].name
        ix = self._queue[self.current_index]
        self.patch_df.at[ix, self.label_col] = annotation
        if self.auto_save:
            self._auto_save()
        self._next_example()

    def _auto_save(self):
        """
        Automatically saves the annotations made so far.

        Returns
        -------
        None
        """
        self.get_labelled_data(sort=True).to_csv(self.annotations_file)

    def get_labelled_data(
        self,
        sort: bool = True,
        index_labels: bool = False,
        include_paths: bool = True,
    ) -> pd.DataFrame:
        """
        Returns the annotations made so far.

        Parameters
        ----------
        sort : bool, optional
            Whether to sort the dataframe by the order of the images in the
            input data, by default True
        index_labels : bool, optional
            Whether to return the label's index number (in the labels list
            provided in setting up the instance) or the human-readable label
            for each row, by default False
        include_paths : bool, optional
            Whether to return a column containing the full path to the
            annotated image or not, by default True

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the labelled images and their associated
            label index.
        """
        filtered_df = self.patch_df[self.patch_df[self.label_col].notna()].copy(
            deep=True
        )

        # force image_id to be index (incase of integer index)
        # TODO: Force all indices to be integers so this is not needed
        if ("image_id" in filtered_df.columns) and (
            filtered_df.index.name != "image_id"
        ):
            filtered_df.set_index("image_id", drop=True, inplace=True)

        if sort:
            filtered_df.sort_values(by=["parent_id", "min_x", "min_y"], inplace=True)

        if index_labels:
            filtered_df[self.label_col] = filtered_df[self.label_col].apply(
                lambda x: self._labels.index(x)
            )

        return filtered_df[
            [self.label_col, self.patch_paths_col, "parent_id", "pixel_bounds"]
        ]

    @property
    def filtered(self) -> pd.DataFrame:
        _filter = ~self.patch_df[self.label_col].isna()
        return self.patch_df[_filter]

    def render_complete(self):
        """
        Renders the completion message once all images have been annotated.

        Returns
        -------
        None
        """
        clear_output()
        display(
            widgets.HTML("<p><b>All annotations done with current settings.</b></p>")
        )
        if self.auto_save:
            self._auto_save()
        for button in self._buttons:
            button.disabled = True
