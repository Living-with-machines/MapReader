from __future__ import annotations

import functools
import hashlib
import json
import os
import random
import string
import warnings
from itertools import product
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display
from numpy import array_split
from PIL import Image, ImageOps

from ..load.loader import load_patches

warnings.filterwarnings("ignore", category=UserWarning)

MAX_SIZE = 100

_CENTER_LAYOUT = widgets.Layout(
    display="flex", flex_flow="column", align_items="center"
)


class Annotator(pd.DataFrame):
    def __init__(
        self,
        patch_df: str | pd.DataFrame | None = None,
        parent_df: str | pd.DataFrame | None = None,
        labels: list = None,
        patch_paths: str | None = None,
        parent_paths: str | None = None,
        metadata_path: str | None = "./maps/metadata.csv",
        annotations_dir: str = "./annotations",
        patch_paths_col: str = "image_path",
        label_col: str = "label",
        show_context: bool = False,
        auto_save: bool = True,
        delimiter: str = ",",
        **kwargs,
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
                        converters={
                            "shape": eval,
                            "pixel_bounds": eval,
                            "coordinates": eval,
                        },
                    )
                else:
                    raise FileNotFoundError(f"[ERROR] Could not find {patch_df}.")
            if not isinstance(patch_df, pd.DataFrame):
                raise ValueError(
                    "[ERROR] ``patch_df`` must be a path to a csv or a pandas DataFrame."
                )

        if parent_df is not None:
            if isinstance(parent_df, str):
                if os.path.exists(parent_df):
                    parent_df = pd.read_csv(
                        parent_df,
                        index_col=0,
                        sep=delimiter,
                        converters={"shape": eval, "coordinates": eval},
                    )
                else:
                    raise FileNotFoundError(f"[ERROR] Could not find {parent_df}.")
            if not isinstance(parent_df, pd.DataFrame):
                raise ValueError(
                    "[ERROR] ``parent_df`` must be a path to a csv or a pandas DataFrame."
                )

        if patch_df is None:
            # If we don't get patch data provided, we'll use the patches and parents to create the dataframes
            if patch_paths:
                parent_paths_df, patch_df = self._load_frames(
                    patch_paths=patch_paths,
                    parent_paths=parent_paths,
                    metadata_path=metadata_path,
                    delimiter=delimiter,
                    label_col=label_col,
                )

                # only take this dataframe if parent_df is None
                if parent_df is None:
                    parent_df = parent_paths_df
            else:
                raise ValueError(
                    "[ERROR] Please specify one of ``patch_df`` or ``patch_paths``."
                )

        # Check for metadata + data
        if not len(patch_df):
            raise ValueError("[ERROR] No patch data available.")
        if not len(parent_df):
            raise ValueError("[ERROR] No metadata (parent data) available.")

        # Check for url column and add to patch dataframe
        if "url" in parent_df.columns:
            patch_df = patch_df.join(parent_df["url"], on="parent_id")
        else:
            raise ValueError(
                "[ERROR] Metadata (parent data) should contain a 'url' column."
            )

        # Add label column if not present
        if label_col not in patch_df.columns:
            patch_df[label_col] = None
        patch_df["changed"] = False

        # Check for image paths column
        if patch_paths_col not in patch_df.columns:
            raise ValueError(
                f"[ERROR] Your DataFrame does not have the image paths column: {patch_paths_col}."
            )

        # Sort by sortby column if provided
        if kwargs.get("sortby"):
            patch_df = patch_df.sort_values(kwargs["sortby"])

        image_list = json.dumps(
            sorted(patch_df[patch_paths_col].to_list()), sort_keys=True
        )

        # Set up annotations file
        username = kwargs.get(
            "username",
            "".join(
                [random.choice(string.ascii_letters + string.digits) for n in range(30)]
            ),
        )
        task_name = kwargs.get("task_name", "task")
        id = hashlib.md5(image_list.encode("utf-8")).hexdigest()

        annotations_file = task_name.replace(" ", "_") + f"_#{username}#-{id}.csv"
        annotations_file = os.path.join(annotations_dir, annotations_file)

        # Ensure labels are of type list
        if not isinstance(labels, list):
            raise SyntaxError("[ERROR] Labels provided must be as a list")

        # Ensure unique values in list
        labels = list(set(labels))

        # Test for existing file
        if os.path.exists(annotations_file):
            print(f"[INFO] Loading existing annotations for {kwargs['username']}.")
            existing_annotations = pd.read_csv(
                annotations_file, index_col=0, sep=delimiter
            )

            if label_col not in existing_annotations.columns:
                raise ValueError(
                    f"[ERROR] Your existing annotations do not have the label column: {label_col}."
                )

            print(existing_annotations[label_col].dtype)

            if existing_annotations[label_col].dtype == int:
                # convert label indices (ints) to labels (strings)
                # this is to convert old annotations format to new annotations format
                existing_annotations[label_col] = existing_annotations[label_col].apply(
                    lambda x: labels[x]
                )

            patch_df = patch_df.join(
                existing_annotations, how="left", lsuffix="_x", rsuffix="_y"
            )
            patch_df[label_col] = patch_df["label_y"].fillna(patch_df[f"{label_col}_x"])
            patch_df = patch_df.drop(
                columns=[
                    f"{label_col}_x",
                    f"{label_col}_y",
                ]
            )
            patch_df["changed"] = patch_df[label_col].apply(
                lambda x: True if x else False
            )

            patch_df[patch_paths_col] = patch_df[f"{patch_paths_col}_x"]
            patch_df = patch_df.drop(
                columns=[
                    f"{patch_paths_col}_x",
                    f"{patch_paths_col}_y",
                ]
            )

        # initiate as a DataFrame
        super().__init__(patch_df)

        ## pixel_bounds = x0, y0, x1, y1
        self["min_x"] = self.pixel_bounds.apply(lambda x: x[0])
        self["min_y"] = self.pixel_bounds.apply(lambda x: x[1])
        self["max_x"] = self.pixel_bounds.apply(lambda x: x[2])
        self["max_y"] = self.pixel_bounds.apply(lambda x: x[3])

        self._labels = labels
        self.label_col = label_col
        self.patch_paths_col = patch_paths_col
        self.annotations_file = annotations_file
        self.show_context = show_context
        self.auto_save = auto_save
        self.username = username
        self.task_name = task_name

        # set up for the annotator
        self.buttons_per_row = kwargs.get("buttons_per_row", None)
        self.stop_at_last_example = kwargs.get("stop_at_last_example", True)
        self._min_values = kwargs.get("min_values", {})
        self._max_values = kwargs.get("max_values", {})  # pixel_bounds = x0, y0, x1, y1

        self.patch_width, self.patch_height = self.get_patch_size()

        # Create annotations_dir
        Path(annotations_dir).mkdir(parents=True, exist_ok=True)

        # Set up standards for context display
        self.surrounding = 1
        self.margin = 1
        self.max_size = MAX_SIZE

        # set up buttons
        self._buttons = []

        # Set max buttons
        if not self.buttons_per_row:
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
        self._queue = self.get_queue()

    @staticmethod
    def _load_frames(
        patch_paths: str | None = None,
        parent_paths: str | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads patches and parents data from given paths and returns the
        corresponding dataframes.

        Parameters
        ----------
        **kwargs :
            Needs to contain "patches" and "parents"

            Needs to contain "metadata" and "metadata_delimiter"

            Needs to contain "label_col"

            Needs to contain "scramble_frame"

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Parents and patches dataframes.
        """
        if patch_paths:
            print(f"[INFO] Loading patches from {patch_paths}.")
        if parent_paths:
            print(f"[INFO] Loading parents from {parent_paths}.")

        maps = load_patches(patch_paths=patch_paths, parent_paths=parent_paths)
        # Add pixel stats
        maps.calc_pixel_stats()

        try:
            maps.add_metadata(kwargs["metadata_path"], delimiter=kwargs["delimiter"])
            print(f"[INFO] Adding metadata from {kwargs['metadata_path']}.")
        except FileNotFoundError:
            print("[INFO] Metadata file not found. Continuing without metadata.")

        parent_df, patch_df = maps.convert_images()

        return parent_df, patch_df

    def get_patch_size(self):
        """
        Calculate and return the width and height of the patches based on the
        first patch of the DataFrame, assuming the same shape of patches
        across the frame.

        Returns
        -------
        Tuple[int, int]
            Width and height of the patches.
        """
        patch_width = (
            self.sort_values("min_x").max_x[0] - self.sort_values("min_x").min_x[0]
        )
        patch_height = (
            self.sort_values("min_y").max_y[0] - self.sort_values("min_y").min_y[0]
        )

        return patch_width, patch_height

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
        Gets the indices of rows which are legible for annotation.

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

        def check_legibility(row):
            if row.label is not None:
                return False

            test = [
                row[col] >= min_value for col, min_value in self._min_values.items()
            ] + [row[col] <= max_value for col, max_value in self._max_values.items()]

            if not all(test):
                return False

            return True

        test = self.copy()
        test["eligible"] = test.apply(check_legibility, axis=1)
        test = test[
            ["eligible"] + [col for col in test.columns if not col == "eligible"]
        ]

        indices = test[test.eligible].index
        if as_type == "list":
            return list(indices)
        if as_type == "index":
            return indices
        return test[test.eligible]

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
            if max(im.size) > self.max_size:
                print(f"[DEBUG] Resizing {image_path} to {self.max_size}")
                im = ImageOps.contain(im, (self.max_size, self.max_size))
                print(f"[DEBUG] --> {im.size}")

            # Dim the image
            if dim is True or dim == "True":
                alpha = Image.new("L", im.size, 60)
                im.putalpha(alpha)

            # Save temp image
            image_path = ".temp.png"
            im.save(image_path)

            # Read as bytes
            with open(image_path, "rb") as f:
                im = f.read()

            Path(image_path).unlink()

            layout = widgets.Layout(margin=f"{self.margin}px")
            return widgets.Image(value=im, layout=layout)

        def get_empty_square():
            im = Image.new(
                size=(self.patch_width, self.patch_height),
                mode="RGB",
                color="white",
            )
            image_path = ".temp.png"
            im.save(image_path)

            with open(image_path, "rb") as f:
                im = f.read()

            return widgets.Image(value=im)

        if self.surrounding > 3:
            display(
                widgets.HTML(
                    """<p style="color:red;"><b>Warning: More than 3 surrounding tiles may crowd the display and not display correctly.</b></p>"""
                )
            )

        ix = self._queue[self.current_index]

        x = self.at[ix, "min_x"]
        y = self.at[ix, "min_y"]
        current_parent = self.at[ix, "parent_id"]

        parent_frame = self.query(f"parent_id=='{current_parent}'")

        deltas = list(range(-self.surrounding, self.surrounding + 1))
        y_and_x = list(
            product(
                [y + y_delta * self.patch_height for y_delta in deltas],
                [x + x_delta * self.patch_width for x_delta in deltas],
            )
        )
        queries = [f"min_x == {x} & min_y == {y}" for y, x in y_and_x]
        items = [parent_frame.query(query) for query in queries]

        # derive ids from items
        ids = [x.index[0] if len(x.index) == 1 else None for x in items]
        ids = [x != ix for x in ids]

        # derive images from items
        images = [
            x.at[x.index[0], "image_path"] if len(x.index) == 1 else None for x in items
        ]

        # zip them
        images = list(zip(images, ids))

        # split them into rows
        per_row = len(deltas)
        image_widgets = [
            [get_path(x[0], dim=x[1]) if x[0] else get_empty_square() for x in lst]
            for lst in array_split(images, per_row)
        ]

        h_boxes = [widgets.HBox(x) for x in image_widgets]

        return widgets.VBox(h_boxes, layout=_CENTER_LAYOUT)

    def annotate(
        self,
        show_context: bool = False,
        min_values: dict | None = None,
        max_values: dict | None = None,
        surrounding: int | None = 1,
        margin: int | None = 0,
        max_size: int | None = MAX_SIZE,
    ) -> None:
        """
        Renders the annotation interface for the first image.

        Parameters
        ----------
        show_context : bool, optional
            Whether or not to display the surrounding context for each image.
            Default: None.
        min_values : dict, optional
            Minimum values for each property to filter images for annotation.
            It should be provided as a dictionary consisting of column names
            (keys) and minimum values as floating point values (values).
            Default: {}.
        max_values : dict, optional
            Maximum values for each property to filter images for annotation.
            It should be provided as a dictionary consisting of column names
            (keys) and minimum values as floating point values (values).
            Default: {}.
        surrounding : int, optional
            The number of surrounding images to show for context. Default: 1.
        margin : int, optional
            The margin to use for the context images. Default: 0.
        max_size : int, optional
            The size in pixels for the longest side to which constrain each
            patch image. Default: 100.

        Returns
        -------
        None
        """
        if max_values is None:
            max_values = {}
        if min_values is None:
            min_values = {}
        self.current_index = -1
        for button in self._buttons:
            button.disabled = False

        self._min_values = min_values
        self._max_values = max_values
        self.surrounding = surrounding
        self.margin = margin
        self.max_size = max_size

        # re-set up queue
        self._queue = self.get_queue()

        self.out = widgets.Output()
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
        if not len(self._queue):
            self.render_complete()
            return

        if isinstance(self.current_index, type(None)) or self.current_index == -1:
            self.current_index = 0
        else:
            current_index = self.current_index + 1

            try:
                self._queue[current_index]
                self.previous_index = self.current_index
                self.current_index = current_index
            except IndexError:
                pass

        ix = self._queue[self.current_index]

        img_path = self.at[ix, self.patch_paths_col]

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
        if not len(self._queue):
            self.render_complete()
            return

        current_index = self.current_index - 1

        if current_index < 0:
            current_index = 0

        try:
            self._queue[current_index]
            self.previous_index = current_index - 1
            self.current_index = current_index
        except IndexError:
            pass

        ix = self._queue[self.current_index]

        img_path = self.at[ix, self.patch_paths_col]

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
            if self.stop_at_last_example:
                self.render_complete()
            else:
                self._prev_example()
            return

        # ix = self.iloc[self.current_index].name
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
                if self.at[ix, self.label_col] == button.description:
                    button.icon = "check"
                else:
                    button.icon = ""

        # display new example
        with self.out:
            clear_output(wait=True)
            image = self.get_patch_image(ix)
            if self.show_context:
                context = self.get_context()
                display(context)
            else:
                display(image)
            add_ins = []
            if self.at[ix, "url"]:
                url = self.at[ix, "url"]
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

    def get_patch_image(self, ix: int) -> widgets.Image:
        """
        Returns the image at the given index.

        Parameters
        ----------
        ix : int
            The index of the image in the dataframe.

        Returns
        -------
        ipywidgets.Image
            A widget displaying the image at the given index.
        """
        image_path = self.at[ix, self.patch_paths_col]
        with open(image_path, "rb") as f:
            image = f.read()

        return widgets.Image(value=image)

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
        self.at[ix, self.label_col] = annotation
        self.at[ix, "changed"] = True
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
        if index_labels:
            col1 = self.filtered[self.label_col].apply(lambda x: self._labels.index(x))
        else:
            col1 = self.filtered[self.label_col]

        if include_paths:
            col2 = self.filtered[self.patch_paths_col]
            df = pd.DataFrame(
                {self.patch_paths_col: col2, self.label_col: col1},
                index=pd.Index(col1.index, name="image_id"),
            )
        else:
            df = pd.DataFrame(col1, index=pd.Index(col1.index, name="image_id"))
        if not sort:
            return df

        df["sort_value"] = df.index.to_list()
        df["sort_value"] = df["sort_value"].apply(
            lambda x: f"{x.split('#')[1]}-{x.split('#')[0]}"
        )
        return df.sort_values("sort_value").drop(columns=["sort_value"])

    @property
    def filtered(self) -> pd.DataFrame:
        _filter = ~self[self.label_col].isna()
        return self[_filter]

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