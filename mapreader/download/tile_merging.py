# Code adapted from https://github.com/baurls/TileStitcher.
from __future__ import annotations

import logging
import os

from PIL import Image
from tqdm import tqdm

from .data_structures import GridBoundingBox, GridIndex
from .tile_loading import DEFAULT_IMG_DOWNLOAD_FORMAT, DEFAULT_TEMP_FOLDER

# Set up logging
logger = logging.getLogger(__name__)

# Default values (constants)
DEFAULT_OUT_FOLDER = "./"
DEFAULT_IMG_STORE_FORMAT = ("png", "PNG")


class TileMerger:
    def __init__(
        self,
        output_folder: str | None = None,
        img_input_format: str | None = None,
        img_output_format: str | None = None,
        show_progress=False,
    ):
        """TileMerger object.

        Parameters
        ----------
        output_folder : Union[str, None], optional
            Path to save map images.
            If None, "./" will be used (i.e. map images will be saved in current directory).
            By default None
        img_input_format : Union[str, None], optional
            Image format of tiles.
            If None, ``png`` will be assumed.
            By default None.
        img_output_format : Union[str, None], optional
            Image format used when saving map images.
            If None, ``png`` will be used.
            By default None.
        show_progress : bool, optional
            Whether or not to show progress bar, by default False.
        """
        self.img_input_format = (
            img_input_format
            if img_input_format is not None
            else DEFAULT_IMG_DOWNLOAD_FORMAT
        )
        self.img_output_format = (
            img_output_format
            if img_output_format is not None
            else DEFAULT_IMG_STORE_FORMAT
        )
        self.output_folder = (
            output_folder if output_folder is not None else DEFAULT_OUT_FOLDER
        )
        self.temp_folder = DEFAULT_TEMP_FOLDER
        self.show_progress = show_progress

    @staticmethod
    def _get_output_name(grid_bb: GridBoundingBox) -> str:
        """Generates map name based on GridBoundingBox

        Parameters
        ----------
        grid_bb : GridBoundingBox

        Returns
        -------
        str
            Map name
        """
        return "map_z{z}_x{x1}-{x2}_y{y1}-{y2}".format(
            z=grid_bb.z,
            x1=grid_bb.lower_corner.x,
            x2=grid_bb.upper_corner.x,
            y1=grid_bb.lower_corner.y,
            y2=grid_bb.upper_corner.y,
        )

    def _generate_tile_name(self, index: GridIndex):
        """Generates tile file names from GridIndex

        Parameters
        ----------
        index : GridIndex

        Returns
        -------
        str
            Tile file name
        """
        return "{}z={}_x={}_y={}.{}".format(
            self.temp_folder, index.z, index.x, index.y, self.img_input_format
        )

    def _load_image_to_grid_cell(self, cell_index: GridIndex):
        """Loads image from GridIndex

        Parameters
        ----------
        cell_index : GridIndex

        Returns
        -------
        PIL.Image
        """
        filename = self._generate_tile_name(cell_index)
        image = Image.open(filename)
        return image

    def _load_tile_size(self, grid_bb: GridBoundingBox):
        """Finds size of tiles based on GridBoundingBox

        Parameters
        ----------
        grid_bb : GridBoundingBox

        Returns
        -------
        tuple
            Size of tile
        """
        try:
            start_image = self._load_image_to_grid_cell(grid_bb.lower_corner)
        except FileNotFoundError:
            logger.warning("Image has missing tiles in bottom left corner.")
            try:
                start_image = self._load_image_to_grid_cell(grid_bb.upper_corner)
            except FileNotFoundError:
                logger.warning("Image has missing tiles in upper right corner.")
                raise FileNotFoundError(
                    "[ERROR] Image is missing tiles for both lower left and upper right corners."
                )

        img_size = start_image.size
        assert (
            img_size[0] == img_size[1]
        ), f"Tiles must be quadratic. This tile, however, is rectangular: {img_size}"
        tile_size = img_size[0]
        return tile_size

    def merge(
        self,
        grid_bb: GridBoundingBox,
        file_name: str | None = None,
        overwrite: bool = False,
    ) -> str | bool:
        """Merges cells contained within GridBoundingBox.

        Parameters
        ----------
        grid_bb : GridBoundingBox
            GridBoundingBox containing tiles to merge
        file_name : Union[str, None], optional
            Name to use when saving map
            If None, default name will be used, by default None
        overwrite : bool, optional
            Whether or not to overwrite existing files, by default False
        Returns
        -------
        str or bool
            out path if file has successfully downloaded, False if not.
        """
        os.makedirs(self.output_folder, exist_ok=True)

        try:
            tile_size = self._load_tile_size(grid_bb)
        except FileNotFoundError:
            return False  # unsuccessful

        merged_image = Image.new(
            "RGBA", (len(grid_bb.x_range) * tile_size, len(grid_bb.y_range) * tile_size)
        )

        logger.info("Merging tiles to one file.")

        for i, x, j, y in tqdm(
            [
                (i, x, j, y)
                for i, x in enumerate(grid_bb.x_range)
                for j, y in enumerate(grid_bb.y_range)
            ],
            disable=not self.show_progress,
        ):
            current_cell = GridIndex(x, y, grid_bb.z)
            try:
                current_tile = Image.open(self._generate_tile_name(current_cell))
                merged_image.paste(current_tile, (tile_size * i, tile_size * j))
            except FileNotFoundError:
                logger.info(f"Cannot find tile with grid_index = {current_cell}")

        logger.info("Writing file..")

        if file_name is None:
            file_name = self._get_output_name(grid_bb)

        out_path = f"{self.output_folder}{file_name}.{self.img_output_format[0]}"
        if not overwrite:
            i = 1
            while os.path.exists(out_path):
                out_path = (
                    f"{self.output_folder}{file_name}_{i}.{self.img_output_format[0]}"
                )
                i += 1
        merged_image.save(out_path, self.img_output_format[1])
        success = out_path if os.path.exists(out_path) else False
        if success is False:
            logger.warning(f"Merge unsuccessful! '{out_path}' not saved.")
        else:
            logger.info(f"Merge successful! The image has been stored at '{out_path}'")

        return success
