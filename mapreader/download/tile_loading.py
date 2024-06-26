# Code adapted from https://github.com/baurls/TileStitcher.
from __future__ import annotations

import logging
import os
import urllib.request

from joblib import Parallel, delayed
from tqdm import tqdm

from .data_structures import GridBoundingBox, GridIndex

# Set up logging
logger = logging.getLogger(__name__)

# Default values (constants)
DEFAULT_TEMP_FOLDER = "_tile_cache/"  # must end with a "/"
DEFAULT_IMG_DOWNLOAD_FORMAT = "png"


class TileDownloader:
    def __init__(
        self,
        tile_servers: list = None,
        img_format: str | None = None,
        show_progress: bool = False,
    ):
        """
        TileDownloader object.

        Parameters
        ----------
        tile_servers : list
            Download urls for tileservers.
        img_format : Union[str, None], optional
            Image format used when saving tiles.
            If None, ``png`` will be used.
            By default None.
        show_progress : bool, optional
            Whether or not to show progress bar, by default False.
        """
        self.tile_servers = tile_servers
        self.temp_folder = DEFAULT_TEMP_FOLDER
        self.img_format = (
            img_format if img_format is not None else DEFAULT_IMG_DOWNLOAD_FORMAT
        )
        self._vis_blocks = 35
        self.show_progress = show_progress

    def generate_tile_name(self, index: GridIndex):
        """Generates tile file names from GridIndex.

        Parameters
        ----------
        index : GridIndex

        Returns
        -------
        str
            Tile file name
        """
        return "{}z={}_x={}_y={}.{}".format(
            self.temp_folder, index.z, index.x, index.y, self.img_format
        )

    def generate_tile_url(self, index: GridIndex, subserver_index: int):
        """Generates tile download urls from GridIndex.

        Parameters
        ----------
        index : GridIndex
        subserver_index : int
            Index no. of subserver to use for download

        Returns
        -------
        str
            Tile download url
        """
        return self.tile_servers[subserver_index].format(
            x=index.x, y=index.y, z=index.z
        )

    def download_tiles(
        self, grid_bb: GridBoundingBox, download_in_parallel: bool = True
    ):
        """Downloads tiles contained within GridBoundingBox.

        Parameters
        ----------
        grid_bb : GridBoundingBox
            GridBoundingBox containing tiles to download
        download_in_parallel : bool, optional
            Whether or not to download tiles in parallel, by default True

        Returns
        -------
        xxxx
        """
        os.makedirs(self.temp_folder, exist_ok=True)
        if not download_in_parallel:
            logger.info(
                f"Downloading {grid_bb.covered_cells} tiles sequentially to disk .."
            )
            return self._download_tiles_sequentially(grid_bb)

        # download in parallel
        logger.info(
            f"Downloading {grid_bb.covered_cells} tiles to disk (in parallel).."
        )
        delayed_downloads = [
            delayed(self._download_tile_in_parallel)(
                GridIndex(x, y, grid_bb.z),
                i,
                len(self.tile_servers),
                grid_bb.covered_cells,
            )
            for i, (x, y) in enumerate(
                (x, y) for x in grid_bb.x_range for y in grid_bb.y_range
            )
        ]

        self._update_progressbar(0.0)
        parallel_pool = Parallel(n_jobs=-1)
        parallel_pool(delayed_downloads)
        self._update_progressbar(1.0)

    def _download_tile(self, tile_cell: GridIndex):
        """Downloads a tile.

        Parameters
        ----------
        tile_cell : GridIndex
            GridIndex of tile to download
        """
        url = self.generate_tile_url(tile_cell, 0)
        file_name = self.generate_tile_name(tile_cell)
        _trigger_download(url, file_name)

    def _download_tile_in_parallel(
        self, tile_cell: GridIndex, i: int, no_server: int, total_nbr: int
    ):
        """Downloads a tile in parallel.

        Parameters
        ----------
        tile_cell : GridIndex
            GridIndex of tile to download
        i : int
            Tile index
        no_server : _type_
            Number of tile servers to download from
        total_nbr : int
            Total number of tiles being downloaded
        """
        url = self.generate_tile_url(tile_cell, i % no_server)
        file_name = self.generate_tile_name(tile_cell)
        _trigger_download(url, file_name)
        share = (i + 1) / total_nbr
        self._update_progressbar(share)

    def _update_progressbar(self, share: float):
        """Updates progress bar.

        Parameters
        ----------
        share : float
            Share to show as completed/done
        """
        if not self.show_progress:
            return
        visible = int(share * self._vis_blocks)
        invisible = self._vis_blocks - visible

        print(
            "\r",
            f"{share * 100:3.0f}%" + "|" + "■" * visible + "□" * invisible + "|",
            end="",
        )

    def _download_tiles_sequentially(self, grid_bb: GridBoundingBox):
        """Downloads tiles sequentially.

        Parameters
        ----------
        grid_bb : GridBoundingBox
            GridBoundingBox containing tiles to download
        """
        for x, y in tqdm([(x, y) for x in grid_bb.x_range for y in grid_bb.y_range]):
            # download tile x,y,z
            tile_cell = GridIndex(x, y, grid_bb.z)
            self._download_tile(tile_cell)


def _trigger_download(url: str, file_path: str):
    """Triggers download of tiles.

    Parameters
    ----------
    url : str
        The url of the tile to download
    file_path : str
        The path to where the file will be saved
    """
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
    headers = {"User-Agent": user_agent}

    try:
        request = urllib.request.Request(url, None, headers)
        response = urllib.request.urlopen(request)
        data = response.read()

        with open(file_path, "wb") as f:
            f.write(data)

    except:
        print(f"[WARNING] {url} not found.")
