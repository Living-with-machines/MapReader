#!/usr/bin/env python
from __future__ import annotations

try:
    import rasterio
    from rasterio.warp import (
        Resampling,
        calculate_default_transform,
        reproject,
    )
    from rasterio.windows import get_data_window
except ImportError:
    pass

import distutils.spawn
import logging
import os
import subprocess
from glob import glob

# Set up logging
logger = logging.getLogger(__name__)


def preprocess_all(
    image_paths: list[str] | str, save_preproc_dir: str, **kwds
) -> list[str]:
    """
    Preprocess all images in a list of file paths or a directory using the
    ``preprocess`` function and save them to the specified directory.

    Parameters
    ----------
    image_paths : str or list of str
        Either a string representing the path to a directory containing
        images (wildcards accepted), or a list of file paths representing
        individual images to be preprocessed.
    save_preproc_dir : str
        The path to the directory where preprocessed images will be saved.
    **kwds : keyword arguments
        Additional keyword arguments to be passed to the ``preprocess``
        function.

    Returns
    -------
    saved_paths : list of str
        A list containing the file paths of the preprocessed images that were
        saved.
    """
    if not isinstance(image_paths, list):
        all_paths = glob(image_paths)
    else:
        all_paths = image_paths

    saved_paths = []
    for one_path in all_paths:
        logger.info(f"Preprocessing: {one_path}")
        saved_path = preprocess(one_path, save_preproc_dir, **kwds)
        saved_paths.append(saved_path)
    return saved_paths


def preprocess(
    image_path: str,
    save_preproc_dir: str,
    dst_crs: str | None = "EPSG:3857",
    crop_prefix: str | None = "preproc_",
    reproj_prefix: str | None = "preproc_tmp_",
    resample_prefix: str | None = "preproc_resample_",
    resize_percent: int | None = 40,
    remove_reproj_file: bool | None = True,
) -> str:
    """
    Preprocesses an image file by reprojecting it to a new coordinate
    reference system, cropping (removing white borders) and resampling it to a
    given percentage size.

    Parameters
    ----------
    image_path : str
        The path to the input image file to be preprocessed.
    save_preproc_dir : str
        The directory to save the preprocessed image files.
    dst_crs : str, optional
        The coordinate reference system to reproject the image to, by default
        ``"EPSG:3857"``.
    crop_prefix : str, optional
        The prefix to use for the cropped image file, by default
        ``"preproc_"``.
    reproj_prefix : str, optional
        The prefix to use for the reprojected image file, by default
        ``"preproc_tmp_"``.
    resample_prefix : str, optional
        The prefix to use for the resampled image file, by default
        ``"preproc_resample_"``.
    resize_percent : int, optional
        The percentage to resize the cropped image by, by default ``40``.
    remove_reproj_file : bool, optional
        Whether to remove the reprojected image file after preprocessing, by
        default ``True``.

    Returns
    -------
    str
        The path to the resampled image file if preprocessing was successful,
        otherwise the path to the cropped image file, or ``"False"`` if
        preprocessing failed.
    """

    # make sure that the output dir exists
    if not os.path.isdir(save_preproc_dir):
        os.makedirs(save_preproc_dir)

    # paths to reprojected, cropped and resampled images
    path2save_reproj = os.path.join(
        save_preproc_dir, reproj_prefix + os.path.basename(image_path)
    )
    path2save_crop = os.path.join(
        save_preproc_dir, crop_prefix + os.path.basename(image_path)
    )
    path2save_resample = os.path.join(
        save_preproc_dir, resample_prefix + os.path.basename(image_path)
    )

    if os.path.isfile(path2save_resample):
        logger.warning(f"{path2save_resample} already exists!")
        return "False"

    with rasterio.open(image_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        # --- reproject
        with rasterio.open(path2save_reproj, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

    # --- crop
    # make sure that the file is cropped correctly before proceeding to the
    # next step
    cropped = False
    with rasterio.open(path2save_reproj) as src:
        try:
            window = get_data_window(src.read(1, masked=True))
            transform = rasterio.windows.transform(window, src.transform)
            # window = Window(col_off=13, row_off=3, width=757, height=711)

            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "height": window.height,
                    "width": window.width,
                    "transform": transform,
                }
            )

            with rasterio.open(path2save_crop, "w", **kwargs) as dst:
                dst.write(src.read(window=window))

            cropped = True
        except Exception as e:
            print(e)

    # if not cropped correctly, clean up and exit
    if not cropped:
        os.remove(path2save_reproj)
        return "False"

    # make sure that the file is resampled correctly before proceeding to the
    # next step
    resampled = False
    try:
        gdal_exec = distutils.spawn.find_executable("gdal_translate")
        if not gdal_exec:
            err_msg = "gdal_translate could not be found. Refer to https://gdal.org/"
            raise ImportError(err_msg)

        # Run gdal using subprocess, in our experiments, this was faster than
        # using the library
        gdal_command = f"{gdal_exec} -of GTiff -strict -outsize {resize_percent}% {resize_percent}% {path2save_crop} {path2save_resample}"  # noqa
        # gdalwarp -t_srs EPSG:3857 -crop_to_cutline -overwrite 126517601.27.tif ./preproc/resize_126517601.27.tif # noqa
        subprocess.run(gdal_command, shell=True)
        resampled = True
    except Exception as e:
        print(e)

    if remove_reproj_file:
        os.remove(path2save_reproj)
        if resampled:
            os.remove(path2save_crop)
    if resampled:
        return path2save_resample
    else:
        return path2save_crop

    """
    # Another way of resampling:
    import rasterio
    from rasterio.enums import Resampling

    upscale_factor = 0.1

    with rasterio.open("./101201040.27.tif") as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
    """
