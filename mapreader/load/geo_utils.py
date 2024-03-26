#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import rasterio
from geopy.distance import geodesic, great_circle
from pyproj import Transformer


def extractGeoInfo(image_path):
    """Extract geographic information (shape, CRS and coordinates) from GeoTiff files

    Parameters
    ----------
    image_path : str
        Path to image

    Returns
    -------
    list
        shape, CRS, coord
    """
    # read the image using rasterio
    tiff_src = rasterio.open(image_path)
    image_height, image_width = tiff_src.height, tiff_src.width
    image_channels = tiff_src.count
    tiff_shape = (image_height, image_width, image_channels)

    # check coordinates are present
    if isinstance(tiff_src.crs, type(None)):
        raise ValueError(f"No coordinates found in {image_path}")
    else:
        tiff_proj = tiff_src.crs.to_string()
        tiff_coord = tuple(tiff_src.bounds)

    print(f"[INFO] Shape: {tiff_shape}. \n[INFO] CRS: {tiff_proj}.")
    print("[INFO] Coordinates: {:.4f} {:.4f} {:.4f} {:.4f}".format(*tiff_coord))

    return tiff_shape, tiff_proj, tiff_coord


def reproject_geo_info(image_path, target_crs="EPSG:4326", calc_size_in_m=False):
    """Extract geographic information from GeoTiff files and reproject to specified CRS (`target_crs`).

    Parameters
    ----------
    image_path : str
        Path to image
    target_crs : str, optional
        Projection to convert coordinates into, by default "EPSG:4326"
    calc_size_in_m : str or bool, optional
        Method to compute pixel widths and heights, choices between "geodesic" and "great-circle" or "gc", by default "great-circle", by default False

    Returns
    -------
    list
        shape, old CRS, new CRS, reprojected coord, size in meters
    """
    tiff_shape, tiff_proj, tiff_coord = extractGeoInfo(image_path)

    # Coordinate transformation: proj1 ---> proj2
    transformer = Transformer.from_crs(tiff_proj, target_crs, always_xy=True)
    coord = transformer.transform_bounds(*tiff_coord)
    print(f"[INFO] New CRS: {target_crs}")
    print("[INFO] Reprojected coordinates: {:.4f} {:.4f} {:.4f} {:.4f}".format(*coord))

    height, width, _ = tiff_shape

    # Calculate the size of image in meters
    xmin, ymin, xmax, ymax = coord
    if calc_size_in_m:
        if calc_size_in_m in ["geodesic", "gd"]:
            bottom = geodesic((ymin, xmin), (ymin, xmax)).meters
            right = geodesic((ymin, xmax), (ymax, xmax)).meters
            top = geodesic((ymax, xmax), (ymax, xmin)).meters
            left = geodesic((ymax, xmin), (ymin, xmin)).meters

        elif calc_size_in_m in ["gc", "great-circle", "great_circle"]:
            bottom = great_circle((ymin, xmin), (ymin, xmax)).meters
            right = great_circle((ymin, xmax), (ymax, xmax)).meters
            top = great_circle((ymax, xmax), (ymax, xmin)).meters
            left = great_circle((ymax, xmin), (ymin, xmin)).meters

        else:
            raise NotImplementedError(
                f'[ERROR] ``calc_size_in_m`` must be one of "great-circle", "great_circle", "gc", "geodesic" or "gd", not: {calc_size_in_m}'
            )

        size_in_m = (left, bottom, right, top)  # anticlockwise order

        mean_pixel_height = np.mean([right / height, left / height])
        mean_pixel_width = np.mean([bottom / width, top / width])

        print(
            f"[INFO] Size in meters of left/bottom/right/top: {left:.2f}/{bottom:.2f}/{right:.2f}/{top:.2f}"
        )
        print(
            f"Each pixel is ~{mean_pixel_height:.3f} X {mean_pixel_width:.3f} meters (height x width)."
        )  # noqa

    else:
        size_in_m = None

    return tiff_shape, tiff_proj, target_crs, coord, size_in_m
