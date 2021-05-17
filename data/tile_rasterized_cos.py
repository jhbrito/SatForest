import os
from math import pi

import numpy
import rasterio
from rasterio import Affine
from rasterio.warp import Resampling, reproject

ZOOM = 14

name = 'COS2015_v1_rasterize'

wd = 'D:/data/COS/COS2015_v1'

try:
    os.mkdir(os.path.join(wd, 'tiles'))
except:
    pass

TRAIN = os.path.join(wd, 'tiles/cos2015-{x}-{y}.tif')

TILE_SIZE = 256

WEB_MERCATOR_WORLDSIZE = 2 * pi * 6378137

WEB_MERCATOR_TILESHIFT = WEB_MERCATOR_WORLDSIZE / 2.0


def tile_bounds(x, y, z):
    """
    Calculate the bounding box of a specific tile.

    Borrowed from:
    https://github.com/geodesign/django-raster/blob/master/raster/tiles/utils.py
    """
    zscale = WEB_MERCATOR_WORLDSIZE / 2 ** z

    xmin = x * zscale - WEB_MERCATOR_TILESHIFT
    xmax = (x + 1) * zscale - WEB_MERCATOR_TILESHIFT
    ymin = WEB_MERCATOR_TILESHIFT - (y + 1) * zscale
    ymax = WEB_MERCATOR_TILESHIFT - y * zscale

    return [xmin, ymin, xmax, ymax]


def tile_index_range(bbox, z, tolerance=0):
    """
    Calculate the index range for a given bounding box and zoomlevel. The bbox
    coordinages are assumed to be in Web Mercator.
    The strict option can be used to force only strict overlaps, based on a
    tolerance.

    Borrowed from:
    https://github.com/geodesign/django-raster/blob/master/raster/tiles/utils.py
    """
    # Calculate tile size for given zoom level.
    zscale = WEB_MERCATOR_WORLDSIZE / 2 ** z

    # Calculate overlaying tile indices.
    result_float = [
        (bbox[0] + WEB_MERCATOR_TILESHIFT) / zscale,
        (WEB_MERCATOR_TILESHIFT - bbox[3]) / zscale,
        (bbox[2] + WEB_MERCATOR_TILESHIFT) / zscale,
        (WEB_MERCATOR_TILESHIFT - bbox[1]) / zscale,
    ]
    # Use integer floor as index. This ensures overlap, since
    # the idex values are counted from the upper left corner.
    result = [None] * 4

    for i in range(4):
        # If the index range is a close call, make sure that only
        # strictly overlapping indices are included.
        if abs(round(result_float[i]) - result_float[i]) < tolerance:
            result[i] = round(result_float[i])
            # For the max range values, reduce so that the edge tile is not
            # included.
            if i > 1:
                result[i] -= 1
        else:
            result[i] = int(result_float[i])

    return result


# Open raster file.
with rasterio.open(os.path.join(wd, '{}.tif'.format(name))) as src:
    scale = src.transform[0]
    # Compute tile range.
    xmin, ymin, xmax, ymax = tile_index_range(src.bounds, ZOOM)
    # Loop through rows and columns within range.
    for x in range(xmin, xmax + 1):
        print('------ X ---------', x)
        for y in range(ymin, ymax + 1):
            # Create warp config.
            extent = tile_bounds(x, y, ZOOM)
            # Compute transform.
            transform = Affine(scale, 0, extent[0], 0, -scale, extent[3])
            # Copy raster profile from source raster.
            dst_creation_args = src.meta.copy()
            # Update with tile level transform and size.
            dst_creation_args.update({
                'transform': transform,
                'width': TILE_SIZE,
                'height': TILE_SIZE,
            })
            # Warp original raster into tile.
            dst_path = TRAIN.format(x=x, y=y)
            with rasterio.open(dst_path, **dst_creation_args, mode='w') as dst:
                # Reproject each band into the destination raster.
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst.transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest,
                )
            # If destination is empty, remove it again.
            delete_file=False
            with rasterio.open(dst_path, **dst_creation_args, mode='r') as dst:
                data = dst.read(1)
                if numpy.sum(data == dst.nodata) == data.size:
                    delete_file=True
            if delete_file:
                os.remove(dst_path)
