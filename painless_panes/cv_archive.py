"""Some unused computer vision functions that could be useful
"""
from typing import List, Tuple

import cv2  # opencv
import numpy
import skimage.feature

from painless_panes import util


def find_possible_corner_points_in_bounding_box(
    image: numpy.ndarray, bbox: List[int]
) -> numpy.ndarray:
    """Find corners that fall within a bounding box

    :param image: The image
    :type image: numpy.ndarray
    :param bbox: A window bounding box in xyxy format
    :type bbox: List[int]
    :returns: The corner points, as an N x 2 array
    :rtype: numpy.ndarray
    """

    # 1. Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Read the bounding box corners and get its width
    bx0, by0, bx1, by1 = bbox
    bwidth = bx1 - bx0
    bheight = by1 - by0
    blength = min(bwidth, bheight)

    # 3. Corner Harris generates a map where high values are likely to have a corner
    block_size = max(2, blength // 40)
    corner_map = cv2.cornerHarris(image_gray, blockSize=block_size, ksize=3, k=0.04)

    # 4. Set low values to zero, clearing out small local maxima
    corner_map[corner_map < 0.01 * corner_map.max()] = corner_map.min()

    # 5. Identify remaining local maxima, which are likely to be corners
    corners = skimage.feature.peak_local_max(corner_map.T, min_distance=5)

    # 6. Select corners that fall within the bounding box
    corners = select_points_in_bounding_box(corners, bbox)

    return corners


def select_points_in_bounding_box(
    points: numpy.ndarray,
    bbox: numpy.ndarray,
) -> numpy.ndarray:
    """Select points that fall within a bounding box

    :param points: The lines
    :type points: numpy.ndarray
    :param bbox: A bounding box in xyxy format
    :type bbox: List[int]
    :return: The points inside the bounding box
    :rtype: numpy.ndarray
    """
    bx0, by0, bx1, by1 = bbox

    xvals = points[:, 0]
    yvals = points[:, 1]

    in_bbox = (
        numpy.greater_equal(xvals, bx0)
        & numpy.less_equal(xvals, bx1)
        & numpy.greater_equal(yvals, by0)
        & numpy.less_equal(yvals, by1)
    )

    points_in_bbox = points[in_bbox]
    return points_in_bbox


def find_nearby_point(
    point: Tuple[int, int], points: List[Tuple[int, int]], dist_thresh: float = 3
) -> Tuple[int, int]:
    """Given a point, search a list to find a point nearby to it

    Points further than a certain distance will not be considered

    :param point: The point
    :type point: Tuple[int, int]
    :param points: The list of points to be searched
    :type points: List[Tuple[int, int]]
    :param dist_thresh: Points further than this pixel distance are considered,
        defaults to 3
    :type dist_thresh: float, optional
    :return: The nearest point, if there is one within the distance threshold
    :rtype: Tuple[int, int]
    """
    x0, y0 = point
    nearest_point = min(points, key=lambda p: util.line_length([x0, y0, *p]))
    nearest_dist = util.line_length([x0, y0, *nearest_point])

    if nearest_dist > dist_thresh:
        return None

    return nearest_point
