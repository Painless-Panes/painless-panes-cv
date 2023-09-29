"""Some unused computer vision functions that could be useful
"""
from typing import List, Tuple

import cv2  # opencv
import numpy
import skimage.feature

from painless_panes import util


def zigzag_sort_corners(
    corners: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Sort corners of a rectangle in zig-zag order for perspective transformation

    :param corners: The corner points of a rectangle
    :type corners: List[Tuple[float, float]]
    :returns: The same corners, in zig-zag sort order
    :rtype: List[Tuple[float, float]]
    """
    corners = numpy.array(corners).tolist()
    y_sorted_corners = sorted(corners, key=lambda c: c[1])
    return sorted(y_sorted_corners[:2]) + sorted(y_sorted_corners[2:])


def align_image_perspective_on_rectangle(
    image: numpy.ndarray,
    rect_corners: List[Tuple[float, float]],
    points_list: List[List[Tuple[float, float]]] = (),
) -> Tuple[numpy.ndarray, List[List[Tuple[float, float]]]]:
    """Align the image perspective on a rectangular object, making it a perfect
    rectangle

    Source: https://stackoverflow.com/q/38285229

    :param image: The image
    :type image: numpy.ndarray
    :param rect_corners: The corners of the rectangle on the image
    :type rect_corners: List[Tuple[float, float]]
    :param points_list: A list of lists of points to be transformed with the image,
        defaults to ()
    :type points_list: List[List[Tuple[float, float]]]
    :return: The image, and the transformed points list
    :rtype: Tuple[numpy.ndarray, List[List[Tuple[float, float]]]]
    """
    # 0. Find the original width, height and cener of the image
    height0, width0, _ = image.shape
    cx0 = width0 / 2
    cy0 = height0 / 2

    # 1. Sort the original corner points and store them in an array
    c_rect0 = numpy.array(zigzag_sort_corners(rect_corners), dtype=numpy.float32)

    # 2. Calculate the apparent aspect ratio of the rectangle
    w_rect0 = max(
        util.line_length([*c_rect0[0], *c_rect0[1]]),
        util.line_length([*c_rect0[2], *c_rect0[3]]),
    )
    h_rect0 = max(
        util.line_length([*c_rect0[0], *c_rect0[2]]),
        util.line_length([*c_rect0[1], *c_rect0[3]]),
    )
    apparent_ratio = w_rect0 / h_rect0

    # 3. Calculate the focal disrance
    m = numpy.ones((4, 3))
    m[:, :2] = c_rect0
    m0x3 = numpy.cross(m[0], m[3])
    m1x3 = numpy.cross(m[1], m[3])
    m2x3 = numpy.cross(m[2], m[3])
    k1 = numpy.dot(m0x3, m[2]) / numpy.dot(m1x3, m[2])
    k2 = numpy.dot(m0x3, m[1]) / numpy.dot(m2x3, m[1])

    n1 = k1 * m[1] - m[0]
    n2 = k2 * m[2] - m[0]

    f = numpy.sqrt(
        numpy.abs(
            (1.0 / (n1[2] * n2[2]))
            * (
                (
                    n1[0] * n2[0]
                    - (n1[0] * n2[2] + n1[2] * n2[0]) * cx0
                    + n1[2] * n2[2] * cx0 * cx0
                )
                + (
                    n1[1] * n2[1]
                    - (n1[1] * n2[2] + n1[2] * n2[1]) * cy0
                    + n1[2] * n2[2] * cy0 * cy0
                )
            )
        )
    )

    # 4. Calculate the real aspect ratio
    A = numpy.array([[f, 0, cx0], [0, f, cy0], [0, 0, 1]])
    At_inv = numpy.linalg.inv(A.T)
    A_inv = numpy.linalg.inv(A)

    true_ratio = numpy.sqrt(
        numpy.dot(numpy.dot(numpy.dot(n1, At_inv), A_inv), n1)
        / numpy.dot(numpy.dot(numpy.dot(n2, At_inv), A_inv), n2)
    )

    # 5. Define transformed rectangle corner based on its aspect ratio
    if true_ratio < apparent_ratio:
        w_rect = w_rect0
        h_rect = w_rect / true_ratio
    else:
        h_rect = h_rect0
        w_rect = true_ratio * h_rect

    c_rect = numpy.array(
        [[0, 0], [w_rect, 0], [0, h_rect], [w_rect, h_rect]], dtype=numpy.float32
    )

    # 6. Get the perspective transformation matrix for the rectangle
    trans_rect = cv2.getPerspectiveTransform(c_rect0, c_rect)

    # 7. Identify how this transforms the image corners
    c_image0 = numpy.array(
        [[0, 0], [width0, 0], [0, height0], [width0, height0]], dtype=numpy.float32
    )
    (c_image,) = cv2.perspectiveTransform(numpy.array([c_image0]), trans_rect)

    # 8. Figure out how it needs to be cropped to fit a rectangle
    crop_x0 = numpy.max(c_image[0::2, 0])
    crop_x1 = numpy.min(c_image[1::2, 0])
    crop_y0 = numpy.max(c_image[:2, 1])
    crop_y1 = numpy.min(c_image[2:, 1])
    c_crop = numpy.array(
        [
            [crop_x0, crop_y0],
            [crop_x1, crop_y0],
            [crop_x0, crop_y1],
            [crop_x1, crop_y1],
        ],
        dtype=numpy.float32,
    )

    # 9. Inverse transform to find the crop corner points on the original image
    trans_rect_inv = numpy.linalg.inv(trans_rect)
    (c_crop0,) = cv2.perspectiveTransform(numpy.array([c_crop]), trans_rect_inv)

    # 10. Now, remove negative values from the cropped image corners and get a new
    # perspective transform that will include the full height and width
    c_crop -= numpy.min(c_crop, axis=0)
    trans_final = cv2.getPerspectiveTransform(c_crop0, c_crop)

    # 11. Get the width and height of the cropped image
    width_final = int(crop_x1 - crop_x0)
    height_final = int(crop_y1 - crop_y0)

    # 12. Transform the image
    image_final = cv2.warpPerspective(image, trans_final, (width_final, height_final))

    # 13. Transform the points list
    points_list_final = []
    for points in points_list:
        (points_final,) = cv2.perspectiveTransform(numpy.array([points]), trans_final)
        points_list_final.append(points_final)

    return image_final, points_list_final


def scale_perspective_to_square(
    image: numpy.ndarray,
    square_corners: List[Tuple[float, float]],
    points_list: List[List[Tuple[float, float]]] = (),
) -> Tuple[numpy.ndarray, List[List[Tuple[float, float]]]]:
    """Align the image perspective on a rectangular object, making it a perfect
    rectangle

    Source: https://stackoverflow.com/q/38285229

    :param image: The image
    :type image: numpy.ndarray
    :param square_corners: The corners of the rectangle on the image
    :type square_corners: List[Tuple[float, float]]
    :param points_list: A list of lists of points to be transformed with the image,
        defaults to ()
    :type points_list: List[List[Tuple[float, float]]]
    :return: The image, and the transformed points list
    :rtype: Tuple[numpy.ndarray, List[List[Tuple[float, float]]]]
    """
    square_corners = numpy.array(
        zigzag_sort_corners(square_corners), dtype=numpy.float32
    )

    # Form vectors for each edge (top, bottom, left, right)
    t_vec = square_corners[1] - square_corners[0]
    b_vec = square_corners[3] - square_corners[2]
    l_vec = square_corners[2] - square_corners[0]
    r_vec = square_corners[3] - square_corners[1]

    # Get average vertical (top, bottom) and horizontal (left, right) vectors
    vx, vy = numpy.average([t_vec, b_vec], axis=0)
    hx, hy = numpy.average([l_vec, r_vec], axis=0)

    # Calculate a horizontal scale factor
    fx = numpy.sqrt((vy**2 - hy**2) / (hx**2 - vx**2))

    # Get the image corner points
    height0, width0, _ = image.shape
    c_image0 = numpy.array(
        [[0, 0], [width0, 0], [0, height0], [width0, height0]], dtype=numpy.float32
    )

    # Apply the scale factor to get the corner points of a scaled image
    height = height0
    width = width0 * fx
    c_image = numpy.array(
        [[1, 1], [width, 1], [1, height], [width, height]], dtype=numpy.float32
    )

    # Get the horizontal scaling transformation
    print("fx:", fx)
    trans = cv2.getPerspectiveTransform(c_image0, c_image)
    print("trans:", trans)

    # Transform the image
    image_out = cv2.warpPerspective(image, trans, (int(width), int(height)))

    # 13. Transform the points list
    points_list_out = []
    for points in points_list:
        (points_out,) = cv2.perspectiveTransform(numpy.array([points]), trans)
        points_list_out.append(points_out)

    return image_out, points_list_out


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
