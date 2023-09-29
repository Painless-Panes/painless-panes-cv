import itertools
from typing import List, Tuple

import cv2  # opencv
import numpy

from painless_panes import model, util

# Module constants
BLUE = (255, 0, 0)  # annotation color
GREEN = (0, 255, 0)  # annotation color
RED = (0, 0, 255)  # annotation color
FONT = cv2.FONT_HERSHEY_SIMPLEX
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def measure_window(
    image: numpy.ndarray, annotate: bool = False
) -> Tuple[Tuple[int, int], str]:
    """Measure a window from an image with an Aruco marker

    :param image: The image
    :type image: numpy.ndarray
    :param annotate: Annotate the image file as a side-effect?, defaults to False
    :type annotate: bool, optional
    :returns: The width and height, in inches, along with an error message, which is
        empty unless the measurement fails for some reason
    :rtype: Tuple[int, int, str]
    """
    # Find the aruco marker
    aruco_corners = find_aruco_corners(image)
    print("Aruco corners:", aruco_corners)

    # Find the model detections (windows)
    detections = model.get_detections(image)
    print("All detections:", detections)

    # Select the central detection
    detection = select_central_detection(image, detections)
    print("Central detection:", detection)

    window_bbox = detection["bounding_box"]
    window_corners = find_window_corners(image, window_bbox)
    print("Window corners:", window_corners)

    # If no marker was found, return early with an error message
    if aruco_corners is None:
        return None, None, "No marker detected!"

    if detection is None:
        return None, None, "No window detected!"

    if window_corners is None:
        return None, None, "Could not find window corners."

    # Get the perspective transformation
    image, (aruco_corners, window_corners) = align_image_perspective_on_rectangle(
        image, window_corners, points_list=(aruco_corners, window_corners)
    )

    image, (aruco_corners, window_corners) = scale_perspective_to_square(
        image, aruco_corners, points_list=(aruco_corners, window_corners)
    )

    # If we have a marker and a detection, make the measurement
    bbox = detection["bounding_box"]
    window_width, window_height = measure_rectangle_dimensions_from_aruco(
        image, bbox, aruco_corners, annotate=True
    )

    print("Measured dimensions:", window_width, window_height)

    return window_width, window_height, image, ""


# Finders
def find_aruco_corners(image: numpy.ndarray, annotate: bool = False) -> numpy.ndarray:
    """Find one aruco marker in an image

    Returns the first one found, if there is one

    :param image: The image
    :type image: numpy.ndarray
    :param annotate: Annotate the image file as a side-effect?, defaults to False
    :type annotate: bool, optional
    :return: The corner positions of the Aruco marker, in pixels:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :rtype: numpy.ndarray
    """
    corners_list, _, _ = cv2.aruco.detectMarkers(
        image, ARUCO_DICT, parameters=ARUCO_PARAMS
    )
    corners_list = numpy.array(corners_list, dtype=numpy.float32)

    if not corners_list.size:
        return None

    corners = numpy.reshape(corners_list[0], (4, 2))

    # If requested, annotate the image as a side-effect
    if annotate:
        annotate_line(image, corners)

    return corners


def find_window_corners(
    image: numpy.ndarray, bbox: List[int], annotate: bool = False
) -> numpy.ndarray:
    """Find the corners of a window in an image

    Returns the first one found, if there is one

    :param image: The image
    :type image: numpy.ndarray
    :param bbox: A window bounding box in xyxy format
    :type bbox: List[int]
    :param annotate: Annotate the image file as a side-effect?, defaults to False
    :type annotate: bool, optional
    :return: The corner positions of the Aruco marker, in pixels:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :rtype: numpy.ndarray
    """
    # 0. Define relative distances
    bx0, by0, bx1, by1 = bbox
    bwidth = bx1 - bx0
    bheight = by1 - by0
    blength = min(bwidth, bheight)

    overhang_thresh = max(2, blength // 80)

    #     a. Define inside margin, assuming vertical edges ar near the bounding box edge
    mx0 = bx0 + 0.1 * bwidth
    mx1 = bx1 - 0.1 * bwidth
    my0 = by0 + 0.5 * bheight
    my1 = by1 - 0.5 * bheight

    #     a. Define outside padding
    px0 = bx0 - 0.1 * bwidth
    px1 = bx1 + 0.1 * bwidth
    py0 = by0 - 0.1 * bheight
    py1 = by1 + 0.1 * bheight

    # 1. Find lines of potential edges and points of potential corners inside the
    # bounding box
    edge_lines = find_lines_in_bounding_box(image, bbox)
    # corner_points = find_possible_corner_points_in_bounding_box(image, bbox)

    # 2. Find possible corner points for the top-left (tr), bottom-right (br), top-right
    # (tr), and bottom-left (bl) corners, tracking the indices of their associated edges
    edge_idxs_dct = {
        # Vertical edges are sorted from the inner margin out, so inner edges come first
        "l": argsort_vertical_lines_in_interval(edge_lines, start=mx0, end=bx0),
        "r": argsort_vertical_lines_in_interval(edge_lines, start=mx1, end=bx1),
        # Horizontal edges are sorted from the outside in, so outer edges come first
        "t": argsort_horizontal_lines_in_interval(edge_lines, start=by0, end=my0),
        "b": argsort_horizontal_lines_in_interval(edge_lines, start=by1, end=my1),
    }

    def _find_corner_points_with_edge_indices(corner_key):
        edge1_key, edge2_key = corner_key
        edge1_idxs = edge_idxs_dct[edge1_key]
        edge2_idxs = edge_idxs_dct[edge2_key]

        pwes = []
        for idx1, idx2 in itertools.product(edge1_idxs, edge2_idxs):
            line1 = edge_lines[idx1]
            line2 = edge_lines[idx2]

            # a. Find an intersection between these edges without overhang
            point = util.line_pair_ray_intersection_no_overhang(
                line1, line2, dist_thresh=overhang_thresh
            )
            if point is not None:
                # b. Check that the point is within the padded bounding box
                xint, yint = point
                if px0 < xint < px1 and py0 < yint < py1:
                    # c. If both checks pass, add it to the list of corner points
                    pwe = (point, idx1, idx2)
                    pwes.append(pwe)

        return pwes

    tl_pwes = _find_corner_points_with_edge_indices("tl")
    br_pwes = _find_corner_points_with_edge_indices("br")
    tr_pwes = _find_corner_points_with_edge_indices("tr")
    bl_pwes = _find_corner_points_with_edge_indices("bl")

    # 3. Find the first complete set of corners with matching edge indices
    window_corners = None
    corner_pwes_iter = itertools.product(tl_pwes, br_pwes, tr_pwes, bl_pwes)
    for tl_pwe, br_pwe, tr_pwe, bl_pwe in corner_pwes_iter:
        tl_point, tidx1, lidx1 = tl_pwe
        br_point, bidx1, ridx1 = br_pwe
        tr_point, tidx2, ridx2 = tr_pwe
        bl_point, bidx2, lidx2 = bl_pwe
        edges_match = (
            tidx1 == tidx2 and lidx1 == lidx2 and bidx1 == bidx2 and ridx1 == ridx2
        )
        if edges_match:
            window_corners = [tl_point, tr_point, br_point, bl_point]
            break

    if annotate:
        annotate_line(image, window_corners, color=RED)

    return window_corners


def find_lines_in_bounding_box(image: numpy.ndarray, bbox: List[int]) -> numpy.ndarray:
    """Find lines that fall within a bounding box

    :param image: The image
    :type image: numpy.ndarray
    :param bbox: A window bounding box in xyxy format
    :type bbox: List[int]
    :returns: The lines, as an N x 4 array
    :rtype: numpy.ndarray
    """
    image = image.copy()

    # 1. Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Read the bounding box corners and get its width
    bx0, by0, bx1, by1 = bbox
    bwidth = bx1 - bx0
    bheight = by1 - by0
    blength = min(bwidth, bheight)

    # 3. Detect straight lines
    fld = cv2.ximgproc.createFastLineDetector(
        length_threshold=blength // 5,
        do_merge=True,
    )
    lines = fld.detect(image_gray)

    # 4. Sort the line points for convenience
    lines = sort_lines_points(lines, axis="y")

    # 5. Select lines that fall within the bounding box
    lines = select_lines_in_bounding_box(lines, bbox)

    # 6. Merge the lines by ray
    lines_groups = util.equivalence_partition(lines, util.lines_fall_on_common_ray)
    lines = numpy.array(list(map(util.merge_lines, lines_groups)))

    return lines


# Sorters and Selectors
def select_central_detection(
    image: numpy.ndarray, detections: List[dict], annotate: bool = False
) -> dict:
    """Select the detection closest to the center

    :param image: The image
    :type image: numpy.ndarray
    :param detections: The list of detections, in the form:
        [
            {
                "class_name": <string>,   # The object class name
                "confidence": <float>,    # The % confidence in the detection
                "bounding_box": <tuple>,  # The bounding box in xyxy format
            },
            ...
        ]
    :type detections: List[dict]
    :param annotate: Annotate the image file as a side-effect?, defaults to False
    :type annotate: bool, optional
    :returns: The central detection from the list
    :rtype: dict
    """
    if not detections:
        return None

    height, width, _ = image.shape

    center_point = numpy.intp([width // 2, height // 2])
    min_dist = numpy.inf
    central_detection = None
    for detection in detections:
        x0, y0, x1, y1 = detection["bounding_box"]
        corners = [[x0, y0], [x1, y1]]
        # Calculate the sum of the corner distances (relative to center)
        dist = sum(numpy.linalg.norm(numpy.subtract(p, center_point)) for p in corners)
        if dist < min_dist:
            central_detection = detection
            min_dist = dist

    if annotate:
        class_name = central_detection["class_name"]
        conf = central_detection["confidence"]
        bbox = central_detection["bounding_box"]
        text = f"{class_name} {conf:.2f}"
        annotate_rectangle(image, bbox, text=text)

    return central_detection


def select_lines_in_bounding_box(
    lines: numpy.ndarray,
    bbox: numpy.ndarray,
) -> numpy.ndarray:
    """Select lines that fall within a bounding box

    :param lines: The lines
    :type lines: numpy.ndarray
    :param bbox: A bounding box in xyxy format
    :type bbox: List[int]
    :return: The lines inside the bounding box
    :rtype: numpy.ndarray
    """
    bx0, by0, bx1, by1 = bbox

    xvals = numpy.squeeze(lines)[:, 0::2]
    yvals = numpy.squeeze(lines)[:, 1::2]

    in_bbox = (
        numpy.all(numpy.greater_equal(xvals, bx0), 1)
        & numpy.all(numpy.less_equal(xvals, bx1), 1)
        & numpy.all(numpy.greater_equal(yvals, by0), 1)
        & numpy.all(numpy.less_equal(yvals, by1), 1)
    )

    lines_in_bbox = lines[in_bbox]
    return lines_in_bbox


def sort_lines_points(
    lines: numpy.ndarray,
    axis: str = "y",
) -> numpy.ndarray:
    """Sort the two points for each line along a particular axis

    :param lines: The lines
    :type lines: numpy.ndarray
    :param axis: The axis to sort along, defaults to "y"
    :type axis: str, optional
    :return: The lines, with points sorted for each
    :rtype: numpy.ndarray
    """
    vals1 = lines[:, 0, 1] if axis == "y" else lines[:, 0, 0]
    vals2 = lines[:, 0, 3] if axis == "y" else lines[:, 0, 2]

    # Identify rows where the values need to be swapped
    swap = numpy.greater(vals1, vals2)

    # Copy the array
    lines_out = lines.copy()

    # Do the swap
    lines_out[swap, :, :2] = lines[swap, :, 2:]
    lines_out[swap, :, 2:] = lines[swap, :, :2]
    return lines_out


def argsort_vertical_lines_in_interval(
    lines: numpy.ndarray,
    start: float,
    end: float,
    vert_angle: float = 30,
) -> numpy.ndarray:
    """Select indices for vertical lines within an x-interval and sort them

    If `end` < `start`, the indices will be reverse-sorted

    :param lines: The lines
    :type lines: numpy.ndarray
    :param start: Starting point of the sort/select interval, defaults to None
    :type start: float, optional
    :param end: Ending point of the sort/select interval, defaults to None
    :type end: float, optional
    :param vert_angle: Cutoff for identifying vertical lines, in degrees, defaults to 30
    :type vert_angle: float, optional
    :return: The sort indices
    :rtype: numpy.ndarray
    """
    min_x_vals = numpy.min(numpy.squeeze(lines)[:, 0::2], axis=1)
    max_x_vals = numpy.max(numpy.squeeze(lines)[:, 0::2], axis=1)

    # Identify whether the range is reversed
    reverse = end < start
    if reverse:
        start, end = end, start

    # 1. Sort the indices by min/max x-value
    if not reverse:
        sort_idxs = list(numpy.argsort(min_x_vals, axis=0))
    else:
        sort_idxs = list(reversed(numpy.argsort(max_x_vals, axis=0)))

    # 2. Filter out indices that are out of bounds
    sort_idxs = [i for i in sort_idxs if min_x_vals[i] > start]
    sort_idxs = [i for i in sort_idxs if max_x_vals[i] < end]

    # 3. Filter out indices for non-vertical lines
    vidxs, _ = argpartition_lines_by_orientation(lines, vert_angle=vert_angle)
    vsort_idxs = [i for i in sort_idxs if i in vidxs]

    return vsort_idxs


def argsort_horizontal_lines_in_interval(
    lines: numpy.ndarray,
    start: float,
    end: float,
    vert_angle: float = 30,
) -> numpy.ndarray:
    """Select indices for horizontal lines within an x-interval and sort them

    If `end` < `start`, the indices will be reverse-sorted

    :param lines: The lines
    :type lines: numpy.ndarray
    :param start: Starting point of the sort/select interval, defaults to None
    :type start: float, optional
    :param end: Ending point of the sort/select interval, defaults to None
    :type end: float, optional
    :param vert_angle: Cutoff for identifying vertical lines, in degrees, defaults to 30
    :type vert_angle: float, optional
    :return: The sort indices
    :rtype: numpy.ndarray
    """
    min_y_vals = numpy.min(numpy.squeeze(lines)[:, 1::2], axis=1)
    max_y_vals = numpy.max(numpy.squeeze(lines)[:, 1::2], axis=1)

    # Identify whether the range is reversed
    reverse = end < start
    if reverse:
        start, end = end, start

    # 1. Sort the indices by min/max x-value
    if not reverse:
        sort_idxs = list(numpy.argsort(min_y_vals, axis=0))
    else:
        sort_idxs = list(reversed(numpy.argsort(max_y_vals, axis=0)))

    # 2. Filter out indices that are out of bounds
    sort_idxs = [i for i in sort_idxs if min_y_vals[i] > start]
    sort_idxs = [i for i in sort_idxs if max_y_vals[i] < end]

    # 3. Filter out indices for non-horizontal lines
    _, hidxs = argpartition_lines_by_orientation(lines, vert_angle=vert_angle)
    hsort_idxs = [i for i in sort_idxs if i in hidxs]

    return hsort_idxs


def argpartition_lines_by_orientation(
    lines: numpy.ndarray, vert_angle: float = 30
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Partition a set of line indices by vertical and horizontal orientation

    Partitions into vertical and horizontal lines based on a vertical angle cutoff

    :param lines: The lines
    :type lines: numpy.ndarray
    :param vert_angle: Cutoff for identifying vertical lines, in degrees, defaults to 30
    :type vert_angle: float, optional
    :return: The vertical lines, and the horizontal lines
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    vert_angle *= numpy.pi / 180.0

    vidxs = []
    hidxs = []

    for idx, line in enumerate(lines):
        ang = util.line_angle(line)
        # If the angle is near vertical, add it to the vertical index list
        if abs(ang) < vert_angle or abs(numpy.pi - ang) < vert_angle:
            vidxs.append(idx)
        # Otherwise, add it to the horizontal index list
        else:
            hidxs.append(idx)

    return vidxs, hidxs


# Perspective
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


# Properties
def measure_window_dimensions_from_aruco_marker(
    image: numpy.ndarray,
    window_corners: List[Tuple[int, int]],
    aruco_corners: List[Tuple[int, int]],
    color: Tuple[int, int, int] = GREEN,
) -> Tuple[int, int]:
    """Measure the dimensions of the bounding box for a detection

    Annotation does not draw the box, it only puts the measured dimensions on the image

    :param image: The image
    :type image: numpy.ndarray
    :param xyxy: A pair of (x, y) pixel coordinates for the top left and bottom right
        corners of the rectangle
    :type xyxy: Tuple[int, int, int, int]
    :param aruco_corners: The corners of the aruco marker
    :type aruco_corners: List[Tuple[int, int]]
    :param annotate: Annotate the image file as a side-effect?, defaults to False
    :type annotate: bool, optional
    :param color: The BGR annotation color
    :type color: Tuple[int, int, int]
    :return: The width and height in inches
    :rtype: Tuple[int, int]
    """
    # 1. Get the pixel -> inches conversion from the Aruco marker perimeter, which is
    # (4 x 15 cm) = 23.622 inches
    perimeter = cv2.arcLength(aruco_corners, True)
    px2in = 23.622 / perimeter

    # 2. Get the rectangle height and width in inches
    x0, y0, x1, y1 = xyxy

    width = int((x1 - x0) * px2in)
    height = int((y1 - y0) * px2in)

    # Annotate
    if annotate:
        text = f"{width}x{height}"
        cv2.putText(image, text, (x0, y0 + 20), FONT, 0.4, color, 1)

    return width, height


def measure_rectangle_dimensions_from_aruco(
    image: numpy.ndarray,
    xyxy: Tuple[int, int, int, int],
    aruco_corners: List[Tuple[int, int]],
    annotate: bool = False,
    color: Tuple[int, int, int] = GREEN,
) -> Tuple[int, int]:
    """Measure the dimensions of the bounding box for a detection

    Annotation does not draw the box, it only puts the measured dimensions on the image

    :param image: The image
    :type image: numpy.ndarray
    :param xyxy: A pair of (x, y) pixel coordinates for the top left and bottom right
        corners of the rectangle
    :type xyxy: Tuple[int, int, int, int]
    :param aruco_corners: The corners of the aruco marker
    :type aruco_corners: List[Tuple[int, int]]
    :param annotate: Annotate the image file as a side-effect?, defaults to False
    :type annotate: bool, optional
    :param color: The BGR annotation color
    :type color: Tuple[int, int, int]
    :return: The width and height in inches
    :rtype: Tuple[int, int]
    """
    # 1. Get the pixel -> inches conversion from the Aruco marker perimeter, which is
    # (4 x 15 cm) = 23.622 inches
    perimeter = cv2.arcLength(aruco_corners, True)
    px2in = 23.622 / perimeter

    # 2. Get the rectangle height and width in inches
    x0, y0, x1, y1 = xyxy

    width = int((x1 - x0) * px2in)
    height = int((y1 - y0) * px2in)

    # Annotate
    if annotate:
        text = f"{width}x{height}"
        cv2.putText(image, text, (x0, y0 + 20), FONT, 0.4, color, 1)

    return width, height


# Annotaters
def annotate_line(
    image: numpy.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = BLUE,
):
    """Annotate an image in-place with a line described by a series of points

    :param image: The image
    :type image: numpy.ndarray
    :param points: The points, as (x, y) pixel coordinates
    :type points: List[Tuple[float, float]]
    :param color: The BGR annotation color
    :type color: Tuple[int, int, int]
    """
    color = tuple(map(int, color))
    points_list = numpy.intp([points])
    cv2.polylines(image, points_list, True, color, 2)


def annotate_lines(
    image: numpy.ndarray,
    points_list: List[List[Tuple[int, int]]],
    color: Tuple[int, int, int] = BLUE,
):
    """Annotate an image in-place with a line described by a series of points

    :param image: The image
    :type image: numpy.ndarray
    :param points_list: A list of lists of points, as (x, y) pixel coordinates
    :type points_list: List[List[Tuple[float, float]]]
    :param color: The BGR annotation color
    :type color: Tuple[int, int, int]
    """
    color = tuple(map(int, color))
    cv2.polylines(image, numpy.intp(points_list), True, color, 2)


def annotate_rectangle(
    image: numpy.ndarray,
    xyxy: Tuple[int, int, int, int],
    text: str = None,
    color: Tuple[int, int, int] = GREEN,
):
    """Annotate an image in-place with a rectangle and, optionally, some text

    :param image: The image
    :type image: numpy.ndarray
    :param xyxy: A pair of (x, y) pixel coordinates for the top left and bottom right
        corners of the rectangle
    :type xyxy: Tuple[int, int, int, int]
    :param text: Descriptive text to put with the rectangle, defaults to None
    :type text: str, optional
    :param color: The BGR annotation color
    :type color: Tuple[int, int, int]
    """
    color = tuple(map(int, color))
    # Make sure the pixel coordinates are integers
    x0, y0, x1, y1 = map(int, xyxy)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

    if text is not None:
        cv2.putText(image, text, (x0, y1 - 5), FONT, 0.4, color, 1)
