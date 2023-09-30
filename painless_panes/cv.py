import itertools
from typing import List, Tuple

import cv2  # opencv
import numpy

from painless_panes import model, util

# Module constants
BLUE = (255, 0, 0)  # annotation color
GREEN = (0, 255, 0)  # annotation color
YELLOW = (0, 255, 255)  # annotation color
RED = (0, 0, 255)  # annotation color
FONT = cv2.FONT_HERSHEY_SIMPLEX
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def measure_window(image: numpy.ndarray) -> Tuple[int, int, numpy.ndarray, str]:
    """Measure a window from an image with an Aruco marker

    :param image: The image
    :type image: numpy.ndarray
    :param annotate: Annotate the image file as a side-effect?, defaults to False
    :type annotate: bool, optional
    :returns: The width and height, in inches, and the annotated image, along with an
        error message, which is empty unless the measurement failed for some reason
    :rtype: Tuple[int, int, numpy.ndarray, str]
    """
    image_annotated = image.copy()

    # Find the aruco marker
    aruco_corners, image_annotated = find_aruco_corners(image, annotate=image_annotated)
    print("Aruco corners:", aruco_corners)

    # Find the model detections (windows)
    detections = model.get_detections(image)
    print("All detections:", detections)

    # Select the central detection
    detection, image_annotated = select_central_window(
        image, detections, annotate=image_annotated
    )
    print("Central detection:", detection)

    # If no marker was found, return early with an error message
    if aruco_corners is None:
        return None, None, image_annotated, "No marker detected!"

    # If no window was found, return early withan error message
    if detection is None:
        return None, None, image_annotated, "No window detected!"

    print("Detection:", detection)
    window_bbox = detection["bounding_box"]
    window_corners, image_annotated = find_window_corners(
        image, window_bbox, annotate=image_annotated
    )
    print("Window corners:", window_corners)

    if window_corners is None:
        return None, None, image_annotated, "Could not find window corners."

    # These functions are supposed to correct for perspective skewing, but they make the
    # results less accurate, so I have commented them out and moved them to _archive.py

    # # Correct for perspective skewing
    # image, (aruco_corners, window_corners) = align_image_perspective_on_rectangle(
    #     image, window_corners, points_list=(aruco_corners, window_corners)
    # )

    # # Scale to make the aruco marker square
    # image, (aruco_corners, window_corners) = scale_perspective_to_square(
    #     image, aruco_corners, points_list=(aruco_corners, window_corners)
    # )

    # If we have a marker and a detection, make the measurement
    (
        window_width,
        window_height,
        image_annotated,
    ) = measure_window_dimensions_from_aruco_marker(
        image, window_corners, aruco_corners, annotate=image_annotated
    )

    print("Measured dimensions:", window_width, window_height)

    return window_width, window_height, image_annotated, ""


# Properties
def measure_window_dimensions_from_aruco_marker(
    image: numpy.ndarray,
    window_corners: List[Tuple[int, int]],
    aruco_corners: List[Tuple[int, int]],
    annotate: numpy.ndarray = None,
) -> Tuple[int, int]:
    """Measure the dimensions of the bounding box for a detection

    Annotation does not draw the box, it only puts the measured dimensions on the image

    :param image: The image
    :type image: numpy.ndarray
    :param window_corners: The corners of a window
    :type window_corners: List[Tuple[int, int]]
    :param aruco_corners: The corners of an aruco marker
    :type aruco_corners: List[Tuple[int, int]]
    :param annotate: A copy of the image to be annotated; If `None`, a new copy created;
        defaults to None
    :type annotate: numpy.ndarray, optional
    :return: The width and height in inches
    :rtype: Tuple[int, int]
    """
    # 1. Get the pixel -> inches conversion from the Aruco marker perimeter, which is
    # (4 x 15 cm) = 23.622 inches
    perimeter = cv2.arcLength(aruco_corners, True)
    px2in = 23.622 / perimeter

    # 2. Get the window height and width in inches
    tl_corner, tr_corner, br_corner, bl_corner = numpy.array(window_corners)
    t_len = util.line_length([*tl_corner, *tr_corner])
    b_len = util.line_length([*bl_corner, *br_corner])
    l_len = util.line_length([*tl_corner, *bl_corner])
    r_len = util.line_length([*tr_corner, *br_corner])

    width = int(numpy.average((t_len, b_len)) * px2in)
    height = int(numpy.average((l_len, r_len)) * px2in)

    # Annotate
    if annotate is None:
        annotate = image.copy()

    text = f"{width}x{height}"
    text_position = numpy.intp(numpy.add(tl_corner, [5, 20]))
    cv2.putText(annotate, text, text_position, FONT, 0.6, RED, 2)

    return width, height, annotate


# Finders
def find_aruco_corners(
    image: numpy.ndarray, annotate: numpy.ndarray = None
) -> numpy.ndarray:
    """Find one aruco marker in an image

    Returns the first one found, if there is one

    :param image: The image
    :type image: numpy.ndarray
    :param annotate: A copy of the image to be annotated; If `None`, a new copy created;
        defaults to None
    :type annotate: numpy.ndarray, optional
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
    if annotate is None:
        annotate = image.copy()

    annotate_line(annotate, corners)

    return corners, annotate


def find_window_corners(
    image: numpy.ndarray, bbox: List[int], annotate: numpy.ndarray = None
) -> numpy.ndarray:
    """Find the corners of a window in an image

    Returns the first one found, if there is one

    :param image: The image
    :type image: numpy.ndarray
    :param bbox: A window bounding box in xyxy format
    :type bbox: List[int]
    :param annotate: A copy of the image to be annotated; If `None`, a new copy created;
        defaults to None
    :type annotate: numpy.ndarray, optional
    :return: The window corner positions, in pixels:
        [
            [x1, y1],  # top-left corner
            [x2, y2],  # top-right corner
            [x3, y3],  # bottom-right corner
            [x4, y4]   # bottom-left corner
        ]
    :rtype: numpy.ndarray
    """
    # 0. Define relative distances
    bx0, by0, bx1, by1 = bbox
    bwidth = bx1 - bx0
    bheight = by1 - by0
    blength = min(bwidth, bheight)

    overhang_thresh = max(3, blength // 50)

    #     a. Define inside margin, assuming vertical edges ar near the bounding box edge
    mx0 = bx0 + 0.2 * bwidth
    mx1 = bx1 - 0.2 * bwidth
    my0 = by0 + 0.3 * bheight
    my1 = by1 - 0.3 * bheight

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
        "l": argsort_vertical_lines_in_interval(edge_lines, start=px0, end=mx0),
        "r": argsort_vertical_lines_in_interval(edge_lines, start=px1, end=mx1),
        # Horizontal edges are sorted from the outside in, so outer edges come first
        "t": argsort_horizontal_lines_in_interval(edge_lines, start=py0, end=my0),
        "b": argsort_horizontal_lines_in_interval(edge_lines, start=py1, end=my1),
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
    window_corners_list = []
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
            window_corners_list.append(window_corners)

    # For now, assume the best selection is the outermost corners
    window_corners = window_corners_list[0]

    if annotate is None:
        annotate = image.copy()

    annotate_line(annotate, window_corners, color=RED)
    return window_corners, annotate


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
def select_central_window(
    image: numpy.ndarray, detections: List[dict], annotate: numpy.ndarray = None
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
    :param annotate: A copy of the image to be annotated; If `None`, a new copy created;
        defaults to None
    :type annotate: numpy.ndarray, optional
    :returns: The central detection from the list
    :rtype: dict
    """
    if not detections:
        return None

    if annotate is None:
        annotate = image.copy()

    height, width, _ = image.shape

    # Identify the detectin closest to he center
    center_point = numpy.intp([width // 2, height // 2])
    min_dist = numpy.inf
    central_idx = None
    for idx, detection in enumerate(detections):
        bbox = detection["bounding_box"]

        corners = numpy.reshape(bbox, (2, 2))
        # Calculate the sum of the corner distances (relative to center)
        dist = sum(numpy.linalg.norm(numpy.subtract(p, center_point)) for p in corners)
        if dist < min_dist:
            central_idx = idx
            min_dist = dist

        annotate_rectangle(annotate, bbox, color=YELLOW)

    # Check for other detections vertically stacked above/below this one, which are
    # likely to be part of the same window
    central_window = detections[central_idx]
    for idx, detection in enumerate(detections):
        dx0, dy0, dx1, dy1 = detection["bounding_box"]
        cx0, cy0, cx1, cy1 = central_window["bounding_box"]
        cwidth = cx1 - cx0
        cheight = cy1 - cy0

        similar_x0 = abs(dx0 - cx0) < 0.1 * cwidth
        similar_x1 = abs(dx1 - cx1) < 0.1 * cwidth
        common_y_edge = (
            abs(dy0 - cy0) < 0.1 * cheight or
            abs(dy1 - cy1) < 0.1 * cheight or
            abs(dy0 - cy1) < 0.2 * cheight or
            abs(dy1 - cy0) < 0.2 * cheight
        )

        # If this detection is vertically stacked above/below the central one, combine
        # their bounding boxes
        if similar_x0 and similar_x1 and common_y_edge:
            central_window["bounding_box"] = (
                min(dx0, cx0),
                min(dy0, cy0),
                max(dx1, cx1),
                max(dy1, cy1),
            )

    class_name = central_window["class_name"]
    conf = central_window["confidence"]
    text = f"{class_name} {conf:.2f}"
    bbox = central_window["bounding_box"]
    annotate_rectangle(annotate, bbox, text=text, color=GREEN)

    return central_window, annotate


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
    vert_angle: float = 20,
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
    vert_angle: float = 20,
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
    lines: numpy.ndarray, vert_angle: float = 20
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
        position = (x0 + 5, y1 - 5)
        cv2.putText(image, text, position, FONT, 0.6, color, 2)
