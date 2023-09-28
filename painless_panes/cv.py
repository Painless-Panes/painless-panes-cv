import itertools
from typing import List, Tuple

import cv2  # opencv
import numpy
import skimage.feature

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
    aruco_corners = find_aruco_corners(image, annotate=annotate)
    print("Aruco corners:", aruco_corners)

    # Find the model detections (windows)
    detections = model.get_detections(image)
    print("All detections:", detections)

    # Select the central detection
    detection = select_central_detection(image, detections, annotate=True)
    print("Central detection:", detection)

    # If no marker was found, return early with an error message
    if aruco_corners is None:
        return None, None, "No marker detected!"

    if detection is None:
        return None, None, "No windows detected!"

    # If we have a marker and a detection, make the measurement
    bbox = detection["bounding_box"]
    width, height = measure_rectangle_dimensions_from_aruco(
        image, bbox, aruco_corners, annotate=True
    )

    print("Measured dimensions:", width, height)

    return width, height, ""


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
    corners_list = numpy.intp(corners_list)

    if not corners_list.size:
        return None

    corners = corners_list[0]

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
    bx0, by0, bx1, by1 = bbox
    bwidth = bx1 - bx0
    bheight = by1 - by0
    cx0 = bx0 + 0.1 * bwidth
    cx1 = bx1 - 0.1 * bwidth
    cy0 = by0 + 0.5 * bheight
    cy1 = by1 - 0.5 * bheight

    # Define some helper data
    edge_keys = {"t", "r", "b", "l"}
    edge_nkey_dct = {"t": "rl", "b": "rl", "r": "tb", "l": "rb"}
    corner_keys = {frozenset("tr"), frozenset("br"), frozenset("bl"), frozenset("tl")}

    edges = find_edges_in_bounding_box(image, bbox)
    corners = find_corners_in_bounding_box(image, bbox)
    corner_pool = corners.tolist()

    # 7. Separate them into left- and right-sorted vertical lines and top- and
    # bottom-sorted horizontal lines
    vedges, hedges = partition_lines_by_orientation(edges)

    edge_idx_pool = {
        "t": argsort_horizontal_lines(hedges, cutoff=cy0, reverse=False),
        "r": argsort_vertical_lines(vedges, cutoff=cx1, reverse=True),
        "b": argsort_horizontal_lines(hedges, cutoff=cy1, reverse=True),
        "l": argsort_vertical_lines(vedges, cutoff=cx0, reverse=False),
    }

    for edge_key, edge_idxs in edge_idx_pool.items():
        print(edge_key)
        color = numpy.random.randint(0, 255, 3)
        if edge_key in "tb":
            edges = hedges[edge_idxs]
        else:
            edges = vedges[edge_idxs]
        points_list = numpy.reshape(edges, (-1, 2, 2))
        annotate_lines(image, points_list, color=color)

    # for corner in corners:
    #     cv2.circle(image, numpy.intp(corner), 1, color=RED, thickness=3)

    # # Cutoffs for vertical line positions (10% in from the bounding box frame)
    # cx0 = bx0 + 0.1 * bwidth
    # cx1 = bx1 - 0.1 * bwidth

    # side_line1 = side_lines["r"].pop(0)
    # side_line1 = side_lines["r"].pop(0)
    # side_line2 = side_lines["b"].pop(0)

    # corner = util.line_pair_ray_intersection(side_line1, side_line2)
    # annotate_line(image, numpy.reshape(side_line1, (-1, 2)), color=GREEN)
    # annotate_line(image, numpy.reshape(side_line2, (-1, 2)), color=RED)
    # cv2.circle(image, numpy.intp(corner), 5, color=BLUE, thickness=3)

    # corners_list = []
    # corners_iter = itertools.pairwise(itertools.cycle(side_keys))
    # for side_key1, side_key2 in corners_iter:
    #     print(side_key1, side_key2)
    #     if not (side_lines[side_key1] and side_lines[side_key2]):
    #         break
    #     side_line1 = side_lines[side_key1].pop(0)
    #     side_line2 = side_lines[side_key2].pop(0)

    # vpoints_list = numpy.reshape(vlines, (-1, 2, 2))
    # hpoints_list = numpy.reshape(hlines, (-1, 2, 2))
    # annotate_lines(image, vpoints_list, color=GREEN)
    # annotate_lines(image, hpoints_list, color=RED)
    return image


def find_edges_in_bounding_box(image: numpy.ndarray, bbox: List[int]) -> numpy.ndarray:
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
    bx0, _, bx1, _ = bbox
    bwidth = bx1 - bx0

    # 3. Detect straight lines
    fld = cv2.ximgproc.createFastLineDetector(
        length_threshold=bwidth // 5,
        do_merge=True,
    )
    lines = fld.detect(image_gray)

    # 4. Sort the line points for convenience
    lines = sort_lines_points(lines, axis="y")

    # 5. Select lines that fall within the bounding box
    lines = select_lines_in_bounding_box(lines, bbox)

    # 6. Merge the lines by ray
    lines_groups = util.equivalence_partition(lines, util.lines_fall_on_common_ray)
    lines = list(map(util.merge_lines, lines_groups))

    return lines


def find_corners_in_bounding_box(
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
    bx0, _, bx1, _ = bbox
    bwidth = bx1 - bx0

    # 3. Corner Harris generates a map where high values are likely to have a corner
    block_size = max(2, bwidth // 40)
    corner_map = cv2.cornerHarris(image_gray, blockSize=block_size, ksize=3, k=0.04)

    # 4. Set low values to zero, clearing out small local maxima
    corner_map[corner_map < 0.01 * corner_map.max()] = corner_map.min()

    # 5. Identify remaining local maxima, which are likely to be corners
    corners = skimage.feature.peak_local_max(corner_map.T, min_distance=5)

    # 6. Select corners that fall within the bounding box
    corners = select_points_in_bounding_box(corners, bbox)

    return corners


# Selectors
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

    print(points.shape)
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

    xvals = lines[:, 0, 0::2]
    yvals = lines[:, 0, 1::2]

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
    """Sort the two points for each lines along a particular axis

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


def partition_lines_by_orientation(
    lines: numpy.ndarray, vert_angle: float = 30
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Partition a set of lines by vertical and horizontal orientation

    Partitions into vertical and horizontal lines based on a vertical angle cutoff

    :param lines: The lines
    :type lines: numpy.ndarray
    :param vert_angle: Cutoff for identifying vertical lines, in degrees, defaults to 30
    :type vert_angle: float, optional
    :return: The vertical lines, and the horizontal lines
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    vert_angle *= numpy.pi / 180.0

    vlines = []
    hlines = []

    for line in lines:
        ang = util.line_angle(line)
        # If the angle is near vertical, add it to `vlines`
        if abs(ang) < vert_angle or abs(numpy.pi - ang) < vert_angle:
            vlines.append(line)
        # Otherwise, add it to `hlines`
        else:
            hlines.append(line)

    vlines = numpy.array(vlines)
    hlines = numpy.array(hlines)

    return vlines, hlines


def argsort_vertical_lines(
    vlines: numpy.ndarray, cutoff: float = None, reverse: bool = False
) -> numpy.ndarray:
    """Sort vertical lines by x-coordinates

    :param vlines: The vertical lines
    :type vlines: numpy.ndarray
    :param cutoff: Drop lines past this cutoff, defaults to None
    :type cutoff: float, optional
    :param reverse: Reverse the sort order?, defaults to False
    :type reverse: bool, optional
    :return: The sorted lines
    :rtype: numpy.ndarray
    """
    min_x_vals = numpy.min(numpy.squeeze(vlines)[:, 0::2], axis=1)
    max_x_vals = numpy.max(numpy.squeeze(vlines)[:, 0::2], axis=1)

    if not reverse:
        vsort_idxs = numpy.argsort(min_x_vals, axis=0)
        vsort_idxs = list(vsort_idxs)
        if cutoff is not None:
            vsort_idxs = [i for i in vsort_idxs if max_x_vals[i] < cutoff]
    else:
        vsort_idxs = numpy.argsort(max_x_vals, axis=0)
        vsort_idxs = list(reversed(vsort_idxs))
        if cutoff is not None:
            vsort_idxs = [i for i in vsort_idxs if min_x_vals[i] > cutoff]

    return vsort_idxs


def argsort_horizontal_lines(
    hlines: numpy.ndarray, cutoff: float = None, reverse: bool = False
) -> numpy.ndarray:
    """Sort horizontal lines by y-coordinates

    :param lines: The lines
    :type lines: numpy.ndarray
    :param cutoff: Drop lines past this cutoff, defaults to None
    :type cutoff: float, optional
    :param reverse: Reverse the sort order?, defaults to False
    :type reverse: bool, optional
    :return: The sorted lines
    :rtype: numpy.ndarray
    """
    min_y_vals = numpy.min(numpy.squeeze(hlines)[:, 1::2], axis=1)
    max_y_vals = numpy.max(numpy.squeeze(hlines)[:, 1::2], axis=1)

    if not reverse:
        hsort_idxs = numpy.argsort(min_y_vals, axis=0)
        hsort_idxs = list(hsort_idxs)
        if cutoff is not None:
            hsort_idxs = [i for i in hsort_idxs if max_y_vals[i] < cutoff]
    else:
        hsort_idxs = numpy.argsort(max_y_vals, axis=0)
        hsort_idxs = list(reversed(hsort_idxs))
        if cutoff is not None:
            hsort_idxs = [i for i in hsort_idxs if min_y_vals[i] > cutoff]

    return hsort_idxs


# Properties
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
