from typing import List, Tuple

import cv2  # opencv
import numpy

from painless_panes import model

# Module constants
BLUE = (255, 0, 0)  # annotation color
GREEN = (0, 255, 0)  # annotation color
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
    :param bbox: A bounding box for the window
    :type bbox: List[int]
    :param annotate: Annotate the image file as a side-effect?, defaults to False
    :type annotate: bool, optional
    :return: The corner positions of the Aruco marker, in pixels:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :rtype: numpy.ndarray
    """
    image = image.copy()

    # 1. Find all edges with canny edge detection
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    all_edges = cv2.Canny(image, 30, 150)

    # 2. Select edges within the bounding box
    x0, y0, x1, y1 = bbox
    edges = numpy.zeros_like(all_edges)
    edges[y0:y1, x0:x1] = all_edges[y0:y1, x0:x1]

    # 3. Detect straight lines among the edges
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(edges)
    # image = fld.drawSegments(image, lines, linecolor=(0, 255, 0))

    # 4. Find the longest


def partition_lines_by_orientation(
    lines: numpy.ndarray, vert_angle: float = 30
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Partition a set of lines by vertical and horizontal orientation

    Partitions into vertical and horizontal lines based on a vertical angle cutoff

    :param lines: The lines
    :type lines: numpy.ndarray
    :param vert_angle: Cutoff for identifying vertical lines, defaults to 30
    :type vert_angle: float, optional
    :return: The vertical lines, and the horizontal lines
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    vlines = []
    hlines = []

    print(lines.shape)
    print("LINES:")
    for line in lines:
        (lin,) = line
        print(line)
        # Calculate the angle
        x, y = numpy.subtract(lin[2:], lin[:2])
        ang = numpy.arctan(x / y)
        # If the angle is near vertical, add it to `vlines`
        if abs(ang) < vert_angle or abs(numpy.pi - ang) < vert_angle:
            vlines.append(line)
        # Otherwise, add it to `hlines`
        else:
            hlines.append(line)

    vlines = numpy.array(vlines)
    hlines = numpy.array(hlines)
    print(vlines.shape)
    print(hlines.shape)
    return vlines, hlines


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
    points_list = numpy.intp([points])
    cv2.polylines(image, points_list, True, color, 2)


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
    # Make sure the pixel coordinates are integers
    x0, y0, x1, y1 = map(int, xyxy)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

    if text is not None:
        cv2.putText(image, text, (x0, y1 - 5), FONT, 0.4, color, 1)
