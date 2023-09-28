import importlib.resources
import io
from typing import Callable, List, Tuple

import cv2  # opencv
import numpy


# Access text file resources
def file_path(module: str, name: str) -> str:
    """Get the path to a file in a module

    :param module: The module where the resource is located, e.g. painless_panes.model
    :type module: str
    :param name: The file name
    :type name: str
    :return: The path
    :rtype: str
    """
    file_opener = importlib.resources.files(module) / name
    # Conversion to string yields the absolute path
    return str(file_opener)


def file_contents(module: str, name: str, strip: bool = True) -> str:
    """Get the contents of a file in a module

    :param module: The module where the resource is located, e.g. painless_panes.model
    :type module: str
    :param name: The file name
    :type name: str
    :param strip: Strip whitespace from the beginning and end? default `True`
    :type strip: bool, optional
    :return: The file contents, as a string
    :rtype: str
    """
    file_opener = importlib.resources.files(module) / name
    file_contents = file_opener.open().read()
    if strip:
        file_contents = file_contents.strip()
    return file_contents


# Image conversion
def opencv_image_array_from_bytes_io(bytes_io: io.BytesIO) -> numpy.ndarray:
    """Get an OpenCV image array from a BytesIO steam

    :param bytes_io: The binary stream, which acts like a file object
    :type bytes_io: io.BytesIO
    :return: An OpenCV image array
    :rtype: numpy.ndarray
    """
    bytes_io.seek(0)
    image = cv2.imdecode(
        numpy.frombuffer(bytes_io.read(), numpy.uint8), cv2.IMREAD_COLOR
    )
    return image


def bytes_io_from_opencv_image_array(image: numpy.ndarray) -> io.BytesIO:
    """Get a BytesIO stream from an OpenCV image array

    :param image: The OpenCV image array
    :type image: numpy.ndarray
    :return: A binary stream, which acts like a file object
    :rtype: io.BytesIO
    """
    _, image_bytes = cv2.imencode(".jpg", image)
    bytes_io = io.BytesIO(image_bytes)
    bytes_io.seek(0)
    return bytes_io


# 2D geometry
def line_length(line: Tuple[float, float, float, float]) -> float:
    """Measure the length of a 2D line

    :param line: A line, in xyxy format
    :type line: Tuple[float, float, float, float]
    :returns: The length
    :rtype: float
    """
    x0, y0, x1, y1 = numpy.squeeze(line)
    length = numpy.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return length


def line_angle(line: Tuple[float, float, float, float], axis: str = "y") -> float:
    """Measure the angle of a 2D line

    :param line: A line, in xyxy format
    :type line: Tuple[float, float, float, float]
    :param axis: The axis to measure the angle from, defaults to 'y'
    :type axis: str, optional
    :returns: The angle in radians
    :rtype: float
    """
    x0, y0, x1, y1 = numpy.squeeze(line)
    num = y1 - y0
    denom = x1 - x0

    # Flip numerator and denominator if measuring from the y axis
    if axis == "y":
        num, denom = denom, num

    ratio = num / denom if not denom == 0.0 else numpy.inf
    ang = numpy.arctan(ratio)
    return ang


def ray_angle_difference(ang1: float, ang2: float) -> float:
    """Get the ray angle difference between two angles

    Returns the smallest angle between the two rays. Examples:
        -15, 15 -> 30
        -15, 90 -> 75
        -90, 90 -> 0
        -83, 95 -> 2

    Modified from https://gamedev.stackexchange.com/a/4472

    :param ang1: The first angle in radians
    :type ang1: float
    :param ang2: The second angle in radians
    :type ang2: float
    :return: The ray angle difference
    :rtype: float
    """
    return numpy.pi - abs(abs(ang1 - ang2) - numpy.pi)


def lines_fall_on_common_ray(
    line1: Tuple[int, int, int, int],
    line2: Tuple[int, int, int, int],
    ang_thresh: float = 3,
    dist_thresh: float = 3,
) -> bool:
    """Find out whether two lines fall on a common ray

    Lines share a common ray if they are parallel (measured by an angle threshold) and
    nearly overlapping (measured by a perpendicular distance threshold).

    :param line1: The first line
    :type line1: Tuple[int, int, int, int]
    :param line2: The second line
    :type line2: Tuple[int, int, int, int]
    :param ang_thresh: Degree angle threshold for assessing parallelity, defaults to 2
    :type ang_thresh: float, optional
    :param dist_thresh: Pixel distance threshold for assessing overlap, defaults to 2
    :type dist_thresh: float, optional
    :return: True if they fall on a common ray, False if they don't
    :rtype: bool
    """
    ang_thresh *= numpy.pi / 180.0

    # 0. Sort so that line A is the longer line
    line_a, line_b = sorted([line1, line2], key=line_length, reverse=True)

    # 1. Find the angle difference to assess whether the rays are parallel
    ang_a = line_angle(line_a)
    ang_b = line_angle(line_b)
    ang_diff = ray_angle_difference(ang_a, ang_b)
    are_parallel = abs(ang_diff) < ang_thresh

    are_overlapping = True
    if are_parallel:
        # 2. Find the perpendicular distances to assess whether the rays are overlapping
        ox, oy = numpy.squeeze(line_a)[:2]
        for px, py in numpy.reshape(line_b, (2, 2)):
            line_p = [ox, oy, px, py]
            len_p = line_length(line_p)
            ang_p = line_angle(line_p)
            ang_diff = ray_angle_difference(ang_a, ang_p)
            # The perpendiculr distance is (point distance) * sin(angle difference)
            perp_dist = len_p * numpy.sin(ang_diff)
            are_parallel &= abs(perp_dist) < dist_thresh

    return are_parallel & are_overlapping


def merge_lines(lines: numpy.ndarray) -> Tuple[float, float, float, float]:
    r"""Merge broken or overlapping lines into a single line

    Lines must fall on the same ray for this to make sense:

      \             \
                     \
        \     =>      \   
                       \
          \             \

    :param lines: The lines to be merged
    :type lines: numpy.ndarray
    :return: The resulting merged line
    :rtype: Tuple[float, float, float, float]
    """
    lens = list(map(line_length, lines))
    angs = list(map(line_angle, lines))
    angs = [(a + numpy.pi) % numpy.pi for a in angs]

    # Do a length-weighted average to find the average line angle
    ang = sum(L * a for L, a in zip(lens, angs)) / sum(lens)

    # Convert the lines to a series of points
    points = numpy.reshape(lines, (-1, 2))

    # Use the first point as the origin
    ox, oy = points[0]

    # Determine the perpendicular and parallel positions of each point along the ray,
    # relative to the origin
    perp_posns = []  # perpendicular positions
    parl_posns = []  # parallel positions
    for px, py in points:
        line_p = [ox, oy, px, py]
        len_p = line_length(line_p)
        ang_p = line_angle(line_p)
        perp_posn = len_p * numpy.sin(ang_p - ang)
        parl_posn = len_p * numpy.cos(ang_p - ang)
        perp_posns.append(perp_posn)
        parl_posns.append(parl_posn)

    # Get the average perpendicular position
    perp_shift = sum(perp_posns) / len(perp_posns)

    # Get the parallel starting and ending positions
    end_posns = [min(parl_posns), max(parl_posns)]

    # Apply the perpendicular shift to the origin (should be small)
    ox = ox + perp_shift * numpy.cos(ang)
    oy = oy - perp_shift * numpy.sin(ang)

    # Get x, y coordinates for the starting and ending positions
    end_points = []
    for end_posn in end_posns:
        end_x = ox + end_posn * numpy.sin(ang)
        end_y = oy + end_posn * numpy.cos(ang)
        end_points.extend([end_x, end_y])

    assert len(end_points) == 4
    return tuple(end_points)


def line_pair_ray_intersection_no_overhang(
    line1: numpy.ndarray,
    line2: numpy.ndarray,
    dist_thresh: int = 2,
) -> Tuple[float, float]:
    """Find the intersection of the rays extending from a pair of lines, returning
    `None` if there is an overhang

    Intersections without overhang are included:
          ______             ______
                            |
        |             or    |
        |                   |

    Intersections with overhang are not included:

      __|______
        |
        |

    :param line1: The first line
    :type line1: numpy.ndarray
    :param line2: The second line
    :type line2: numpy.ndarray
    :param dist_thresh: The pixel distance threshold of allowed overhang, defaults to 2
    :type dist_thresh: int, optional
    :return: The x, y coordinates of the intersection point
    :rtype: Tuple[float, float]
    """
    xint, yint = line_pair_ray_intersection(line1, line2)

    def _line_overhang(line):
        x0, y0, x1, y1 = numpy.squeeze(line)

        # Sort the line bounds
        bx0, bx1 = sorted([x0, x1])
        by0, by1 = sorted([y0, y1])

        # Add a small amount of margin, in case of rounding error
        bx0 -= dist_thresh / 10.0
        by0 -= dist_thresh / 10.0
        bx1 += dist_thresh / 10.0
        by1 += dist_thresh / 10.0

        # If the point lies on the line, the overhang is the minimum end point distance
        if (bx0 < xint < bx1) and (by0 < yint < by1):
            end_points = [(x0, y0), (x1, y1)]
            overhang = min(line_length([xint, yint, x, y]) for x, y in end_points)
        else:
            overhang = 0

        return overhang

    overhang1 = _line_overhang(line1)
    overhang2 = _line_overhang(line2)

    if overhang1 > dist_thresh or overhang2 > dist_thresh:
        return None

    return xint, yint


def line_pair_ray_intersection(
    line1: numpy.ndarray, line2: numpy.ndarray
) -> Tuple[float, float]:
    """Find the intersection of the rays extending from a pair of lines

    Formula:

        x = (x2 cot(ang2) - x1 cot(ang1) - (y2 - y1)) / (cot(ang2) - cot(ang1))
        y = (y2 tan(ang2) - y1 tan(ang1) - (x2 - x1)) / (tan(ang2) - tan(ang1))

    where
        the points are (x1, y1) and (x2, y2),
        the angles are ang1 and ang2
        cot() and tan() are cotangent and tangent

    :param line1: The first line
    :type line1: numpy.ndarray
    :param line2: The second line
    :type line2: numpy.ndarray
    :return: The x, y coordinates of the intersection point
    :rtype: Tuple[float, float]
    """
    x1, y1 = numpy.squeeze(line1)[:2]
    x2, y2 = numpy.squeeze(line2)[:2]
    ang1 = line_angle(line1)
    ang2 = line_angle(line2)

    # Parallel lines don't intersect, so return None
    if ang1 == ang2:
        return None

    tan1 = numpy.tan(ang1)
    tan2 = numpy.tan(ang2)
    cot1 = 1.0 / tan1
    cot2 = 1.0 / tan2

    xint = (x2 * cot2 - x1 * cot1 - (y2 - y1)) / (cot2 - cot1)
    yint = (y2 * tan2 - y1 * tan1 - (x2 - x1)) / (tan2 - tan1)
    return xint, yint


#  Miscellaneous
def equivalence_partition(
    iterable: List[object], relation: Callable
) -> List[List[object]]:
    """Partitions a set of objects into equivalence classes

      See https://stackoverflow.com/a/38924631

    :param iterable: Collection of objects to be partitioned
    :type iterable: List[object]
    :param relation: A function of two objects returning `True` if they are equivalent
    :type relation: Callable
    :return: The same objects, grouped into equivalent classes
    :rtype: List[List[object]]
    """
    classes = []
    for obj in iterable:  # for each object
        # find the class it is in
        found = False
        for cls in classes:
            # is it equivalent to this class?
            if relation(next(iter(cls)), obj):
                cls.append(obj)
                found = True
                break
        if not found:  # it is in a new class
            classes.append([obj])

    return classes
