from typing import List, Tuple

import cv2  # opencv
import numpy as np
import ultralytics as ul

from painless_panes import util

# Read model info from painless_panes/model
MODEL_FILE_PATH = util.file_path("painless_panes.model", "model.pt")
CLASSES_FILE_CONTENTS = util.file_contents("painless_panes.model", "classes.txt")

# Module constants
BLUE = (255, 0, 0)  # annotation color
GREEN = (0, 255, 0)  # annotation color
FONT = cv2.FONT_HERSHEY_SIMPLEX
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MODEL = ul.YOLO(MODEL_FILE_PATH)
CLASS_NAMES = CLASSES_FILE_CONTENTS.splitlines()


def find_aruco_marker(
    image: np.ndarray, annotate: bool = False
) -> Tuple[List[int], np.ndarray]:
    """Find one aruco marker in an image

    Returns the first one found, if there is one

    :param image: The image
    :type image: np.ndarray
    :param annotate: Annotate the image file as a side-effect?, defaults to False
    :type annotate: bool, optional
    :return: _description_
    :rtype: Tuple[List[int], np.ndarray]
    """
    corners, _, _ = cv2.aruco.detectMarkers(image, ARUCO_DICT, parameters=ARUCO_PARAMS)
    corners = tuple(np.intp(corners[0])) if corners else None

    # If requested, annotate the image as a side-effect
    if annotate and corners:
        cv2.polylines(image, np.intp([corners]), True, BLUE, 2)

    return corners
