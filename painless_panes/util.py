import importlib.resources
from io import BytesIO

import cv2  # opencv
import numpy as np
from numpy import ndarray


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
def opencv_image_array_from_stream(bytes_io: BytesIO) -> ndarray:
    """Get an OpenCV-compatible image object from a BytesIO steam

    :param bytes_io: _description_
    :type bytes_io: BytesIO
    :return: _description_
    :rtype: ndarray
    """
    bytes_io.seek(0)
    image = cv2.imdecode(np.frombuffer(bytes_io.read(), np.uint8), cv2.IMREAD_COLOR)
    return image
