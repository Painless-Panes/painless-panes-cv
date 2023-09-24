import importlib.resources
import io

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
