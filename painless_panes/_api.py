"""Flask API
"""
import io

import flask
import flask_cors

from painless_panes import cv, util

api = flask.Flask(__name__)
flask_cors.CORS(api, expose_headers="*")


@api.route("/cv/api/ping", methods=["GET"])
def ping():
    """@api {get} /cv/api/ping Ping the server to make sure it is running

    @apiSuccess {Object} If successful, sends a 200 code with the following content:
        {"contents": "pong"}
    """
    return {"contents": "pong"}, 200


@api.route("/cv/api/window/measure", methods=["POST"])
def measure_window():
    """@api {post} /cv/api/window/measure Measure window dimensions from a photo

    @apiBody {Blob} image The photo image file, as a blob

    @apiSuccessHeader {String} width The measured width, in inches
    @apiSuccessHeader {String} height The measured height, in inches
    @apiSuccess {Blob} The annotated image file, as a Blob

    Note: If the image was processed but the measurement was unsuccessful, this will
    still return a 200 code. In this case, it will return a partially annotated image,
    which may contain either the Aruco marker or the detected window, along with an
    error message header.

    @apiErrorHeader {String} error_message An error message
    @apiError {Blob} The partially annotated image
    """
    # Get the FileStorage object
    file_storage = flask.request.files["image"]

    # Create a BytesIO object
    bytes_io = io.BytesIO()

    # Save the FileStorage object to the BytesIO object
    file_storage.save(bytes_io)

    # Decode the image as an OpenCV image object
    image = util.opencv_image_array_from_bytes_io(bytes_io)
    width, height, annotated_image, error_message = cv.measure_window(image)

    # Send the BytesIO object as a file, putting measurements in the headers
    annotated_bytes_io = util.bytes_io_from_opencv_image_array(annotated_image)
    res = flask.make_response(
        flask.send_file(annotated_bytes_io, mimetype="image/jpeg")
    )

    if height is not None and width is not None:
        res.headers.add("width", width)
        res.headers.add("height", height)
    else:
        res.headers.add("error_message", error_message)

    return res, 200
