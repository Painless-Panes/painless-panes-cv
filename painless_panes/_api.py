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
    return {"contents": "pong"}, 200


@api.route("/cv/api/window/measure", methods=["POST"])
def measure_window():
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

    print("Headers added:", res.headers)

    return res
