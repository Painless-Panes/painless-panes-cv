"""Flask API
"""
import io

import flask
import flask_cors

from painless_panes import cv, util

api = flask.Flask(__name__)
flask_cors.CORS(api)


@api.route("/cv/api/window/measure", methods=["POST"])
def measure_window():
    print("here")

    # Get the FileStorage object
    file_storage = flask.request.files["image"]

    # Create a BytesIO object
    bytes_io = io.BytesIO()

    # Save the FileStorage object to the BytesIO object
    file_storage.save(bytes_io)

    # Decode the image as an OpenCV image object
    image = util.opencv_image_array_from_bytes_io(bytes_io)
    annotated_image = image.copy()
    aruco_corners = cv.find_aruco_marker(annotated_image, annotate=True)
    print("Aruco corners:", aruco_corners)

    import cv2
    cv2.imwrite("test.jpg", annotated_image)

    # Send the BytesIO object as a file, putting measurements in the headers
    annotated_bytes_io = util.bytes_io_from_opencv_image_array(annotated_image)
    res = flask.send_file(annotated_bytes_io, mimetype="image/jpeg")
    res.headers.add("width", 12)
    res.headers.add("height", 34)

    print("Sending response!")
    return res
