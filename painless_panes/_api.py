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
    image = util.opencv_image_array_from_stream(bytes_io)
    aruco_corners = cv.find_aruco_marker(image, annotate=True)
    print("corners:", aruco_corners)
    import cv2
    cv2.imwrite("test.jpg", image)

    # Send the BytesIO object as a file, putting measurements in the headers
    bytes_io.seek(0)
    res = flask.send_file(bytes_io, mimetype="image/jpeg")
    res.headers.add("width", 12)
    res.headers.add("height", 34)

    print("Sending response!")
    return res
