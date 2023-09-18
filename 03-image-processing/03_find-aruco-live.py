import cv2 as cv
import numpy as np

COLOR = (0, 255, 0)
FONT = cv.FONT_HERSHEY_SIMPLEX

ARUCO_PARAMS = cv.aruco.DetectorParameters()
# ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)


def detect_objects(frame):
    # Convert Image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Create a Mask with adaptive threshold
    mask = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 5
    )

    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # cv.imshow("mask", mask)
    objects_contours = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 2000:
            # cnt = cv.approxPolyDP(cnt, 0.03*cv.arcLength(cnt, True), True)
            objects_contours.append(cnt)

    return objects_contours


# Load video capture
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, image = capture.read()

    # First, get the Aruco marker
    corners, _, _ = cv.aruco.detectMarkers(image, ARUCO_DICT, parameters=ARUCO_PARAMS)
    print(corners)


    # Aruco perimeter
    px2cm = 1.
    if corners:
        cv.polylines(image, np.intp(corners), True, (255, 0, 0), 5)
        aruco_perimeter = cv.arcLength(np.intp(corners[0]), True)
        print('perimeter:', aruco_perimeter)
        px2cm = 60. / aruco_perimeter
        print("pixel ratio:", px2cm)

    contours = detect_objects(image)

    # Draw object boundaries
    for contour in contours:
        # Find minimum area rectangle
        rect = cv.minAreaRect(contour)
        (x, y), (w, h), angle = rect

        # Convert it to a box
        box = cv.boxPoints(rect)
        cv.polylines(image, np.intp([box]), True, COLOR, 2)

        cv.putText(image, f"width: {w * px2cm:.1f}", np.intp([x - 100, y - 15]), FONT, 1, COLOR, 2)
        cv.putText(image, f"height: {h * px2cm:.1f}", np.intp([x - 100, y + 15]), FONT, 1, COLOR, 2)

    cv.imshow("Image", image)
    key = cv.waitKey(1)
    if key == 27:
        break

capture.release()
cv.destroyAllWindows()
