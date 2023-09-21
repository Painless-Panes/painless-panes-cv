import cv2 as cv
import numpy as np
import ultralytics as ul


BLUE = (255, 0, 0)  # annotation color
GREEN = (0, 255, 0)  # annotation color
FONT = cv.FONT_HERSHEY_SIMPLEX
ARUCO_PARAMS = cv.aruco.DetectorParameters()
ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
MODEL = ul.YOLO("custom-model.pt")
CLASS_NAMES = open("classes.txt").read().strip().splitlines()


def find_aruco_marker(image):
    corners, _, _ = cv.aruco.detectMarkers(image, ARUCO_DICT, parameters=ARUCO_PARAMS)
    return tuple(np.intp(corners[0])) if corners else None


def inches_per_pixel_from_aruco_marker(aruco_corners):
    """Get the pixel to inches conversion from the aruco marker

    :param image: The image
    """
    perimeter = cv.arcLength(np.intp(aruco_corners), True)
    px2in = 23.622 / perimeter
    return px2in


def find_windows(image):
    """Find window in an image using a window object detection model

    :param image: The image
    :param window_model: A YOLO object detection model
    """
    # Convert RGB => BGR for prediction
    bgr_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = MODEL.predict(bgr_image, conf=0.2, project=".")

    windows = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            x0, y0, x1, y1 = map(int, box.xyxy[0])
            windows.append({
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(box.conf),
                "bounding_box": (x0, y0, x1, y1),
            })

    return windows


filename = "example2"
image = cv.imread(f"{filename}.jpg")
height, width, channels = image.shape

# Resize the image
width = int(width * 800 / height)
height = int(height * 800 / height)
image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)

# If an Aruco marker can be detected, get the pixel distance ratio
aruco_corners = find_aruco_marker(image)
px2in = 1.
if aruco_corners:
    # Draw a blue box around the aruco marker
    cv.polylines(image, np.intp([aruco_corners]), True, BLUE, 2)
    # Get th conversion
    px2in = inches_per_pixel_from_aruco_marker(aruco_corners)

# Convert RGB => BGR for prediction
windows = find_windows(image)

for window in windows:
        class_name = window["class_name"]
        conf = window["confidence"]
        x0, y0, x1, y1 = window["bounding_box"]
        print(x0, y0, x1, y1)

        w = (x1 - x0) * px2in
        h = (y1 - y0) * px2in

        cv.rectangle(image, (x0, y0), (x1, y1), GREEN, 2)
        cv.putText(image, f"{class_name} {conf:.2f}", (x0, y0 - 5), FONT, 1, GREEN, 2)
        cv.putText(image, f"{w:.0f}x{h:.0f}", (x0, y1 - 5), FONT, 1, GREEN, 2)

cv.imshow("Annotated", image)
cv.imwrite(f"annotated/{filename}_annotated.jpg", image)

# Stop running when we close the window
cv.waitKey()
cv.destroyAllWindows()
