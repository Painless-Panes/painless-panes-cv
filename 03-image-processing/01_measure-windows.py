import cv2 as cv
import numpy as np
import ultralytics as ul

BLUE = (255, 0, 0)  # annotation color
GREEN = (0, 255, 0)  # annotation color
FONT = cv.FONT_HERSHEY_SIMPLEX
ARUCO_PARAMS = cv.aruco.DetectorParameters()
ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

model = ul.YOLO("custom-model.pt")
class_names = open("classes.txt").read().strip().splitlines()

filename = "example"
image = cv.imread(f"{filename}.jpg")
height, width, channels = image.shape

# Resize the image
width = int(width * 800 / height)
height = int(height * 800 / height)
image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)

# If an Aruco marker can be detected, get the pixel distance ratio
px2in = 1.0
corners, _, _ = cv.aruco.detectMarkers(image, ARUCO_DICT, parameters=ARUCO_PARAMS)
if corners:
    cv.polylines(image, np.intp(corners), True, BLUE, 2)
    perimeter = cv.arcLength(np.intp(corners[0]), True)
    print("pixel perimeter:", perimeter)
    px2in = 23.622 / perimeter
    print("pixels per inch:", 1.0 / px2in)

# Convert RGB => BGR for prediction
bgr_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
results = model.predict(bgr_image, conf=0.5, project=".")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = class_names[class_id]
        conf = float(box.conf)
        x0, y0, x1, y1 = map(int, box.xyxy[0])
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
