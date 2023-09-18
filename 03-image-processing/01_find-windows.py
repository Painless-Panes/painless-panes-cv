import cv2 as cv
import ultralytics as ul

COLOR = (0, 255, 0)  # annotation color
FONT = cv.FONT_HERSHEY_SIMPLEX

model = ul.YOLO("custom-model.pt")
class_names = open("classes.txt").read().strip().splitlines()

filename = "example"
image = cv.imread(f"{filename}.jpg")
height, width, channels = image.shape

# Resize the image
width = int(width * 800 / height)
height = int(height * 800 / height)
image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)

# Convert RGB => BGR for prediction
bgr_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
results = model.predict(bgr_image, conf=0.5, project=".")

image_out = image.copy()
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = class_names[class_id]
        conf = float(box.conf)
        x0, y0, x1, y1 = map(int, box.xyxy[0])
        print(x0, y0, x1, y1)

        cv.rectangle(image_out, (x0, y0), (x1, y1), COLOR, 2)
        cv.putText(image_out, class_name, (x0, y0 - 5), FONT, 1, COLOR, 2)
        cv.putText(image_out, f"{conf:.2f}", (x0, y1 - 5), FONT, 1, COLOR, 2)

cv.imshow("Annotated", image_out)
cv.imwrite(f"annotated/{filename}_annotated.jpg", image)

# Stop running when we close the window
cv.waitKey()
cv.destroyAllWindows()
