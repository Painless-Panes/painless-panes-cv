import cv2
import painless_panes

file_name = "example1.jpg"      # cuts off bottom
# file_name = "example7.png"      # wonky corners

image = cv2.imread(f"original/{file_name}")
height, width, _ = image.shape

width = int(width * 800 / height)
height = 800
image = cv2.resize(image, (width, height))

w_width, w_height, image_annotated, message = painless_panes.cv.measure_window(image)
print("Window dimensions:", w_width, w_height)

cv2.imshow("Original", image)
cv2.imshow("Annotated", image_annotated)
cv2.imwrite(f"annotated/{file_name}", image_annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
