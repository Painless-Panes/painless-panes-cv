"""Employs YOLO v8 model using ONNX runtime

See https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py
and https://stackoverflow.com/a/77078624
"""
import cv2
import numpy

from painless_panes import util

MODEL_FILE_PATH = util.file_path("painless_panes.model", "model.onnx")
CLASSES = util.file_contents("painless_panes.model", "classes.txt").splitlines()


def draw_bounding_box(img, class_id, confidence, x0, y0, x1, y1):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = (0, 0, 255)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    cv2.putText(img, label, (x0 - 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_detections(image):
    """_summary_

    Pre- and post-processing code copied from
    https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py

    :param original_image: _description_
    :type original_image: _type_
    :return: _description_
    :rtype: _type_
    """
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(MODEL_FILE_PATH)
    height, width, _ = image.shape
    length = max((height, width))
    normalized_image = numpy.zeros((length, length, 3), numpy.uint8)
    normalized_image[0:height, 0:width] = image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(
        normalized_image, scalefactor=1 / 255, size=(640, 640), swapRB=True
    )
    model.setInput(blob)
    outputs = model.forward()

    outputs = numpy.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for idx in range(rows):
        classes_scores = outputs[0][idx][4:]
        (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            nxywh = [
                outputs[0][idx][0] - (0.5 * outputs[0][idx][2]),
                outputs[0][idx][1] - (0.5 * outputs[0][idx][3]),
                outputs[0][idx][2],
                outputs[0][idx][3],
            ]
            boxes.append(nxywh)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    for idx in range(len(result_boxes)):
        index = result_boxes[idx]
        nxywh = boxes[index]
        x0 = int(nxywh[0] * scale)
        y0 = int(nxywh[1] * scale)
        x1 = int((nxywh[0] + nxywh[2]) * scale)
        y1 = int((nxywh[1] + nxywh[3]) * scale)
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "bounding_box": (x0, y0, x1, y1),
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            image,
            class_ids[index],
            scores[index],
            x0,
            y0,
            x1,
            y1,
        )

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections
