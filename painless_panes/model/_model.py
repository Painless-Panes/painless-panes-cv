"""Employs YOLO v8 model using ONNX runtime
"""
import cv2
import numpy

from painless_panes import util

MODEL_FILE_PATH = util.file_path("painless_panes.model", "model.onnx")
CLASSES = util.file_contents("painless_panes.model", "classes.txt").splitlines()


def get_detections(image, conf=0.25):
    """Find the YOLO model detections in an image

    Pre- and post-processing steps copied from
    https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py
    with modifications

    :param image: The image
    :type image: numpy.ndarray
    :param conf: Confidence threshold, defaults to 0.25
    :type conf: float, optional
    :returns: The list of detections, in the form:
        [
            {
                "class_name": <string>,   # The object class name
                "confidence": <float>,    # The % confidence in the detection
                "bounding_box": <tuple>,  # The bounding box in xyxy format
            },
            ...
        ]
    :rtype: List[dict]
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
    outputs = model.forward()[0].T

    boxes = []
    scores = []
    class_ids = []

    for output in outputs:
        classes_scores = output[4:]
        (_, max_score, _, (_, max_class_id)) = cv2.minMaxLoc(classes_scores)
        if max_score >= 0.25:
            # Bounding box in normalized x, y, w, h format
            nxywh = [
                output[0] - (0.5 * output[2]),
                output[1] - (0.5 * output[3]),
                output[2],
                output[3],
            ]
            boxes.append(nxywh)
            scores.append(max_score)
            class_ids.append(max_class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.45, 0.5)

    detections = []
    for index in indices:
        nxywh = boxes[index]
        x0 = int(nxywh[0] * scale)
        y0 = int(nxywh[1] * scale)
        x1 = int((nxywh[0] + nxywh[2]) * scale)
        y1 = int((nxywh[1] + nxywh[3]) * scale)

        # Bounding box in x, y, x, y (top left, bottom right corner) format
        bbox = (x0, y0, x1, y1)
        detection = {
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "bounding_box": bbox,
        }
        detections.append(detection)

    return detections
