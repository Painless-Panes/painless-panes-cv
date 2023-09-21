from ultralytics import YOLO

model = YOLO("custom-model.pt")

model.export(format="tfjs")
