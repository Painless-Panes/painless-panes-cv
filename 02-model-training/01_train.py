from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='annotated_data.yaml',
    epochs=200,
    imgsz=640,
    project='./runs'
)