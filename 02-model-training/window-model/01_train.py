from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # `n` for "nano", the smallest model

model.train(
    data='annotated_data.yaml',
    epochs=200,
    imgsz=640,
    project='./runs'
)