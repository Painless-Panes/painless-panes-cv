from ultralytics import YOLO

model = YOLO("custom-model.pt")

filename = "example2"
results = model(
    f"test/{filename}.jpg",
    show=True,
    conf=0.1,
    save=True,
    project=".",
)
