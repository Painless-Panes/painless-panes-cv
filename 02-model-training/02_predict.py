from ultralytics import YOLO

model = YOLO("custom-model.pt")

filename = "example"
results = model(
    f"test/{filename}.jpg",
    show=True,
    conf=0.2,
    save=True,
    project=".",
)
