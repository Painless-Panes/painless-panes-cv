from ultralytics import YOLO

model = YOLO("window-model-nano.pt")

filename = "VN5b_loh2zYOww=="
results = model(
    f"test/{filename}.jpg",
    show=True,
    conf=0.2,
    save=True,
    project=".",
)
