from ultralytics import YOLO

model = YOLO("window_model.pt")

# filename = "q7kuAkPnuupHrA=="
filename = "uEv9z6sFD80aXQ=="
results = model(
    f"test/{filename}.jpg",
    show=True,
    conf=0.2,
    save=True,
    project=".",
)
