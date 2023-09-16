from ultralytics import YOLO

model = YOLO('window_model.pt')

results = model('test/oooPIIHMhyD4yQ==.jpg', show=True)