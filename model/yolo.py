from ultralytics import YOLO

model = YOLO('model/yolo11n.pt')
results = model("https://ultralytics.com/images/bus.jpg")

print(results)