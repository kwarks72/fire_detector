from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg")
results[0].save("result.jpg")
print("완료! result.jpg를 열어보세요")