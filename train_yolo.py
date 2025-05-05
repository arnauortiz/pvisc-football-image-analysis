from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="datasets/football-players-detection-1/data.yaml", epochs=50, imgsz=640
)
