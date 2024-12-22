from ultralytics import YOLO
import torch
# Load a pretrained YOLO11n model
model = YOLO("yolov8m.pt")

# Run inference on an image
results = model("images.jpeg")  # list of 1 Results object
print(results[0].boxes.cls)
ts = results[0].boxes.cls
if torch.any(ts == 0):
    print("me")
results[0].show()