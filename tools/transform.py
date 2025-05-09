from ultralytics import YOLO

# Load a model
model = YOLO("../yolov5/yolov5s.pt")  # load an official model

# Export the model
model.export(format="onnx")

