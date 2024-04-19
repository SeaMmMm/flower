from ultralytics import YOLO

# https://docs.ultralytics.com/tasks/detect/
# 1.
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="flower.yaml", epochs=100, imgsz=1024, batch=1)


# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(data='flower.yaml', epochs=100, imgsz=1024, batch = 8)
