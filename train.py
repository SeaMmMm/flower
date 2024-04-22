import yaml
from ultralytics import YOLO


# 加载自定义超参数
def load_hyperparameters(path):
    with open(path) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    return hyp


# 应用超参数到模型
def apply_hyperparameters(model, hyp):
    for k, v in hyp.items():
        setattr(model, k, v)


hyp_path = "/Users/smc/Downloads/毕设/flower/hyps/hyp.scratch.yaml"
hyp = load_hyperparameters(hyp_path)


# https://docs.ultralytics.com/tasks/detect/
# 1.
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

apply_hyperparameters(model, hyp)

# Train the model
results = model.train(data="flower.yaml", epochs=100, imgsz=1024, batch=1)


# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(data='flower.yaml', epochs=100, imgsz=1024, batch = 8)
