from ultralytics import YOLO

# Load a model
# 模型权重
model = YOLO('runs/detect/train3/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# 数据集路径
results = model('my-upload-service/uploads')  # return a list of Results objects


# Process results list
for i,result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename=f'my-upload-service/downloads/{i}.jpg')  # save to disk