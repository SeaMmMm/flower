# from ultralytics import YOLO
# import cv2

# # Load a model
# # 模型权重
# model = YOLO("runs/detect/train4/weights/best.pt")  # pretrained YOLOv8n model

# # Run batched inference on a list of images
# # 数据集路径
# results = model("my-upload-service/uploads")  # return a list of Results objects


# # classes = ["WT", "T1-C5-C1", "T1-C5-E5"]
# # Process results list
# for i, result in enumerate(results):
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     # for label in enumerate(boxes.cls):
#     #     print(classes[int(label[1].item())])

#     # result.show()  # display to screen
#     result.save(filename=f"my-upload-service/downloads/{i}.jpg")  # save to disk


from ultralytics import YOLO
from collections import defaultdict
import os
import cv2

# Load a model
model = YOLO("runs/detect/train6/weights/best.pt")  # pretrained YOLOv8n model

# Define colors for each class
class_names = ["WT", "T1-C5-C1", "T1-C5-E5"]  # 类别名称
class_colors = [(0, 0, 255), (255, 0, 0), (255, 0, 255)]  # 每个类别对应的颜色

# Find a file in the uploads directory
uploads_dir = "my-upload-service/uploads"
image_files = os.listdir(uploads_dir)
image_path = os.path.join(
    uploads_dir, image_files[0]
)  # Use the first image in the directory

# Run batched inference on a list of images
results = model("my-upload-service/uploads")  # return a list of Results objects

# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # 检测到的边界框坐标和类别概率
    labels = boxes.cls  # 获取类别索引
    image = cv2.imread(image_path)  # 读取原始图片

    # Draw bounding boxes on the image
    for j, box in enumerate(boxes.xyxy):
        box_values = box.int().tolist()  # 边界框坐标
        # 解包边界框的值
        x1, y1, x2, y2 = box_values[:4]

        # Draw bounding box with class color
        class_color = class_colors[int(labels[j].item())]  # 使用颜色列表中的颜色
        cv2.rectangle(image, (x1, y1), (x2, y2), class_color, 2)  # 使用颜色绘制边界框
        for idx, (name, color) in enumerate(zip(class_names, class_colors)):
            cv2.putText(
                image,
                f"{name}",
                (10, 30 * (idx + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )  # 添加类别名称

    # Save the annotated image
    cv2.imwrite(f"my-upload-service/downloads/0.jpg", image)  # 保存带有标注的图片


# Define a dictionary to store the counts
class_counts = {}

# Process results list
for j, result in enumerate(results):
    boxes = result.boxes  # 检测到的边界框坐标和类别概率
    labels = boxes.cls  # 获取类别索引

    for label in enumerate(labels):
        class_name = class_names[int(label[1].item())]  # 获取类别名称
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        class_counts["total"] = class_counts.get("total", 0) + 1

print(class_counts)
print("succeed!")
