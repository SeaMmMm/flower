from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
from ultralytics import YOLO
from collections import defaultdict
import os
import cv2


app = Flask(__name__)
CORS(app)

# 配置上传文件夹和允许的扩展名
UPLOAD_FOLDER = "/Users/smc/Downloads/毕设/flower/my-upload-service/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tif"}

# Define colors for each class
class_names = ["WT", "T1-C5-C1", "T1-C5-E5"]  # 类别名称
class_colors = [(0, 0, 255), (255, 0, 0), (255, 0, 255)]  # 每个类别对应的颜色


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("runs/detect/train4/weights/best.pt")  # pretrained YOLOv8n model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_images(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 检查文件是否为图片格式（这里假设支持的图片格式为'.jpg', '.jpeg', '.png', '.gif', '.tif）
        if file_path.endswith((".jpg", ".jpeg", ".png", ".gif", "tif")):
            # 删除图片文件
            os.remove(file_path)


@app.route("/upload", methods=["POST"])
def upload_file():
    clear_images("/Users/smc/Downloads/毕设/flower/my-upload-service/uploads")
    # 检查是否有文件在请求中
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    # 如果用户没有选择文件，浏览器也会提交一个空的文件无文件名。
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        file.save(
            "/Users/smc/Downloads/毕设/flower/my-upload-service/uploads/"
            + secure_filename(file.filename)
        )

        # Find a file in the uploads directory
        uploads_dir = "my-upload-service/uploads"
        image_files = os.listdir(uploads_dir)
        print(image_files)
        image_path = os.path.join(
            uploads_dir, image_files[0]
        )  # Use the first image in the directory

        results = model(
            "/Users/smc/Downloads/毕设/flower/my-upload-service/uploads"
        )  # return a list of Results objects

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
                class_color = class_colors[
                    int(labels[j].item())
                ]  # 使用颜色列表中的颜色
                cv2.rectangle(
                    image, (x1, y1), (x2, y2), class_color, 2
                )  # 使用颜色绘制边界框
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
            cv2.imwrite(
                f"my-upload-service/downloads/0.jpg", image
            )  # 保存带有标注的图片

        return jsonify(
            {"message": "File uploaded successfully", "filename": file.filename}
        )

    return jsonify({"error": "File not allowed"})


@app.route("/downloads/<filename>", methods=["GET"])
def download_file(filename):
    return send_file(
        "/Users/smc/Downloads/毕设/flower/my-upload-service/downloads/0.jpg",
        as_attachment=True,
    )


@app.route("/data", methods=["GET"])
def get_data():
    class_counts = {}
    results = model(
        "/Users/smc/Downloads/毕设/flower/my-upload-service/uploads"
    )  # return a list of Results objects
    # Process results list
    for j, result in enumerate(results):
        boxes = result.boxes  # 检测到的边界框坐标和类别概率
        labels = boxes.cls  # 获取类别索引

        for label in enumerate(labels):
            class_name = class_names[int(label[1].item())]  # 获取类别名称
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            # class_counts["total"] = class_counts.get("total", 0) + 1

    return jsonify(class_counts)


if __name__ == "__main__":
    # 确保上传文件夹存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
