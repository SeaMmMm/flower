import cv2
import numpy as np
from pathlib import Path
import os


def load_annotation(ann_path):
    boxes = []
    with open(ann_path, "r") as f:
        for line in f.readlines():
            class_label, x_center, y_center, width, height = map(float, line.split())
            boxes.append([class_label, x_center, y_center, width, height])
    return np.array(boxes, dtype=np.float32)


def save_annotation(ann_path, boxes):
    with open(ann_path, "w") as f:
        for box in boxes:
            class_label = int(box[0])  # 确保类别标签为整数
            # 将类别标签以外的坐标信息保留为浮点数
            box_info = " ".join(map(str, [class_label] + list(box[1:])))
            f.write(box_info + "\n")


def flip_image_and_boxes(image, boxes):
    flipped_image = cv2.flip(image, 1)
    new_boxes = boxes.copy()
    for i in range(len(boxes)):
        new_boxes[i][1] = 1 - boxes[i][1]  # 翻转 x_center
    return flipped_image, new_boxes


def is_box_inside_image(box, img_width, img_height):
    _, x_center, y_center, width, height = box
    # 转换为绝对坐标以便更容易进行边界检查
    x1 = max(min((x_center - width / 2), 1.0), 0)
    y1 = max(min((y_center - height / 2), 1.0), 0)
    x2 = max(min((x_center + width / 2), 1.0), 0)
    y2 = max(min((y_center + height / 2), 1.0), 0)
    # 如果框具有非零面积，则认为框在图像内部
    return (x2 > x1) and (y2 > y1)


def adjust_and_filter_boxes(boxes, crop_x, crop_y, crop_w, crop_h, orig_w, orig_h):
    new_boxes = []
    for box in boxes:
        class_label, x_center, y_center, width, height = box
        # 调整坐标到裁剪区域
        x_center = (x_center * orig_w - crop_x) / crop_w
        y_center = (y_center * orig_h - crop_y) / crop_h
        width /= crop_w / orig_w
        height /= crop_h / orig_h
        # 检查调整后的框是否在裁剪后的图像内部
        if is_box_inside_image([class_label, x_center, y_center, width, height], 1, 1):
            new_boxes.append([class_label, x_center, y_center, width, height])
    return np.array(new_boxes, dtype=np.float32)


def random_crop(image, boxes):
    h, w, _ = image.shape
    crop_x = np.random.randint(0, w // 4)
    crop_y = np.random.randint(0, h // 4)
    crop_w = np.random.randint(3 * w // 4, w)
    crop_h = np.random.randint(3 * h // 4, h)

    cropped_image = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    new_boxes = adjust_and_filter_boxes(boxes, crop_x, crop_y, crop_w, crop_h, w, h)

    return cropped_image, new_boxes


# 设置路径
dataset_base = "datasets/flower"
images_dir = Path(dataset_base) / "images/train"
labels_dir = Path(dataset_base) / "labels/train"
aug_images_dir = Path(dataset_base) / "images/train_aug"
aug_labels_dir = Path(dataset_base) / "labels/train_aug"

# 创建用于增强图像和标签的目录
aug_images_dir.mkdir(parents=True, exist_ok=True)
aug_labels_dir.mkdir(parents=True, exist_ok=True)

for image_path in images_dir.glob("*.jpg"):
    image = cv2.imread(str(image_path))
    ann_path = labels_dir / (image_path.stem + ".txt")
    boxes = load_annotation(ann_path)

    # 应用随机裁剪增强
    cropped_image, cropped_boxes = random_crop(image, boxes)
    aug_image_path = aug_images_dir / (image_path.stem + "_cropped.jpg")
    aug_ann_path = aug_labels_dir / (image_path.stem + "_cropped.txt")
    cv2.imwrite(str(aug_image_path), cropped_image)
    save_annotation(aug_ann_path, cropped_boxes)

    # 应用翻转增强
    flipped_image, flipped_boxes = flip_image_and_boxes(image, boxes)
    aug_image_path = aug_images_dir / (image_path.stem + "_flipped.jpg")
    aug_ann_path = aug_labels_dir / (image_path.stem + "_flipped.txt")
    cv2.imwrite(str(aug_image_path), flipped_image)
    save_annotation(aug_ann_path, flipped_boxes)

    print(f"处理完成 {image_path.name}")
