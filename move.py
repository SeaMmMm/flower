from pathlib import Path
import shutil

# 设置源文件夹和目标文件夹的路径
images_aug_dir_path = Path("datasets/flower/images/train_aug")
labels_aug_dir_path = Path("datasets/flower/labels/train_aug")

images_train_dir_path = Path("datasets/flower/images/train")
labels_train_dir_path = Path("datasets/flower/labels/train")

# 移动增强后的图像文件到原始的train文件夹
for aug_image_path in images_aug_dir_path.glob("*"):
    target_image_path = images_train_dir_path / aug_image_path.name
    shutil.move(str(aug_image_path), str(target_image_path))
    print(f"Moved augmented image to {target_image_path}")

# 移动增强后的标注文件到原始的train文件夹
for aug_label_path in labels_aug_dir_path.glob("*"):
    target_label_path = labels_train_dir_path / aug_label_path.name
    shutil.move(str(aug_label_path), str(target_label_path))
    print(f"Moved augmented label to {target_label_path}")
