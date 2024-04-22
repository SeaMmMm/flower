from pathlib import Path


def check_and_fix_labels(folder_path):
    for label_path in Path(folder_path).rglob("*.txt"):
        valid_lines = []  # 用于存储有效的标注行
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                # 将类别标签转换为整数，坐标值转换为浮点数
                class_label = int(parts[0])
                coordinates = list(map(float, parts[1:]))
                # 检查所有坐标值是否在[0, 1]范围内
                if all(0 <= n <= 1 for n in coordinates):
                    valid_lines.append(line)

        # 用仅包含有效标注的内容重写标注文件
        with open(label_path, "w") as f:
            f.writelines(valid_lines)


# 调用函数，'path/to/your/labels/folder' 替换成你的标注文件夹路径
check_and_fix_labels("datasets/flower/labels/train")
print("Done!")
