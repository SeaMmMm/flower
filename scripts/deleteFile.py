from pathlib import Path
import os

# 指定需要清理的文件夹路径
folder_path = 'datasets/flower/labels/train'

# 使用Path对象遍历指定文件夹
for file_path in Path(folder_path).iterdir():
    # 检查文件名是否仅由数字组成
    if not file_path.stem.isdigit():
        # 如果不是，则删除该文件
        os.remove(file_path)
        print(f'Removed {file_path}')
