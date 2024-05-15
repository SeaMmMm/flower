import yaml
import pandas as pd

# YAML配置文件
config = """
lr0: 0.009
lrf: 0.2
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 2.0
warmup_momentum: 0.8
warmup_bias_lr: 0.2
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.25
anchor_t: 4.0
mosaic: 1.0
mixup: 0.5
hsv_h: 0.3
hsv_s: 0.9
hsv_v: 0.6
degrees: 45
translate: 0.3
scale: 0.5
shear: 20
perspective: 0.1
flipud: 0.5
fliplr: 0.76
mosaic_scale: (0.5, 1.5)
mixup_scale: (0.5, 1.5)
copy_paste: 0.5
erasing: 0.5
"""

# 将YAML配置文件转换为Python字典
config_dict = yaml.safe_load(config)

# 将字典转换为DataFrame
df = pd.DataFrame(list(config_dict.items()), columns=["Parameter", "Value"])

# 将DataFrame保存为Excel文件
df.to_excel("config.xlsx", index=False)
