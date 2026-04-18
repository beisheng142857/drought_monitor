import torch
import numpy as np
import os

# 1. 挂载并设置路径 (请替换为你的真实路径)
base_dir = 'autodl-tmp/zyk_drought_monitor/data'
x_path = os.path.join(base_dir, 'dataset_X.pt')
y_path = os.path.join(base_dir, 'dataset_Y.pt')

# 2. 加载特征张量 X
# 形状预期: (Batch, Time, Channels, Height, Width) -> (N, 5, 4, 128, 128)
print("正在加载 X_tensor...")
X_tensor = torch.load(x_path)
print(f"X_tensor 形状: {X_tensor.shape}")

# 3. 提取评估时刻的特征 (通常用最后一个月，即索引 -1)
# 提取后的形状: (Batch, Channels, Height, Width)
last_month_features = X_tensor[:, -1, :, :, :]

# 假设通道 0 是 NDVI，通道 1 是 NDWI
ndvi = last_month_features[:, 0, :, :]
ndwi = last_month_features[:, 1, :, :]

# 4. 初始化伪标签张量 Y (全零，默认 0: 无旱)
# 形状: (Batch, Height, Width)，数据类型必须是 long 用于交叉熵分类
Y_tensor = torch.zeros((X_tensor.shape[0], 128, 128), dtype=torch.long)

# 5. 定义干旱等级的映射规则 (经验阈值，后续可根据文献或实际分布调整)
# 1: 轻度干旱 (Light)
mask_light = (ndvi < 0.6) & (ndwi < 0.2)
Y_tensor[mask_light] = 1

# 2: 中度干旱 (Moderate)
mask_moderate = (ndvi < 0.4) & (ndwi < 0.0)
Y_tensor[mask_moderate] = 2

# 3: 重度干旱 (Severe)
mask_severe = (ndvi < 0.2) & (ndwi < -0.1)
Y_tensor[mask_severe] = 3

# 处理无效区域: 比如 NDVI 完全为 0 或极低的地方可能是水体或建筑
mask_invalid = (ndvi <= 0.0)
Y_tensor[mask_invalid] = 0 # 视作无旱或忽略

print(f"Y_tensor 生成完毕！形状: {Y_tensor.shape}")
print(f"标签类别分布: {torch.bincount(Y_tensor.flatten())}")

# 6. 保存到云盘
torch.save(Y_tensor, y_path)
print(f"伪标签已成功保存至云盘: {y_path}")
