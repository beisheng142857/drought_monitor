import os

import numpy as np
import torch
from sklearn.cluster import KMeans

# 1. 挂载并设置路径 (请替换为你的真实路径)
base_dir = '/root/autodl-tmp/data_proc'
output_dir = os.path.join(base_dir, 'data_proc')
os.makedirs(output_dir, exist_ok=True)
x_path = os.path.join(base_dir, 'dataset_X_2025.pt')
y_path = os.path.join(output_dir, 'dataset_Y_2025.pt')

N_CLUSTERS = 4
VALID_NDVI_THRESHOLD = 0.05
RANDOM_STATE = 42
N_INIT = 10


def generate_pseudo_labels(x_tensor: torch.Tensor, n_clusters: int = N_CLUSTERS) -> torch.Tensor:
    if x_tensor.ndim != 5:
        raise ValueError(f"X_tensor 形状应为 (Batch, Time, Channels, H, W)，当前维度为 {x_tensor.ndim}")

    if x_tensor.shape[2] < 3:
        raise ValueError("当前标签构建至少需要 3 个通道：NDVI、VV、VH。")

    print("正在基于最后一个月的多源特征生成伪标签...")
    last_month_features = x_tensor[:, -1, :, :, :].cpu().numpy()

    # 当前通道定义：0=NDVI, 1=VV, 2=VH, 3=VVVH
    ndvi = last_month_features[:, 0, :, :]
    vv = last_month_features[:, 1, :, :]
    vh = last_month_features[:, 2, :, :]

    valid_mask = np.isfinite(ndvi) & np.isfinite(vv) & np.isfinite(vh) & (ndvi > VALID_NDVI_THRESHOLD)
    if not np.any(valid_mask):
        raise ValueError("没有找到可用于聚类的有效像元，请检查 X_tensor 数值范围与 NDVI 阈值。")

    batch, height, width = ndvi.shape
    features = np.stack([ndvi[valid_mask], vv[valid_mask], vh[valid_mask]], axis=1)

    print(f"执行多源特征协同聚类 (NDVI + VV + VH)，有效样本数: {features.shape[0]}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = kmeans.fit_predict(features)

    cluster_ndvi_means = np.array([
        features[labels == idx, 0].mean() if np.any(labels == idx) else -np.inf
        for idx in range(n_clusters)
    ])
    rank = np.argsort(cluster_ndvi_means)[::-1]
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(rank)}
    ordered_labels = np.vectorize(label_map.get)(labels).astype(np.int64)

    y_array = np.zeros((batch, height, width), dtype=np.int64)
    y_array[valid_mask] = ordered_labels

    y_tensor = torch.from_numpy(y_array)
    print(f"Y_tensor 生成完毕！形状: {y_tensor.shape}")
    print(f"标签类别分布: {torch.bincount(y_tensor.flatten(), minlength=n_clusters)}")
    return y_tensor


# 2. 加载特征张量 X
# 形状预期: (Batch, Time, Channels, Height, Width) -> (N, 5, 4, 128, 128)
print("正在加载 X_tensor...")
X_tensor = torch.load(x_path)
print(f"X_tensor 形状: {X_tensor.shape}")

# 3. 生成伪标签张量 Y
Y_tensor = generate_pseudo_labels(X_tensor)

# 4. 保存到云盘
torch.save(Y_tensor, y_path)
print(f"伪标签已成功保存至云盘: {y_path}")
