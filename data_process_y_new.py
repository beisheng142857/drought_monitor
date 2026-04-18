import os

import numpy as np
import torch
from sklearn.cluster import KMeans

DATA_DIR = '/root/autodl-tmp/zyk_drought_monitor/data'
TARGET_YEAR = 2023
N_CLUSTERS = 4
VALID_NDVI_THRESHOLD = 0.05
RANDOM_STATE = 42
N_INIT = 10


def generate_pseudo_labels(
    x_tensor: torch.Tensor,
    n_clusters: int = N_CLUSTERS,
    valid_ndvi_threshold: float = VALID_NDVI_THRESHOLD,
) -> torch.Tensor:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 维度应为 5，但实际得到 {x_tensor.ndim}。')

    if x_tensor.shape[2] < 3:
        raise ValueError('当前伪标签构建至少需要 3 个通道：NDVI、VV、VH。')

    print('加载特征并按当前通道定义生成伪标签...')
    last_month = x_tensor[:, -1, :, :, :].cpu().numpy()

    ndvi = last_month[:, 0, :, :]
    vv = last_month[:, 1, :, :]
    vh = last_month[:, 2, :, :]

    valid_mask = np.isfinite(ndvi) & np.isfinite(vv) & np.isfinite(vh) & (ndvi > valid_ndvi_threshold)
    if not np.any(valid_mask):
        raise ValueError('没有找到可用于聚类的有效像元，请检查 X_tensor 数值范围与 NDVI 阈值。')

    batch, height, width = ndvi.shape
    features = np.stack([ndvi[valid_mask], vv[valid_mask], vh[valid_mask]], axis=1)

    print(f'执行多源特征协同聚类 (NDVI + VV + VH)，有效样本数: {features.shape[0]}')
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_INIT)
    cluster_labels = kmeans.fit_predict(features)

    cluster_ndvi_means = np.array([
        features[cluster_labels == idx, 0].mean() if np.any(cluster_labels == idx) else -np.inf
        for idx in range(n_clusters)
    ])
    rank = np.argsort(cluster_ndvi_means)[::-1]
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(rank)}
    ordered_labels = np.vectorize(label_map.get)(cluster_labels).astype(np.int64)

    y_array = np.zeros((batch, height, width), dtype=np.int64)
    y_array[valid_mask] = ordered_labels

    y_tensor = torch.from_numpy(y_array)
    print(f'Y_tensor 生成完毕，形状: {tuple(y_tensor.shape)}')
    print(f'标签类别分布: {torch.bincount(y_tensor.flatten(), minlength=n_clusters)}')
    return y_tensor


if __name__ == '__main__':
    x_path = os.path.join(DATA_DIR, f'dataset_X_{TARGET_YEAR}.pt')
    y_path = os.path.join(DATA_DIR, f'dataset_Y_{TARGET_YEAR}.pt')

    if not os.path.exists(x_path):
        raise FileNotFoundError(f'未找到输入张量: {x_path}')

    x_tensor = torch.load(x_path)
    y_tensor = generate_pseudo_labels(x_tensor)
    torch.save(y_tensor, y_path)
    print(f'Y_tensor 已保存至: {y_path}')
