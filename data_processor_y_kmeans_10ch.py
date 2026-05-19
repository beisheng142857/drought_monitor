import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

base_dir = '/root/autodl-tmp/zyk_drought_monitor/data'
output_dir = os.path.join(base_dir, 'data_proc')
os.makedirs(output_dir, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025]
N_CLUSTERS = 4
VALID_NDVI_THRESHOLD = 0.05
RANDOM_STATE = 42
N_INIT = 10
FEATURE_ORDER = [
    'NDVI', 'EVI', 'NDMI', 'NDWI', 'MSAVI',
    'VV', 'VH', 'VVVH', 'VVDIFFVH', 'RVI',
]
CHANNEL_MAP = {name: idx for idx, name in enumerate(FEATURE_ORDER)}
KMEANS_FEATURES = ['NDVI', 'EVI', 'NDMI', 'NDWI', 'MSAVI', 'VV', 'VH', 'RVI']


def generate_kmeans_pseudo_labels(x_tensor: torch.Tensor, n_clusters: int = N_CLUSTERS) -> torch.Tensor:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 形状应为 (Batch, Time, Channels, H, W)，当前维度为 {x_tensor.ndim}')

    required_max_index = max(CHANNEL_MAP[channel] for channel in KMEANS_FEATURES)
    if x_tensor.shape[2] <= required_max_index:
        raise ValueError(
            f'当前 K-Means 标签构建需要通道 {KMEANS_FEATURES}，但 X_tensor 仅有 {x_tensor.shape[2]} 个通道。'
        )

    print('正在基于最后一个月的 10 通道多源特征生成 K-Means 伪标签...')
    last_month_features = x_tensor[:, -1, :, :, :].cpu().numpy()
    ndvi = last_month_features[:, CHANNEL_MAP['NDVI'], :, :]

    valid_mask = np.isfinite(ndvi) & (ndvi > VALID_NDVI_THRESHOLD)
    for feature_name in KMEANS_FEATURES:
        feature = last_month_features[:, CHANNEL_MAP[feature_name], :, :]
        valid_mask &= np.isfinite(feature)

    if not np.any(valid_mask):
        raise ValueError('没有找到可用于聚类的有效像元，请检查 X_tensor 数值范围与 NDVI 阈值。')

    batch, height, width = ndvi.shape
    raw_features = np.stack(
        [last_month_features[:, CHANNEL_MAP[name], :, :][valid_mask] for name in KMEANS_FEATURES],
        axis=1,
    )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_features)

    print(f'执行多特征 K-Means 聚类，有效样本数: {scaled_features.shape[0]}，特征维度: {scaled_features.shape[1]}')
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = kmeans.fit_predict(scaled_features)

    ndvi_index = KMEANS_FEATURES.index('NDVI')
    cluster_ndvi_means = np.array([
        raw_features[labels == idx, ndvi_index].mean() if np.any(labels == idx) else -np.inf
        for idx in range(n_clusters)
    ])
    rank = np.argsort(cluster_ndvi_means)[::-1]
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(rank)}
    ordered_labels = np.vectorize(label_map.get)(labels).astype(np.int64)

    y_array = np.zeros((batch, height, width), dtype=np.int64)
    y_array[valid_mask] = ordered_labels

    y_tensor = torch.from_numpy(y_array)
    print(f'Y_tensor 生成完毕，形状: {y_tensor.shape}')
    print(f'标签类别分布: {torch.bincount(y_tensor.flatten(), minlength=n_clusters)}')
    return y_tensor


if __name__ == '__main__':
    for year in YEARS:
        x_path = os.path.join(base_dir, f'dataset_X_{year}_new.pt')
        y_path = os.path.join(output_dir, f'dataset_Y_kmeans_{year}_10ch.pt')

        print(f'\n===== 开始生成 {year} 年 K-Means 伪标签 =====')
        print('正在加载 X_tensor...')
        x_tensor = torch.load(x_path, map_location='cpu')
        print(f'X_tensor 形状: {x_tensor.shape}')

        y_tensor = generate_kmeans_pseudo_labels(x_tensor)

        torch.save(y_tensor, y_path)
        print(f'{year} 年 K-Means 伪标签已保存至: {y_path}')
