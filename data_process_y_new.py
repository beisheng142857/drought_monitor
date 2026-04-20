import argparse
import os
from typing import List

import numpy as np
import torch
from sklearn.cluster import KMeans

N_CLUSTERS = 4
VALID_NDVI_THRESHOLD = 0.05
RANDOM_STATE = 42
N_INIT = 10
NDVI_LIGHT_THRESHOLD = 0.60
NDVI_MODERATE_THRESHOLD = 0.40
NDVI_SEVERE_THRESHOLD = 0.20
VV_HIGH_THRESHOLD = -10.0
VV_MID_THRESHOLD = -13.0
VH_HIGH_THRESHOLD = -16.0
VH_MID_THRESHOLD = -19.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='构建 forecasting V2 按月份展开的时间序列 Y_tensor')
    parser.add_argument('--input_dir', type=str, default='/root/autodl-tmp/zyk_drought_monitor/data_V2')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/zyk_drought_monitor/data_V2')
    parser.add_argument('--years', nargs='+', type=int, default=[2025]) #2021, 2022, 2023, 2024, 2025
    parser.add_argument('--input_prefix', type=str, default='sequence_X')
    parser.add_argument('--output_prefix', type=str, default='sequence_Y_threshold')
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--n_clusters', type=int, default=N_CLUSTERS)
    return parser.parse_args()


def generate_threshold_labels_for_month(x_month: np.ndarray) -> np.ndarray:
    ndvi = x_month[:, 0, :, :]
    vv = x_month[:, 1, :, :]
    vh = x_month[:, 2, :, :]

    valid_mask = np.isfinite(ndvi) & np.isfinite(vv) & np.isfinite(vh) & (ndvi > VALID_NDVI_THRESHOLD)
    if not np.any(valid_mask):
        raise ValueError('没有找到可用于生成 threshold 标签的有效像元。')

    batch, height, width = ndvi.shape
    y_array = np.zeros((batch, height, width), dtype=np.int64)
    light_mask = valid_mask & ((ndvi < NDVI_LIGHT_THRESHOLD) | (vv < VV_HIGH_THRESHOLD) | (vh < VH_HIGH_THRESHOLD))
    moderate_mask = valid_mask & ((ndvi < NDVI_MODERATE_THRESHOLD) | (vv < VV_MID_THRESHOLD) | (vh < VH_MID_THRESHOLD))
    severe_mask = valid_mask & ((ndvi < NDVI_SEVERE_THRESHOLD) & (vv < VV_MID_THRESHOLD) & (vh < VH_MID_THRESHOLD))

    y_array[light_mask] = 1
    y_array[moderate_mask] = 2
    y_array[severe_mask] = 3
    return y_array


def generate_kmeans_labels_for_month(x_month: np.ndarray, n_clusters: int) -> np.ndarray:
    ndvi = x_month[:, 0, :, :]
    vv = x_month[:, 1, :, :]
    vh = x_month[:, 2, :, :]

    valid_mask = np.isfinite(ndvi) & np.isfinite(vv) & np.isfinite(vh) & (ndvi > VALID_NDVI_THRESHOLD)
    if not np.any(valid_mask):
        raise ValueError('没有找到可用于生成 kmeans 标签的有效像元。')

    batch, height, width = ndvi.shape
    features = np.stack([ndvi[valid_mask], vv[valid_mask], vh[valid_mask]], axis=1)
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
    return y_array


def generate_sequence_labels(x_tensor: torch.Tensor, label_mode: str, n_clusters: int) -> torch.Tensor:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 形状应为 (Batch, Time, Channels, H, W)，当前为 {x_tensor.shape}')
    if x_tensor.shape[2] < 3:
        raise ValueError('当前标签构建至少需要 3 个通道：NDVI、VV、VH。')

    x_np = x_tensor.cpu().numpy()
    time_steps = x_np.shape[1]
    month_labels: List[np.ndarray] = []

    for month_idx in range(time_steps):
        print(f'正在生成第 {month_idx} 个时间步对应的标签...')
        x_month = x_np[:, month_idx, :, :, :]
        if label_mode == 'threshold':
            y_month = generate_threshold_labels_for_month(x_month)
        else:
            y_month = generate_kmeans_labels_for_month(x_month, n_clusters)
        month_labels.append(y_month)

    y_sequence = np.stack(month_labels, axis=1)
    y_tensor = torch.from_numpy(y_sequence)
    print(f'生成的时间序列标签形状 (Batch, Time, H, W): {tuple(y_tensor.shape)}')
    print(f'整体标签分布: {torch.bincount(y_tensor.flatten(), minlength=n_clusters if label_mode == "kmeans" else 4)}')
    return y_tensor


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for year in args.years:
        x_path = os.path.join(args.input_dir, f'{args.input_prefix}_{year}.pt')
        if not os.path.exists(x_path):
            raise FileNotFoundError(f'未找到输入张量: {x_path}')

        print('=' * 80)
        print(f'开始构建 {year} 年 forecasting V2 时间序列标签')
        x_tensor = torch.load(x_path, map_location='cpu')
        y_tensor = generate_sequence_labels(x_tensor, args.label_mode, args.n_clusters)

        if args.label_mode == 'threshold':
            output_name = f'{args.output_prefix}_{year}.pt'
        else:
            output_name = f'sequence_Y_kmeans_{year}.pt'

        output_path = os.path.join(args.output_dir, output_name)
        torch.save(y_tensor, output_path)
        print(f'已保存: {output_path}')


if __name__ == '__main__':
    main()
