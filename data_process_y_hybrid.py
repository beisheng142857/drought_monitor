import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

N_CLUSTERS = 4
RANDOM_STATE = 42
N_INIT = 10
VALID_NDVI_THRESHOLD = 0.05

# 基础阈值规则
NDVI_LIGHT_THRESHOLD = 0.60
NDVI_MODERATE_THRESHOLD = 0.40
NDVI_SEVERE_THRESHOLD = 0.20
VV_HIGH_THRESHOLD = -10.0
VV_MID_THRESHOLD = -13.0
VH_HIGH_THRESHOLD = -16.0
VH_MID_THRESHOLD = -19.0

# 辅助门控规则
NDWI_WATER_MASK_THRESHOLD = 0.20
NDMI_DRY_SUPPORT_THRESHOLD = 0.00
MSAVI_DRY_SUPPORT_THRESHOLD = 0.30

# 与 gee_downloader.py / config.py 中的默认导出顺序保持一致
FEATURE_ORDER = [
    'NDVI', 'EVI', 'NDMI', 'NDWI', 'MSAVI',
    'VV', 'VH', 'VVVH', 'VVDIFFVH', 'RVI'
]
CHANNEL_MAP = {name: idx for idx, name in enumerate(FEATURE_ORDER)}
KMEANS_FEATURES = ['NDVI', 'EVI', 'NDMI', 'NDWI', 'MSAVI', 'VV', 'VH', 'RVI']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='构建 forecasting V2 hybrid 时间序列 Y_tensor')
    parser.add_argument('--input_dir', type=str, default='/root/autodl-tmp/zyk_drought_monitor/data_V2')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/zyk_drought_monitor/data_V2')
    parser.add_argument('--years', nargs='+', type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument('--input_prefix', type=str, default='sequence_X')
    parser.add_argument('--output_prefix', type=str, default='sequence_Y_hybrid')
    parser.add_argument('--n_clusters', type=int, default=N_CLUSTERS)
    parser.add_argument('--promote_threshold0_min_cluster', type=int, default=2)
    parser.add_argument('--promote_threshold1_min_cluster', type=int, default=2)
    return parser.parse_args()


def validate_x_tensor(x_tensor: torch.Tensor) -> None:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 形状应为 (Batch, Time, Channels, H, W)，当前为 {x_tensor.shape}')

    required_features = ['NDVI', 'EVI', 'NDMI', 'NDWI', 'MSAVI', 'VV', 'VH', 'RVI']
    required_max_index = max(CHANNEL_MAP[name] for name in required_features)
    if x_tensor.shape[2] <= required_max_index:
        raise ValueError(
            f'当前 hybrid 标签构建需要通道 {required_features}，'
            f'按默认特征顺序应包含 {len(FEATURE_ORDER)} 个通道，当前仅有 {x_tensor.shape[2]} 个通道。'
        )


def build_threshold_components(x_month: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ndvi = x_month[:, CHANNEL_MAP['NDVI'], :, :]
    vv = x_month[:, CHANNEL_MAP['VV'], :, :]
    vh = x_month[:, CHANNEL_MAP['VH'], :, :]
    ndwi = x_month[:, CHANNEL_MAP['NDWI'], :, :]
    ndmi = x_month[:, CHANNEL_MAP['NDMI'], :, :]
    msavi = x_month[:, CHANNEL_MAP['MSAVI'], :, :]

    valid_mask = np.isfinite(ndvi) & np.isfinite(vv) & np.isfinite(vh) & (ndvi > VALID_NDVI_THRESHOLD)
    aux_finite = np.isfinite(ndwi) & np.isfinite(ndmi) & np.isfinite(msavi)
    non_water_mask = aux_finite & (ndwi < NDWI_WATER_MASK_THRESHOLD)
    rule_base_mask = valid_mask & non_water_mask

    dry_support_mask = (ndmi < NDMI_DRY_SUPPORT_THRESHOLD) | (msavi < MSAVI_DRY_SUPPORT_THRESHOLD)
    severe_support_mask = (ndmi < NDMI_DRY_SUPPORT_THRESHOLD) & (msavi < MSAVI_DRY_SUPPORT_THRESHOLD)

    y_threshold = np.zeros(ndvi.shape, dtype=np.int64)

    light_mask = rule_base_mask & (
        (ndvi < NDVI_LIGHT_THRESHOLD) |
        (vv < VV_HIGH_THRESHOLD) |
        (vh < VH_HIGH_THRESHOLD)
    )
    moderate_mask = rule_base_mask & dry_support_mask & (
        (ndvi < NDVI_MODERATE_THRESHOLD) |
        (vv < VV_MID_THRESHOLD) |
        (vh < VH_MID_THRESHOLD)
    )
    severe_mask = rule_base_mask & severe_support_mask & (
        (ndvi < NDVI_SEVERE_THRESHOLD) &
        (vv < VV_MID_THRESHOLD) &
        (vh < VH_MID_THRESHOLD)
    )

    y_threshold[light_mask] = 1
    y_threshold[moderate_mask] = 2
    y_threshold[severe_mask] = 3
    return y_threshold, valid_mask, rule_base_mask, dry_support_mask, severe_support_mask


def generate_kmeans_labels_for_month(x_month: np.ndarray, n_clusters: int) -> np.ndarray:
    ndvi = x_month[:, CHANNEL_MAP['NDVI'], :, :]

    valid_mask = np.isfinite(ndvi) & (ndvi > VALID_NDVI_THRESHOLD)
    for feature_name in KMEANS_FEATURES:
        feature = x_month[:, CHANNEL_MAP[feature_name], :, :]
        valid_mask &= np.isfinite(feature)

    if not np.any(valid_mask):
        raise ValueError('没有找到可用于生成 kmeans 标签的有效像元。')

    height, width = ndvi.shape[1], ndvi.shape[2]
    raw_features = np.stack(
        [x_month[:, CHANNEL_MAP[name], :, :][valid_mask] for name in KMEANS_FEATURES],
        axis=1,
    )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = kmeans.fit_predict(scaled_features)

    ndvi_index = KMEANS_FEATURES.index('NDVI')
    cluster_ndvi_means = np.array([
        raw_features[labels == idx, ndvi_index].mean() if np.any(labels == idx) else -np.inf
        for idx in range(n_clusters)
    ])

    # NDVI 均值越高，越接近 0（无旱）；越低，越接近 3（重旱）
    rank = np.argsort(cluster_ndvi_means)[::-1]
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(rank)}
    ordered_labels = np.vectorize(label_map.get)(labels).astype(np.int64)

    y_kmeans = np.zeros(ndvi.shape, dtype=np.int64)
    y_kmeans[valid_mask] = ordered_labels
    return y_kmeans


def fuse_labels_for_month(
    y_threshold: np.ndarray,
    y_kmeans: np.ndarray,
    valid_mask: np.ndarray,
    rule_base_mask: np.ndarray,
    dry_support_mask: np.ndarray,
    severe_support_mask: np.ndarray,
    promote_threshold0_min_cluster: int,
    promote_threshold1_min_cluster: int,
) -> np.ndarray:
    y_hybrid = y_threshold.copy()

    # 两者一致时直接采用；阈值法默认作为主标签
    agreement_mask = valid_mask & (y_threshold == y_kmeans)
    y_hybrid[agreement_mask] = y_threshold[agreement_mask]

    # 阈值法判为无旱，但聚类显示有较明显干旱时，只在辅助干旱证据存在时提升
    promote_from_0_mask = (
        rule_base_mask &
        (y_threshold == 0) &
        (y_kmeans >= promote_threshold0_min_cluster) &
        dry_support_mask
    )
    y_hybrid[promote_from_0_mask] = y_kmeans[promote_from_0_mask]

    # 阈值法判为轻旱，聚类判得更重时，可在干旱支撑存在时升级
    promote_from_1_mask = (
        rule_base_mask &
        (y_threshold == 1) &
        (y_kmeans >= promote_threshold1_min_cluster) &
        dry_support_mask
    )
    y_hybrid[promote_from_1_mask] = y_kmeans[promote_from_1_mask]

    # 阈值法判为中旱，若聚类也指向重旱且强支撑成立，则提升到重旱
    promote_2_to_3_mask = (
        rule_base_mask &
        (y_threshold == 2) &
        (y_kmeans == 3) &
        severe_support_mask
    )
    y_hybrid[promote_2_to_3_mask] = 3

    # 阈值法已判为重旱时直接保留，避免聚类把高置信重旱降级
    severe_keep_mask = valid_mask & (y_threshold == 3)
    y_hybrid[severe_keep_mask] = 3

    # 水体、辅助信息缺失或无效区域统一置 0，保持与原脚本风格一致
    invalid_or_unsupported_mask = ~valid_mask | ~rule_base_mask
    y_hybrid[invalid_or_unsupported_mask] = 0

    return y_hybrid


def generate_hybrid_labels_for_month(
    x_month: np.ndarray,
    n_clusters: int,
    promote_threshold0_min_cluster: int,
    promote_threshold1_min_cluster: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_threshold, valid_mask, rule_base_mask, dry_support_mask, severe_support_mask = build_threshold_components(x_month)
    y_kmeans = generate_kmeans_labels_for_month(x_month, n_clusters)
    y_hybrid = fuse_labels_for_month(
        y_threshold=y_threshold,
        y_kmeans=y_kmeans,
        valid_mask=valid_mask,
        rule_base_mask=rule_base_mask,
        dry_support_mask=dry_support_mask,
        severe_support_mask=severe_support_mask,
        promote_threshold0_min_cluster=promote_threshold0_min_cluster,
        promote_threshold1_min_cluster=promote_threshold1_min_cluster,
    )
    return y_hybrid, y_threshold, y_kmeans


def generate_sequence_hybrid_labels(
    x_tensor: torch.Tensor,
    n_clusters: int,
    promote_threshold0_min_cluster: int,
    promote_threshold1_min_cluster: int,
) -> torch.Tensor:
    validate_x_tensor(x_tensor)

    x_np = x_tensor.cpu().numpy()
    time_steps = x_np.shape[1]
    hybrid_labels: List[np.ndarray] = []

    for month_idx in range(time_steps):
        print(f'正在生成第 {month_idx} 个时间步对应的 hybrid 标签...')
        x_month = x_np[:, month_idx, :, :, :]
        y_hybrid, y_threshold, y_kmeans = generate_hybrid_labels_for_month(
            x_month=x_month,
            n_clusters=n_clusters,
            promote_threshold0_min_cluster=promote_threshold0_min_cluster,
            promote_threshold1_min_cluster=promote_threshold1_min_cluster,
        )

        print(f'  threshold 分布: {np.bincount(y_threshold.flatten(), minlength=4)}')
        print(f'  kmeans 分布:    {np.bincount(y_kmeans.flatten(), minlength=4)}')
        print(f'  hybrid 分布:    {np.bincount(y_hybrid.flatten(), minlength=4)}')
        hybrid_labels.append(y_hybrid)

    y_sequence = np.stack(hybrid_labels, axis=1)
    y_tensor = torch.from_numpy(y_sequence.astype(np.int64))
    print(f'生成的 hybrid 时间序列标签形状 (Batch, Time, H, W): {tuple(y_tensor.shape)}')
    print(f'整体 hybrid 标签分布: {torch.bincount(y_tensor.flatten(), minlength=4)}')
    return y_tensor


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for year in args.years:
        x_path = os.path.join(args.input_dir, f'{args.input_prefix}_{year}.pt')
        if not os.path.exists(x_path):
            raise FileNotFoundError(f'未找到输入张量: {x_path}')

        print('=' * 80)
        print(f'开始构建 {year} 年 forecasting V2 hybrid 时间序列标签')
        x_tensor = torch.load(x_path, map_location='cpu')
        y_tensor = generate_sequence_hybrid_labels(
            x_tensor=x_tensor,
            n_clusters=args.n_clusters,
            promote_threshold0_min_cluster=args.promote_threshold0_min_cluster,
            promote_threshold1_min_cluster=args.promote_threshold1_min_cluster,
        )

        output_path = os.path.join(args.output_dir, f'{args.output_prefix}_{year}.pt')
        torch.save(y_tensor, output_path)
        print(f'已保存: {output_path}')


if __name__ == '__main__':
    main()
