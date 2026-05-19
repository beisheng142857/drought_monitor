import os

import numpy as np
import torch

# 1. 设置路径
base_dir = '/root/autodl-tmp/data_proc'
YEARS = [2021, 2022, 2023, 2024, 2025]

# 2. 硬阈值配置
NDVI_VALID_THRESHOLD = 0.05
NDVI_LIGHT_THRESHOLD = 0.60
NDVI_MODERATE_THRESHOLD = 0.40
NDVI_SEVERE_THRESHOLD = 0.20
VV_HIGH_THRESHOLD = -10.0
VV_MID_THRESHOLD = -13.0
VH_HIGH_THRESHOLD = -16.0
VH_MID_THRESHOLD = -19.0
CHANNEL_MAP = {
    'NDVI': 0,
    'EVI': 1,
    'NDMI': 2,
    'NDWI': 3,
    'MSAVI': 4,
    'VV': 5,
    'VH': 6,
    'VVVH': 7,
    'VVDIFFVH': 8,
    'RVI': 9,
}


def generate_threshold_labels(x_tensor: torch.Tensor) -> torch.Tensor:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 形状应为 (Batch, Time, Channels, H, W)，当前维度为 {x_tensor.ndim}')

    required_channels = ['NDVI', 'VV', 'VH']
    required_max_index = max(CHANNEL_MAP[channel] for channel in required_channels)
    if x_tensor.shape[2] <= required_max_index:
        raise ValueError(
            f'当前标签构建至少需要通道 {required_channels}，但 X_tensor 仅有 {x_tensor.shape[2]} 个通道。'
        )

    print('正在基于最后一个月的硬阈值规则生成伪标签...')
    last_month_features = x_tensor[:, -1, :, :, :].cpu().numpy()

    ndvi = last_month_features[:, CHANNEL_MAP['NDVI'], :, :]
    vv = last_month_features[:, CHANNEL_MAP['VV'], :, :]
    vh = last_month_features[:, CHANNEL_MAP['VH'], :, :]

    valid_mask = np.isfinite(ndvi) & np.isfinite(vv) & np.isfinite(vh) & (ndvi > NDVI_VALID_THRESHOLD)
    if not np.any(valid_mask):
        raise ValueError('没有找到可用于生成标签的有效像元，请检查 X_tensor 数值范围与 NDVI 阈值。')

    batch, height, width = ndvi.shape
    y_array = np.zeros((batch, height, width), dtype=np.int64)

    light_mask = valid_mask & (
        (ndvi < NDVI_LIGHT_THRESHOLD) |
        (vv < VV_HIGH_THRESHOLD) |
        (vh < VH_HIGH_THRESHOLD)
    )
    moderate_mask = valid_mask & (
        (ndvi < NDVI_MODERATE_THRESHOLD) |
        (vv < VV_MID_THRESHOLD) |
        (vh < VH_MID_THRESHOLD)
    )
    severe_mask = valid_mask & (
        (ndvi < NDVI_SEVERE_THRESHOLD) &
        (vv < VV_MID_THRESHOLD) &
        (vh < VH_MID_THRESHOLD)
    )

    y_array[light_mask] = 1
    y_array[moderate_mask] = 2
    y_array[severe_mask] = 3

    y_tensor = torch.from_numpy(y_array)
    print(f'Y_tensor 生成完毕！形状: {y_tensor.shape}')
    print(f'标签类别分布: {torch.bincount(y_tensor.flatten(), minlength=4)}')
    return y_tensor


for year in YEARS:
    x_path = os.path.join(base_dir, f'dataset_X_{year}_new.pt')
    y_path = os.path.join(base_dir, f'dataset_Y_{year}_new_threshold.pt')

    print(f'\n===== 开始生成 {year} 年阈值伪标签 =====')
    print('正在加载 X_tensor...')
    X_tensor = torch.load(x_path)
    print(f'X_tensor 形状: {X_tensor.shape}')

    Y_tensor = generate_threshold_labels(X_tensor)

    torch.save(Y_tensor, y_path)
    print(f'{year} 年硬阈值版伪标签已成功保存至云盘: {y_path}')
