import os

import numpy as np
import torch

# 1. 设置路径
base_dir = '/root/autodl-tmp/data_proc'
YEARS = [2021, 2022, 2023, 2024, 2025]

# 2. 主阈值配置（核心仍基于 NDVI + VV + VH）
NDVI_VALID_THRESHOLD = 0.05
NDVI_LIGHT_THRESHOLD = 0.60
NDVI_MODERATE_THRESHOLD = 0.40
NDVI_SEVERE_THRESHOLD = 0.20
VV_HIGH_THRESHOLD = -10.0
VV_MID_THRESHOLD = -13.0
VH_HIGH_THRESHOLD = -16.0
VH_MID_THRESHOLD = -19.0

# 3. 辅助门控配置（10 通道增强）
USE_AUX_GATING = True
NDWI_WATER_MASK_THRESHOLD = 0.20   # NDWI 过高常对应水体/湿区，避免误判为旱情
NDMI_DRY_SUPPORT_THRESHOLD = 0.00  # NDMI 低于该阈值时更支持干旱判断
MSAVI_DRY_SUPPORT_THRESHOLD = 0.30 # MSAVI 偏低时更支持干旱判断

# 4. 特征通道映射（与 10 通道 X_tensor 一致）
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


def generate_threshold_labels_v2(x_tensor: torch.Tensor) -> torch.Tensor:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 形状应为 (Batch, Time, Channels, H, W)，当前维度为 {x_tensor.ndim}')

    required_channels = ['NDVI', 'VV', 'VH']
    required_max_index = max(CHANNEL_MAP[channel] for channel in required_channels)
    if x_tensor.shape[2] <= required_max_index:
        raise ValueError(
            f'当前标签构建至少需要通道 {required_channels}，但 X_tensor 仅有 {x_tensor.shape[2]} 个通道。'
        )

    print('正在基于最后一个月生成优化版阈值伪标签...')
    last_month_features = x_tensor[:, -1, :, :, :].cpu().numpy()

    ndvi = last_month_features[:, CHANNEL_MAP['NDVI'], :, :]
    vv = last_month_features[:, CHANNEL_MAP['VV'], :, :]
    vh = last_month_features[:, CHANNEL_MAP['VH'], :, :]

    valid_mask = np.isfinite(ndvi) & np.isfinite(vv) & np.isfinite(vh) & (ndvi > NDVI_VALID_THRESHOLD)
    if not np.any(valid_mask):
        raise ValueError('没有找到可用于生成标签的有效像元，请检查 X_tensor 数值范围与 NDVI 阈值。')

    # 默认全部为 0（无旱）
    y_array = np.zeros(ndvi.shape, dtype=np.int64)

    # 基础规则（兼容旧版语义）
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

    if USE_AUX_GATING and x_tensor.shape[2] > CHANNEL_MAP['MSAVI']:
        ndwi = last_month_features[:, CHANNEL_MAP['NDWI'], :, :]
        ndmi = last_month_features[:, CHANNEL_MAP['NDMI'], :, :]
        msavi = last_month_features[:, CHANNEL_MAP['MSAVI'], :, :]

        aux_finite = np.isfinite(ndwi) & np.isfinite(ndmi) & np.isfinite(msavi)

        # 水体/湿区掩膜：不参与干旱判定
        non_water_mask = aux_finite & (ndwi < NDWI_WATER_MASK_THRESHOLD)
        rule_base_mask = valid_mask & non_water_mask

        # 轻旱：保持较敏感，主要做水体过滤
        light_mask = light_mask & rule_base_mask

        # 中旱：加入 NDMI 或 MSAVI 支撑，减少单通道噪声触发
        dry_support_mask = (ndmi < NDMI_DRY_SUPPORT_THRESHOLD) | (msavi < MSAVI_DRY_SUPPORT_THRESHOLD)
        moderate_mask = moderate_mask & rule_base_mask & dry_support_mask

        # 重旱：更严格，要求 NDMI 与 MSAVI 同步偏低
        severe_support_mask = (ndmi < NDMI_DRY_SUPPORT_THRESHOLD) & (msavi < MSAVI_DRY_SUPPORT_THRESHOLD)
        severe_mask = severe_mask & rule_base_mask & severe_support_mask

    # 由弱到强赋值，再由强覆盖弱
    y_array[light_mask] = 1
    y_array[moderate_mask] = 2
    y_array[severe_mask] = 3

    y_tensor = torch.from_numpy(y_array)
    print(f'Y_tensor 生成完毕！形状: {y_tensor.shape}')
    print(f'标签类别分布: {torch.bincount(y_tensor.flatten(), minlength=4)}')
    return y_tensor


for year in YEARS:
    x_path = os.path.join(base_dir, f'dataset_X_{year}.pt')
    y_path = os.path.join(base_dir, f'dataset_Y_{year}_threshold_v2.pt')

    print(f'\n===== 开始生成 {year} 年优化版阈值伪标签 =====')
    print('正在加载 X_tensor...')
    x_tensor = torch.load(x_path)
    print(f'X_tensor 形状: {x_tensor.shape}')

    y_tensor = generate_threshold_labels_v2(x_tensor)

    torch.save(y_tensor, y_path)
    print(f'{year} 年优化版阈值伪标签已保存: {y_path}')
