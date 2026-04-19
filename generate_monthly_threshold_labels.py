import argparse
import os
from typing import Sequence

import numpy as np
import torch

NDVI_VALID_THRESHOLD = 0.05
NDVI_LIGHT_THRESHOLD = 0.60
NDVI_MODERATE_THRESHOLD = 0.40
NDVI_SEVERE_THRESHOLD = 0.20
VV_HIGH_THRESHOLD = -10.0
VV_MID_THRESHOLD = -13.0
VH_HIGH_THRESHOLD = -16.0
VH_MID_THRESHOLD = -19.0


def generate_threshold_labels_for_month(
    x_tensor: torch.Tensor,
    target_month_index: int,
) -> torch.Tensor:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 形状应为 (Batch, Time, Channels, H, W)，当前为 {x_tensor.shape}')
    if x_tensor.shape[2] < 3:
        raise ValueError('当前标签构建至少需要 3 个通道：NDVI、VV、VH。')
    if target_month_index < 0 or target_month_index >= x_tensor.shape[1]:
        raise IndexError(f'target_month_index 越界，当前时间步数为 {x_tensor.shape[1]}')

    target_features = x_tensor[:, target_month_index, :, :, :].cpu().numpy()
    ndvi = target_features[:, 0, :, :]
    vv = target_features[:, 1, :, :]
    vh = target_features[:, 2, :, :]

    valid_mask = np.isfinite(ndvi) & np.isfinite(vv) & np.isfinite(vh) & (ndvi > NDVI_VALID_THRESHOLD)
    if not np.any(valid_mask):
        raise ValueError('没有找到可用于生成标签的有效像元，请检查输入张量。')

    batch, height, width = ndvi.shape
    y_array = np.zeros((batch, height, width), dtype=np.int64)

    light_mask = valid_mask & ((ndvi < NDVI_LIGHT_THRESHOLD) | (vv < VV_HIGH_THRESHOLD) | (vh < VH_HIGH_THRESHOLD))
    moderate_mask = valid_mask & ((ndvi < NDVI_MODERATE_THRESHOLD) | (vv < VV_MID_THRESHOLD) | (vh < VH_MID_THRESHOLD))
    severe_mask = valid_mask & ((ndvi < NDVI_SEVERE_THRESHOLD) & (vv < VV_MID_THRESHOLD) & (vh < VH_MID_THRESHOLD))

    y_array[light_mask] = 1
    y_array[moderate_mask] = 2
    y_array[severe_mask] = 3

    y_tensor = torch.from_numpy(y_array)
    print(f'标签生成完毕，target_month_index={target_month_index}，形状={tuple(y_tensor.shape)}')
    print(f'标签类别分布: {torch.bincount(y_tensor.flatten(), minlength=4)}')
    return y_tensor


def parse_args():
    parser = argparse.ArgumentParser(description='按指定月份索引从 X_tensor 生成 threshold 标签 pt 文件')
    parser.add_argument('--x_path', type=str, required=True)
    parser.add_argument('--target_month_index', type=int, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.x_path):
        raise FileNotFoundError(f'未找到输入张量: {args.x_path}')

    x_tensor = torch.load(args.x_path, map_location='cpu')
    y_tensor = generate_threshold_labels_for_month(x_tensor, args.target_month_index)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(y_tensor, args.output_path)
    print(f'已保存指定月份标签文件: {args.output_path}')


if __name__ == '__main__':
    main()
