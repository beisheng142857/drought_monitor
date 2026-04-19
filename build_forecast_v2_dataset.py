import argparse
import os
from typing import List, Tuple

import torch


def build_forecast_v2_samples(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    input_steps: int,
    lead_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 形状应为 (Batch, Time, Channels, H, W)，当前为 {x_tensor.shape}')
    if y_tensor.ndim not in (3, 4):
        raise ValueError(f'Y_tensor 形状应为 (Batch, H, W) 或 (Batch, Time, H, W)，当前为 {y_tensor.shape}')
    if x_tensor.shape[0] == 0:
        raise ValueError('X_tensor 为空。')
    if input_steps <= 0 or lead_steps <= 0:
        raise ValueError('input_steps 和 lead_steps 必须为正整数。')

    total_time = x_tensor.shape[1]
    if y_tensor.ndim == 3:
        raise ValueError('forecasting V2 需要按时间展开的标签张量，Y_tensor 不能只有 (Batch, H, W)。')
    if y_tensor.shape[0] != x_tensor.shape[0] or y_tensor.shape[1] != total_time:
        raise ValueError('X_tensor 与 Y_tensor 的 batch/time 维度不一致。')

    x_samples: List[torch.Tensor] = []
    y_samples: List[torch.Tensor] = []

    max_start = total_time - input_steps - lead_steps + 1
    if max_start <= 0:
        raise ValueError('时间长度不足以构造 forecasting V2 样本，请增大月份数或减小 input_steps/lead_steps。')

    for start_idx in range(max_start):
        input_end = start_idx + input_steps
        target_idx = input_end + lead_steps - 1
        x_samples.append(x_tensor[:, start_idx:input_end, :, :, :])
        y_samples.append(y_tensor[:, target_idx, :, :])

    forecast_x = torch.cat(x_samples, dim=0).contiguous()
    forecast_y = torch.cat(y_samples, dim=0).contiguous()
    return forecast_x, forecast_y


def parse_args():
    parser = argparse.ArgumentParser(description='构建 forecasting 数据集 V2（连续月序列 + 滑动窗口）')
    parser.add_argument('--x_path', type=str, required=True)
    parser.add_argument('--y_path', type=str, required=True)
    parser.add_argument('--input_steps', type=int, default=4)
    parser.add_argument('--lead_steps', type=int, default=1)
    parser.add_argument('--output_x_path', type=str, required=True)
    parser.add_argument('--output_y_path', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.x_path):
        raise FileNotFoundError(f'未找到输入 X_tensor: {args.x_path}')
    if not os.path.exists(args.y_path):
        raise FileNotFoundError(f'未找到输入 Y_tensor: {args.y_path}')

    x_tensor = torch.load(args.x_path, map_location='cpu')
    y_tensor = torch.load(args.y_path, map_location='cpu')
    forecast_x, forecast_y = build_forecast_v2_samples(
        x_tensor=x_tensor,
        y_tensor=y_tensor,
        input_steps=args.input_steps,
        lead_steps=args.lead_steps,
    )

    os.makedirs(os.path.dirname(args.output_x_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_y_path), exist_ok=True)
    torch.save(forecast_x, args.output_x_path)
    torch.save(forecast_y, args.output_y_path)

    print(f'forecast_v2_X 形状: {tuple(forecast_x.shape)}')
    print(f'forecast_v2_Y 形状: {tuple(forecast_y.shape)}')
    print(f'已保存: {args.output_x_path}')
    print(f'已保存: {args.output_y_path}')


if __name__ == '__main__':
    main()
