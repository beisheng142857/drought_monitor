import argparse
import os
from typing import Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

RAW_DATA_DIR = '/root/autodl-tmp/data_forecast_V2'
OUTPUT_DIR = '/root/autodl-tmp/zyk_drought_monitor/data_V2'
DEFAULT_YEARS = [2021, 2022, 2023, 2024] #, 2025
DEFAULT_MONTHS = [4, 5, 6, 7, 8, 9]
FILE_PREFIX = 'Fused_100m'
PATCH_SIZE = 128
STRIDE = 64
NODATA_THRESHOLD = 0.4
NORMALIZE_CHANNELS = False
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='构建 forecasting V2 连续月份序列 X_tensor')
    parser.add_argument('--input_dir', type=str, default=RAW_DATA_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--years', nargs='+', type=int, default=DEFAULT_YEARS)
    parser.add_argument('--months', nargs='+', type=int, default=DEFAULT_MONTHS)
    parser.add_argument('--file_prefix', type=str, default=FILE_PREFIX)
    parser.add_argument('--patch_size', type=int, default=PATCH_SIZE)
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--nodata_threshold', type=float, default=NODATA_THRESHOLD)
    parser.add_argument('--normalize_channels', action='store_true')
    parser.add_argument('--output_prefix', type=str, default='sequence_X')
    return parser.parse_args()


def build_tiff_paths(year: int, months: Sequence[int], input_dir: str, file_prefix: str) -> list[str]:
    return [os.path.join(input_dir, f'{file_prefix}_{year}_{month:02d}.tif') for month in months]


def validate_tiff_paths(tiff_paths: Sequence[str]) -> None:
    missing_paths = [path for path in tiff_paths if not os.path.exists(path)]
    if missing_paths:
        missing_text = '\n'.join(f'  - {path}' for path in missing_paths)
        raise FileNotFoundError(f'以下 TIFF 文件不存在：\n{missing_text}')


def read_window_with_mask(dataset: rasterio.io.DatasetReader, window: Window) -> tuple[np.ndarray, np.ndarray]:
    data = dataset.read(window=window).astype(np.float32)
    invalid_mask = ~np.isfinite(data)
    nodata_value = dataset.nodata
    if nodata_value is not None:
        invalid_mask |= data == nodata_value
    return data, invalid_mask


def normalize_channels_inplace(full_tensor: np.ndarray) -> None:
    _, _, channels, _, _ = full_tensor.shape
    for channel_idx in range(channels):
        channel_slice = full_tensor[:, :, channel_idx, :, :]
        valid_mask = channel_slice != 0.0
        valid_pixels = channel_slice[valid_mask]
        if valid_pixels.size == 0:
            print(f'波段 {channel_idx} 全是 0，跳过归一化。')
            continue
        mean = valid_pixels.mean()
        std = valid_pixels.std()
        channel_slice[valid_mask] = (valid_pixels - mean) / (std + EPS)
        print(f'波段 {channel_idx} 归一化完成 (mean={mean:.4f}, std={std:.4f})')


def process_tiffs_to_tensor(
    tiff_paths: Sequence[str],
    patch_size: int,
    stride: int,
    nodata_threshold: float,
    normalize_channels: bool,
) -> torch.Tensor:
    validate_tiff_paths(tiff_paths)
    print(f'开始处理 {len(tiff_paths)} 个连续月份时相...')
    datasets = [rasterio.open(path) for path in tiff_paths]
    try:
        reference = datasets[0]
        height, width, bands = reference.height, reference.width, reference.count
        print(f'影像尺寸: {width} x {height}, 波段数: {bands}')

        for dataset in datasets[1:]:
            if dataset.height != height or dataset.width != width or dataset.count != bands:
                raise ValueError('输入 TIFF 的空间尺寸或波段数不一致，无法堆叠为连续时间序列。')

        valid_patches: list[np.ndarray] = []
        best_invalid_ratio = 1.0
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                window = Window(x, y, patch_size, patch_size)
                window_data = []
                window_invalid_masks = []
                for dataset in datasets:
                    data, invalid_mask = read_window_with_mask(dataset, window)
                    window_data.append(data)
                    window_invalid_masks.append(invalid_mask)

                patch = np.stack(window_data, axis=0)
                invalid_mask = np.stack(window_invalid_masks, axis=0)
                invalid_ratio = invalid_mask.mean()
                best_invalid_ratio = min(best_invalid_ratio, float(invalid_ratio))
                if invalid_ratio > nodata_threshold:
                    continue

                patch[invalid_mask] = 0.0
                valid_patches.append(patch)

        if not valid_patches:
            raise ValueError(
                '没有提取到有效图块。'
                f' 当前允许最大无效比例={nodata_threshold:.2f}，最佳窗口无效比例={best_invalid_ratio:.4f}。'
            )

        full_tensor = np.asarray(valid_patches, dtype=np.float32)
        if normalize_channels:
            print('正在按波段对非零像元做归一化...')
            normalize_channels_inplace(full_tensor)

        full_tensor = np.nan_to_num(full_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        torch_tensor = torch.from_numpy(full_tensor)
        print(f'生成的张量形状 (Batch, Time, Channels, H, W): {tuple(torch_tensor.shape)}')
        return torch_tensor
    finally:
        for dataset in datasets:
            dataset.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for year in args.years:
        print('=' * 80)
        print(f'开始构建 {year} 年的 forecasting V2 连续月份 X_tensor')
        tiff_files = build_tiff_paths(year, args.months, args.input_dir, args.file_prefix)
        x_tensor = process_tiffs_to_tensor(
            tiff_paths=tiff_files,
            patch_size=args.patch_size,
            stride=args.stride,
            nodata_threshold=args.nodata_threshold,
            normalize_channels=args.normalize_channels,
        )
        output_path = os.path.join(args.output_dir, f'{args.output_prefix}_{year}.pt')
        torch.save(x_tensor, output_path)
        print(f'已保存: {output_path}')
        print(f'{year} 年月份序列: {args.months}')


if __name__ == '__main__':
    main()
