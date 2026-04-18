import os
from typing import Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

RAW_DATA_DIR = '/root/autodl-tmp/zyk_drought_monitor/data_raw/gee_tiffs'
OUTPUT_DIR = '/root/autodl-tmp/zyk_drought_monitor/data'
TARGET_YEAR = 2023
TARGET_MONTHS = [5, 6, 7, 8, 9]
FILE_PREFIX = 'Fused_100m'
PATCH_SIZE = 128
STRIDE = 64
NODATA_THRESHOLD = 0.4
NORMALIZE_CHANNELS = False
EPS = 1e-8


def build_tiff_paths(year: int, months: Sequence[int], input_dir: str) -> list[str]:
    return [os.path.join(input_dir, f'{FILE_PREFIX}_{year}_{month:02d}.tif') for month in months]


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
            print(f'⚠️ 波段 {channel_idx} 全是 0，跳过归一化。')
            continue

        mean = valid_pixels.mean()
        std = valid_pixels.std()
        channel_slice[valid_mask] = (valid_pixels - mean) / (std + EPS)
        print(f'波段 {channel_idx} 归一化完成 (mean={mean:.4f}, std={std:.4f})')



def process_tiffs_to_tensor(
    tiff_paths: Sequence[str],
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    nodata_threshold: float = NODATA_THRESHOLD,
    normalize_channels: bool = NORMALIZE_CHANNELS,
) -> torch.Tensor:
    validate_tiff_paths(tiff_paths)
    print(f'开始处理 {len(tiff_paths)} 个时相的影像...')

    datasets = [rasterio.open(path) for path in tiff_paths]
    try:
        reference = datasets[0]
        height, width, bands = reference.height, reference.width, reference.count
        print(f'影像尺寸: {width} x {height}, 波段数: {bands}')

        for dataset in datasets[1:]:
            if dataset.height != height or dataset.width != width or dataset.count != bands:
                raise ValueError('输入 TIFF 的空间尺寸或波段数不一致，无法直接堆叠为时间序列。')

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
                f' 当前允许的最大无效像元比例为 {nodata_threshold:.2f}，'
                f'但扫描到的最佳窗口无效比例也有 {best_invalid_ratio:.4f}。'
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


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tiff_files = build_tiff_paths(TARGET_YEAR, TARGET_MONTHS, RAW_DATA_DIR)
    output_path = os.path.join(OUTPUT_DIR, f'dataset_X_{TARGET_YEAR}.pt')

    x_tensor = process_tiffs_to_tensor(
        tiff_files,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        nodata_threshold=NODATA_THRESHOLD,
        normalize_channels=NORMALIZE_CHANNELS,
    )
    torch.save(x_tensor, output_path)
    print(f'X_tensor 已保存至: {output_path}')
