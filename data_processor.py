# ==========================================
# 第一步：导入所需依赖库
# ==========================================
import os
from typing import Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

# ==========================================
# 第二步：定义核心的数据切片与张量转换函数
# ==========================================
def validate_tiff_paths(tiff_paths: Sequence[str]) -> None:
    missing_paths = [path for path in tiff_paths if not os.path.exists(path)]
    if missing_paths:
        missing_text = '\n'.join(f'  - {path}' for path in missing_paths)
        raise FileNotFoundError(f"以下 TIFF 文件不存在：\n{missing_text}")


def read_window_with_mask(dataset, window: Window):
    data = dataset.read(window=window).astype(np.float32)
    invalid_mask = ~np.isfinite(data)

    nodata_value = dataset.nodata
    if nodata_value is not None:
        invalid_mask |= data == nodata_value

    return data, invalid_mask


def process_tiffs_to_tensor(tiff_paths, patch_size=128, stride=128, nodata_threshold=0.1, normalize_channels=False):
    validate_tiff_paths(tiff_paths)
    print(f"开始处理，共 {len(tiff_paths)} 个时相...")

    datasets = [rasterio.open(path) for path in tiff_paths]
    try:
        reference = datasets[0]
        height, width, channels = reference.height, reference.width, reference.count
        print(f"影像全局尺寸: {width} x {height}, 波段数: {channels}")

        for dataset in datasets[1:]:
            if dataset.height != height or dataset.width != width or dataset.count != channels:
                raise ValueError("输入 TIFF 的空间尺寸或波段数不一致，无法直接堆叠为时间序列。")

        valid_patches = []
        best_invalid_ratio = 1.0

        print(f"开始以 {patch_size}x{patch_size} 尺寸进行切片...")
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                window = Window(x, y, patch_size, patch_size)
                patch_parts = []
                invalid_masks = []

                for dataset in datasets:
                    data, invalid_mask = read_window_with_mask(dataset, window)
                    patch_parts.append(data)
                    invalid_masks.append(invalid_mask)

                patch = np.stack(patch_parts, axis=0)
                invalid_mask = np.stack(invalid_masks, axis=0)
                invalid_ratio = invalid_mask.mean()
                best_invalid_ratio = min(best_invalid_ratio, float(invalid_ratio))

                if invalid_ratio > nodata_threshold:
                    continue

                patch[invalid_mask] = 0.0
                valid_patches.append(patch)

        if not valid_patches:
            raise ValueError(
                "没有提取到有效的图块，请检查文件路径、无效像元比例或原始影像是否大面积缺失。"
                f" 当前阈值为 {nodata_threshold:.2f}，最佳窗口无效比例为 {best_invalid_ratio:.4f}。"
            )

        numpy_tensor = np.asarray(valid_patches, dtype=np.float32)

        if normalize_channels:
            print("开始按波段对非零像元做归一化...")
            for channel_idx in range(channels):
                channel_slice = numpy_tensor[:, :, channel_idx, :, :]
                valid_mask = channel_slice != 0.0
                valid_pixels = channel_slice[valid_mask]

                if valid_pixels.size == 0:
                    print(f"⚠️ 波段 {channel_idx} 全为 0，跳过归一化。")
                    continue

                mean = valid_pixels.mean()
                std = valid_pixels.std()
                channel_slice[valid_mask] = (valid_pixels - mean) / (std + 1e-8)
                print(f"波段 {channel_idx} 归一化完成 (Mean: {mean:.4f}, Std: {std:.4f})")

        numpy_tensor = np.nan_to_num(numpy_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        torch_tensor = torch.from_numpy(numpy_tensor)

        print("转换完成！")
        print(f"生成的张量形状 (Batch, Time, Channels, H, W): {torch_tensor.shape}")
        return torch_tensor
    finally:
        for dataset in datasets:
            dataset.close()


# ==========================================
# 第三步：执行转换并保存至云盘
# ==========================================

# 记得保留你刚才修改好的真实云盘路径
base_dir = '/content/drive/MyDrive/GEE_Drought_Project'
fin_dir = '/content/drive/MyDrive/drought_monitor'
output_dir = os.path.join(fin_dir, 'data_proc')
os.makedirs(output_dir, exist_ok=True)

tiff_files = [
    os.path.join(base_dir, 'Fused_100m_2025_05.tif'),
    os.path.join(base_dir, 'Fused_100m_2025_06.tif'),
    os.path.join(base_dir, 'Fused_100m_2025_07.tif'),
    os.path.join(base_dir, 'Fused_100m_2025_08.tif'),
    os.path.join(base_dir, 'Fused_100m_2025_09.tif')
]

X_tensor = process_tiffs_to_tensor(tiff_files, patch_size=128, stride=128, nodata_threshold=0.1, normalize_channels=False)

save_path = os.path.join(output_dir, 'dataset_X_2025.pt')
torch.save(X_tensor, save_path)

print(f"张量已成功保存至云盘: {save_path}")
