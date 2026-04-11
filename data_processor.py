# ==========================================
# 第一步：挂载 Google Drive
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

# ==========================================
# 第二步：导入所需依赖库
# ==========================================
import os
import numpy as np
import rasterio
import torch

# ==========================================
# 第三步：定义核心的数据切片与张量转换函数
# ==========================================
def process_tiffs_to_tensor(tiff_paths, patch_size=128, stride=128, nodata_threshold=0.1):
    print(f"开始处理，共 {len(tiff_paths)} 个时相...")

    datasets = [rasterio.open(path) for path in tiff_paths]
    meta = datasets[0].meta
    height, width = meta['height'], meta['width']
    channels = meta['count']

    print(f"影像全局尺寸: {width} x {height}, 波段数: {channels}")

    time_series_data = np.stack([ds.read() for ds in datasets], axis=0)

    nodata_value = meta.get('nodata', None)
    if nodata_value is not None:
        time_series_data[time_series_data == nodata_value] = np.nan

    valid_patches = []

    print(f"开始以 {patch_size}x{patch_size} 尺寸进行切片...")
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):

            patch = time_series_data[:, :, y:y+patch_size, x:x+patch_size]

            nan_ratio = np.isnan(patch).sum() / patch.size
            if nan_ratio > nodata_threshold:
                continue

            # 【已经修复：将 nan_0 修改为了 nan】
            patch = np.nan_to_num(patch, nan=0.0)
            valid_patches.append(patch)

    for ds in datasets:
        ds.close()

    if not valid_patches:
        raise ValueError("没有提取到有效的图块，请检查文件路径是否正确或影像是否全是空白！")

    numpy_tensor = np.array(valid_patches, dtype=np.float32)
    torch_tensor = torch.from_numpy(numpy_tensor)

    print(f"转换完成！")
    print(f"生成的张量形状 (Batch, Time, Channels, H, W): {torch_tensor.shape}")

    return torch_tensor


# ==========================================
# 第四步：执行转换并保存至云盘
# ==========================================

# 记得保留你刚才修改好的真实云盘路径
base_dir = '/content/drive/MyDrive/GEE_Drought_Project'

tiff_files = [
    os.path.join(base_dir, 'Fused_100m_2023_05.tif'),
    os.path.join(base_dir, 'Fused_100m_2023_06.tif'),
    os.path.join(base_dir, 'Fused_100m_2023_07.tif'),
    os.path.join(base_dir, 'Fused_100m_2023_08.tif'),
    os.path.join(base_dir, 'Fused_100m_2023_09.tif')
]

X_tensor = process_tiffs_to_tensor(tiff_files, patch_size=128, stride=128)

save_path = os.path.join(base_dir, 'dataset_X_2023.pt')
torch.save(X_tensor, save_path)

print(f"张量已成功保存至云盘: {save_path}")
