import os
import numpy as np
import rasterio
from rasterio.windows import Window
import torch

# 【修改点 1】: 将默认容忍度从 0.1 放宽到 0.4 (允许 40% 的数据是 NaN/缺失)
def process_tiffs_to_tensor_robust(tiff_paths, patch_size=128, stride=64, nodata_threshold=0.4):
    print(f"开始处理 {len(tiff_paths)} 个时相的影像...")
    
    with rasterio.open(tiff_paths[0]) as src:
        h, w = src.height, src.width
        bands = src.count

    valid_patches = []
    datasets = [rasterio.open(p) for p in tiff_paths]
    
    min_invalid_ratio = 1.0 # 用于诊断：记录全图中质量最好的一块的缺失率

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            window = Window(x, y, patch_size, patch_size)
            patch = np.stack([ds.read(window=window) for ds in datasets], axis=0)
            
            invalid_mask = np.isnan(patch) | np.isinf(patch) | (patch < -1000) | (patch > 10000)
            invalid_ratio = invalid_mask.sum() / patch.size
            
            # 记录看到过的最低缺失率
            if invalid_ratio < min_invalid_ratio:
                min_invalid_ratio = invalid_ratio
            
            if invalid_ratio > nodata_threshold:
                continue
            
            patch[invalid_mask] = 0.0
            valid_patches.append(patch)

    for ds in datasets: 
        ds.close()
    
    # 【修改点 2】: 给出更精确的诊断报错信息
    if not valid_patches:
        raise ValueError(f"\n❌ 严重错误：所有图块都被过滤掉了！\n"
                         f"-> 你设置的最高容忍度是允许 {nodata_threshold*100}% 的像素缺失。\n"
                         f"-> 但经扫描，全图中数据质量【最好】的一块，其无效像素比例也达到了 {min_invalid_ratio*100:.2f}%！\n"
                         f"结论：你的 TIFF 影像中存在整月或整波段级别的大面积缺失，请检查 GEE 下载源。")

    full_tensor = np.array(valid_patches, dtype=np.float32) 
    
    print("正在执行严格波段归一化... (已启用内存优化与加速)")
    for c in range(bands):
        # 创建对当前波段的引用，不占用新内存
        band_slice = full_tensor[:, :, c, :, :]
        
        # 获取有效像素的布尔掩码
        valid_mask = band_slice != 0.0
        
        # 仅提取有效像素参与计算
        valid_pixels = band_slice[valid_mask]
        
        if len(valid_pixels) > 0:
            mean, std = valid_pixels.mean(), valid_pixels.std()
            # 【核心优化】：使用掩码就地(in-place)更新数值
            # 只有 valid_mask 为 True 的地方会被修改，0.0 依然是 0.0
            band_slice[valid_mask] = (valid_pixels - mean) / (std + 1e-8)
            print(f"波段 {c} 归一化完成 (Mean: {mean:.4f}, Std: {std:.4f})")
        else:
            print(f"⚠️ 警告：波段 {c} 全是 0.0，没有提取到任何有效数据！")

    full_tensor = np.nan_to_num(full_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
# 执行部分保持不变...

if __name__ == "__main__":
    # 请根据你当前的实际路径进行修改
    base_dir = '/content/drive/MyDrive/GEE_Drought_Project'
    fin_dir = '/content/drive/MyDrive/drought_monitor/data_new'

    # 假设你的文件名为 Fused_100m_2023_05.tif 到 09.tif
    tiff_files = [os.path.join(base_dir, f'Fused_100m_2023_{m:02d}.tif') for m in range(5, 10)]
    
    out_x_path = os.path.join(fin_dir, 'dataset_X_2023.pt')
    
    # 执行生成并保存
    X_tensor = process_tiffs_to_tensor_robust(tiff_files, patch_size=128, stride=64)
    torch.save(X_tensor, out_x_path)
    print(f"安全干净的 X_tensor 已保存至: {out_x_path}")