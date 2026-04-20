import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 设置中文字体
from matplotlib import font_manager
FONT_PATH = '/root/autodl-tmp/zyk_drought_monitor/SimHei.ttf'
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

# 假设你的 10 个通道分别是这些（请根据你 data_V2 的实际情况修改！）
FEATURE_NAMES = ['NDVI', 'VV', 'VH', 'VVVH', '降水(PRE)', '气温(TMP)', '土壤湿度(SM)', '地表温度(LST)', '蒸散发(ET)', '高程(DEM)']

def plot_channel_attention(attn_weights, save_path="channel_attention.png"):
    """
    绘制通道注意力权重柱状图
    假设 attn_weights 形状经过处理后为 (10,)
    """
    plt.figure(figsize=(10, 6))
    
    # 画柱状图，使用不同的颜色渐变
    bars = plt.bar(FEATURE_NAMES, attn_weights, color='skyblue', edgecolor='black')
    
    # 突出显示权重最大的前 3 个特征（标红）
    top3_indices = np.argsort(attn_weights)[-3:]
    for idx in top3_indices:
        bars[idx].set_color('coral')
        bars[idx].set_edgecolor('black')
        
    plt.title('通道注意力权重分布 (Channel Attention Weights)', fontsize=16)
    plt.ylabel('注意力权重分数 (0~1)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"已保存通道注意力图: {save_path}")
    plt.show()


def plot_spatial_attention(ref_image, attn_map, save_path="spatial_attention.png"):
    """
    绘制空间注意力热力图
    ref_image: 你的参考底图 (128, 128)，比如当月的 NDVI 或 VV
    attn_map: 注意力权重矩阵 (128, 128)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 原始底图
    ax = axes[0]
    im1 = ax.imshow(ref_image, cmap='gray')
    ax.set_title('参考底图 (如 NDVI)')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    
    # 2. 纯注意力热力图
    ax = axes[1]
    # 使用 'jet' 或 'hot' color map 最有高级感
    im2 = ax.imshow(attn_map, cmap='jet') 
    ax.set_title('空间注意力分布 (Spatial Attention)')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    
    # 3. 混合叠加图 (Overlay) - 论文中最爱用的图
    ax = axes[2]
    ax.imshow(ref_image, cmap='gray') # 底图
    im3 = ax.imshow(attn_map, cmap='jet', alpha=0.5) # 加上 50% 透明度的热力图
    ax.set_title('底图与注意力叠加 (Overlay)')
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"已保存空间注意力图: {save_path}")
    plt.show()

# =====================================================================
# 主执行逻辑：如何提取并使用
# =====================================================================
if __name__ == '__main__':
    # 1. 加载你的数据和模型
    # (这里省略完整的 load_model_from_checkpoint 代码，你可以直接从 forecast_compare.py 抄过来)
    # model, _ = load_model_from_checkpoint(...)
    # x_sample = ... # 取一个 batch 的测试数据，形状比如 [1, 4, 10, 128, 128]
    
    # 2. 模拟进行一次前向传播
    # model.eval()
    # with torch.no_grad():
    #     _ = model(x_sample)
        
    # 3. 提取刚刚算出来的注意力权重！
    # 注意：这里的路径 "model.encoder[0].input_attn" 取决于你在 ConvLSTM 里的变量是怎么命名的
    # 如果你是直接作为 ConvLSTM 的属性，可能是 model.input_attn.saved_attn_weights
    
    # attn_weights_raw = model.input_attn.saved_attn_weights 
    
    # ---------------- 模拟数据测试画图 ----------------
    print("正在生成模拟演示图...")
    # 如果是通道注意力，形状可能是 [1, 10, 1, 1] 压缩为 [10]
    fake_channel_weights = np.random.rand(10)
    fake_channel_weights = fake_channel_weights / np.sum(fake_channel_weights) # 归一化
    plot_channel_attention(fake_channel_weights)
    
    # 如果是空间注意力，形状可能是 [1, 1, 128, 128] 压缩为 [128, 128]
    fake_ref_image = np.random.rand(128, 128) # 模拟 NDVI 底图
    
    # 模拟一个注意力集中的区域（比如在中心位置发现了旱情前兆）
    x, y = np.meshgrid(np.linspace(-1,1,128), np.linspace(-1,1,128))
    fake_spatial_weights = np.exp(-2 * (x**2 + y**2)) 
    plot_spatial_attention(fake_ref_image, fake_spatial_weights)