import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

CLASS_NAMES = ['无旱', '轻旱', '中旱', '重/特旱']
CLASS_COLORS = ['#2ca25f', '#fee08b', '#f46d43', '#a50026']
DROUGHT_CMAP = ListedColormap(CLASS_COLORS)
DROUGHT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], DROUGHT_CMAP.N)
FONT_PATH = '/root/autodl-tmp/zyk_drought_monitor/SimHei.ttf'
DEFAULT_OUTPUT_DIR = '/root/autodl-tmp/zyk_drought_monitor/drought_outputs/ppt_figures'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='将 NDVI / 真值 / 预测 tif 自动排版为适合答辩 PPT 的三联图')
    parser.add_argument('--pred_tif', type=str, required=True, help='预测结果 GeoTIFF 路径')
    parser.add_argument('--gt_tif', type=str, required=True, help='真值结果 GeoTIFF 路径')
    parser.add_argument('--reference_tif', type=str, required=True, help='参考底图 GeoTIFF 路径，一般为 NDVI')
    parser.add_argument('--output_path', type=str, default='', help='输出 PNG 路径')
    parser.add_argument('--title', type=str, default='干旱预测结果空间展示')
    parser.add_argument('--subtitle', type=str, default='左：参考 NDVI    中：真实干旱等级    右：模型预测干旱等级')
    parser.add_argument('--model_name', type=str, default='ConvLSTM-Attention')
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=260)
    return parser.parse_args()


def setup_chinese_font(font_path: str) -> None:
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False


def read_single_band_tif(path: str) -> Tuple[np.ndarray, Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'文件不存在: {path}')
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
    return arr, meta


def normalize_ndvi(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    vmin = np.percentile(finite, 2)
    vmax = np.percentile(finite, 98)
    if vmax <= vmin:
        vmin = float(finite.min())
        vmax = float(finite.max())
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)


def class_distribution_text(label_map: np.ndarray) -> str:
    label_map = label_map.astype(np.int64)
    valid_mask = (label_map >= 0) & (label_map < len(CLASS_NAMES))
    valid_pixels = label_map[valid_mask]
    if valid_pixels.size == 0:
        return '无有效像元'
    counts = np.bincount(valid_pixels.reshape(-1), minlength=len(CLASS_NAMES))
    total = counts.sum()
    return ' | '.join([f'{name}:{counts[i] / total:.1%}' for i, name in enumerate(CLASS_NAMES)])


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    valid_mask = (
        (y_true >= 0) & (y_true < len(CLASS_NAMES)) &
        (y_pred >= 0) & (y_pred < len(CLASS_NAMES))
    )
    if valid_mask.sum() == 0:
        return 0.0, 0.0
    pixel_acc = float((y_true[valid_mask] == y_pred[valid_mask]).mean())
    error_ratio = 1.0 - pixel_acc
    return pixel_acc, error_ratio


def build_output_path(args: argparse.Namespace) -> str:
    if args.output_path:
        return args.output_path
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    filename = f'ppt_triptych_{args.model_name.replace(" ", "_").lower()}_year{args.year}_sample{args.sample_index:04d}.png'
    return os.path.join(DEFAULT_OUTPUT_DIR, filename)


def main() -> None:
    args = parse_args()
    setup_chinese_font(FONT_PATH)

    pred_map, _ = read_single_band_tif(args.pred_tif)
    gt_map, _ = read_single_band_tif(args.gt_tif)
    ref_map, _ = read_single_band_tif(args.reference_tif)

    if pred_map.shape != gt_map.shape or pred_map.shape != ref_map.shape:
        raise ValueError(f'三幅图尺寸不一致: pred={pred_map.shape}, gt={gt_map.shape}, ref={ref_map.shape}')

    pred_map = pred_map.astype(np.int32)
    gt_map = gt_map.astype(np.int32)
    ref_vis = normalize_ndvi(ref_map)
    error_mask = (pred_map != gt_map).astype(np.int32)
    pixel_acc, error_ratio = calc_metrics(gt_map, pred_map)

    fig = plt.figure(figsize=(15.8, 6.8), facecolor='white')
    gs = fig.add_gridspec(2, 4, height_ratios=[14, 2.7], width_ratios=[1, 1, 1, 0.78])

    ax_ref = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_legend = fig.add_subplot(gs[:, 3])
    ax_note = fig.add_subplot(gs[1, 0:3])

    ref_im = ax_ref.imshow(ref_vis, cmap='YlGn')
    ax_ref.set_title('参考底图（NDVI）', fontsize=13, pad=10)
    ax_ref.axis('off')

    gt_im = ax_gt.imshow(gt_map, cmap=DROUGHT_CMAP, norm=DROUGHT_NORM)
    ax_gt.set_title('真实干旱等级', fontsize=13, pad=10)
    ax_gt.axis('off')

    pred_im = ax_pred.imshow(pred_map, cmap=DROUGHT_CMAP, norm=DROUGHT_NORM)
    ax_pred.set_title(f'预测干旱等级\n{args.model_name}', fontsize=13, pad=10)
    ax_pred.axis('off')

    fig.colorbar(ref_im, ax=[ax_ref], location='bottom', shrink=0.82, pad=0.06)
    cbar = fig.colorbar(pred_im, ax=[ax_gt, ax_pred], location='bottom', shrink=0.88, pad=0.06)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(CLASS_NAMES)

    ax_legend.axis('off')
    legend_handles = [Patch(facecolor=color, edgecolor='black', label=name) for color, name in zip(CLASS_COLORS, CLASS_NAMES)]
    error_patch = Patch(facecolor='#6a3d9a', edgecolor='black', label='错分区域（可口头说明）')
    legend_handles.append(error_patch)
    ax_legend.legend(
        handles=legend_handles,
        loc='center',
        frameon=True,
        fontsize=11,
        title='图例',
        title_fontsize=12,
        borderpad=1.0,
        labelspacing=1.0,
    )

    ax_note.axis('off')
    note_text = (
        f'样本编号：{args.sample_index}    年份：{args.year}    像元准确率：{pixel_acc:.2%}    错分比例：{error_ratio:.2%}\n'
        f'真实类别占比：{class_distribution_text(gt_map)}\n'
        '说明：该图展示了单个测试样本块的空间预测结果，可用于答辩中说明模型对干旱空间分布的识别能力。'
    )
    ax_note.text(0.01, 0.82, note_text, fontsize=11, va='top', ha='left')

    fig.suptitle(args.title, fontsize=17, y=0.98, fontweight='bold')
    fig.text(0.5, 0.92, args.subtitle, ha='center', fontsize=11)

    output_path = build_output_path(args)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[OK] 已生成 PPT 三联图: {output_path}')


if __name__ == '__main__':
    main()
