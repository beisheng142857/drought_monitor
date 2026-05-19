import argparse
import copy
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm

code_dir = '/root/autodl-tmp/zyk_drought_monitor'
if code_dir not in sys.path:
    sys.path.append(code_dir)
os.chdir(code_dir)

from configs.config import model_params
from models.baseline.convgru import ConvGRU
from models.baseline.convlstm import ConvLSTM
from models.baseline.traj_gru import TrajGRU

CLASS_NAMES = ['无旱', '轻旱', '中旱', '重/特旱']
FEATURE_NAMES = ['NDVI', 'EVI', 'NDMI', 'NDWI', 'MSAVI', 'VV', 'VH', 'VVVH', 'VVDIFFVH', 'RVI']
FONT_PATH = '/root/autodl-tmp/zyk_drought_monitor/SimHei.ttf'
DROUGHT_CMAP = ListedColormap(['#2ca25f', '#fee08b', '#f46d43', '#a50026'])
DROUGHT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], DROUGHT_CMAP.N)


def setup_chinese_font(font_path: str):
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_prop = font_manager.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        print(f'[OK] 已加载中文字体: {font_name}')
    else:
        print(f'[WARN] 中文字体文件不存在: {font_path}')


def find_existing_file(candidate_dirs: List[str], candidate_names: List[str]) -> str:
    checked_paths = []
    for directory in candidate_dirs:
        for name in candidate_names:
            path = os.path.join(directory, name)
            checked_paths.append(path)
            if os.path.exists(path):
                return path
    checked_text = '\n'.join(f'  - {path}' for path in checked_paths)
    raise FileNotFoundError(f'未找到任何候选文件，请检查数据路径或文件名：\n{checked_text}')


def resolve_paths(data_dirs: List[str], label_mode: str, year: int) -> Tuple[str, str]:
    x_candidates = [f'dataset_X_{year}.pt', f'dataset_X_{year}_new.pt']
    x_path = find_existing_file(data_dirs, x_candidates)
    if label_mode == 'threshold':
        y_candidates = [
            f'dataset_Y_{year}_threshold.pt',
            f'dataset_Y_{year}_new_threshold.pt',
            'dataset_Y_threshold.pt',
        ]
    else:
        y_candidates = [f'dataset_Y_{year}.pt', f'dataset_Y_{year}_new.pt', 'dataset_Y.pt']
    y_path = find_existing_file(data_dirs, y_candidates)
    return x_path, y_path


def infer_model_type_from_ckpt(ckpt_path: str) -> str:
    lower = os.path.basename(ckpt_path).lower()
    if 'convlstm_attn' in lower:
        return 'convlstm_attn'
    if 'convlstm_no_attn' in lower:
        return 'convlstm_no_attn'
    if 'convgru' in lower:
        return 'convgru'
    if 'traj_gru' in lower or 'trajgru' in lower:
        return 'traj_gru'
    raise ValueError(f'无法从文件名推断模型类型: {ckpt_path}')


def build_monitor_input(x_tensor: torch.Tensor, required_steps: int) -> torch.Tensor:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 形状应为 (Batch, Time, Channels, H, W)，当前为 {x_tensor.shape}')
    if x_tensor.shape[1] < required_steps:
        raise ValueError(f'当前时间步数为 {x_tensor.shape[1]}，少于模型需要的 {required_steps} 个时间步。')
    return x_tensor[:, -required_steps:, :, :, :].contiguous()



def build_model(model_type: str, device: torch.device, actual_channels: int, window_in: int) -> torch.nn.Module:
    if model_type in ['convlstm_attn', 'convlstm_no_attn']:
        cfg = copy.deepcopy(model_params['convlstm']['core'])
        cfg['window_in'] = window_in
        cfg['encoder_params']['input_dim'] = actual_channels
        input_attn_params = copy.deepcopy(cfg['input_attn_params']) if model_type == 'convlstm_attn' else None
        if input_attn_params is not None:
            input_attn_params['input_dim'] = 1
        model = ConvLSTM(
            input_size=cfg['input_size'],
            window_in=cfg['window_in'],
            num_layers=cfg['num_layers'],
            encoder_params=cfg['encoder_params'],
            input_attn_params=input_attn_params,
            device=device,
        )
    elif model_type == 'convgru':
        cfg = copy.deepcopy(model_params['convgru']['core'])
        cfg['window_in'] = window_in
        cfg['encoder_params']['input_dim'] = actual_channels
        model = ConvGRU(
            input_size=cfg['input_size'],
            window_in=cfg['window_in'],
            num_layers=cfg['num_layers'],
            encoder_params=cfg['encoder_params'],
            num_classes=cfg['num_classes'],
            device=device,
        )
    elif model_type == 'traj_gru':
        cfg = copy.deepcopy(model_params['traj_gru']['core'])
        cfg['window_in'] = window_in
        cfg['encoder_params']['input_dim'] = actual_channels
        model = TrajGRU(
            input_size=cfg['input_size'],
            window_in=cfg['window_in'],
            window_out=cfg['window_out'],
            encoder_params=cfg['encoder_params'],
            decoder_params=cfg['decoder_params'],
            num_classes=cfg['num_classes'],
            device=device,
        )
    else:
        raise ValueError(f'不支持的模型类型: {model_type}')
    return model.to(device)


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        return obj['state_dict']
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise ValueError(f'无法识别 checkpoint 格式: {path}')


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    actual_channels: int,
    window_in: int,
) -> Tuple[torch.nn.Module, str]:
    model_type = infer_model_type_from_ckpt(ckpt_path)
    model = build_model(model_type, device, actual_channels, window_in)
    state_dict = load_state_dict(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, model_type


def predict_single(model, x_sample: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = x_sample.unsqueeze(0).float().to(device)
        hidden = model.init_hidden(batch_size=1) if hasattr(model, 'hidden') else None
        logits = model(x=x, hidden=hidden)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return pred


def plot_reference_feature(ax, feature_map: np.ndarray, title: str):
    finite_mask = np.isfinite(feature_map)
    if not np.any(finite_mask):
        vis_map = np.zeros_like(feature_map, dtype=np.float32)
    else:
        valid = feature_map[finite_mask]
        vmin = np.percentile(valid, 2)
        vmax = np.percentile(valid, 98)
        if vmax <= vmin:
            vmin = float(valid.min())
            vmax = float(valid.max())
        if vmax <= vmin:
            vis_map = np.zeros_like(feature_map, dtype=np.float32)
        else:
            vis_map = np.clip((feature_map - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    im = ax.imshow(vis_map, cmap='YlGn')
    ax.set_title(title)
    ax.axis('off')
    return im


def plot_label_map(ax, label_map: np.ndarray, title: str):
    im = ax.imshow(label_map, cmap=DROUGHT_CMAP, norm=DROUGHT_NORM)
    ax.set_title(title)
    ax.axis('off')
    return im



def summarize_label_distribution(label_map: np.ndarray) -> List[Tuple[str, int, float]]:
    total = label_map.size
    counts = np.bincount(label_map.reshape(-1), minlength=len(CLASS_NAMES))
    summary = []
    for idx, class_name in enumerate(CLASS_NAMES):
        cnt = int(counts[idx])
        ratio = cnt / total if total > 0 else 0.0
        summary.append((class_name, cnt, ratio))
    return summary



def build_distribution_text(prefix: str, label_map: np.ndarray) -> str:
    summary = summarize_label_distribution(label_map)
    lines = [prefix]
    for class_name, cnt, ratio in summary:
        lines.append(f'  {class_name}: {cnt} ({ratio:.2%})')
    return '\n'.join(lines)



def safe_feature_name(channel_index: int) -> str:
    if 0 <= channel_index < len(FEATURE_NAMES):
        return FEATURE_NAMES[channel_index]
    return f'Channel-{channel_index}'



def parse_channel_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(',') if part.strip()]



def resolve_time_index(time_index: int, total_steps: int) -> int:
    idx = time_index if time_index >= 0 else total_steps + time_index
    if idx < 0 or idx >= total_steps:
        raise IndexError(f'时间索引越界: {time_index}，当前时间步数={total_steps}')
    return idx



def find_latest_nonzero_time(x_sample: torch.Tensor, channel_index: int, prefer_time_index: int) -> int:
    total_steps = x_sample.shape[0]
    start_idx = resolve_time_index(prefer_time_index, total_steps)
    for t in range(start_idx, -1, -1):
        channel_map = x_sample[t, channel_index]
        if torch.any(channel_map != 0):
            return t
    return start_idx



def print_feature_stats(feature_map: np.ndarray, feature_name: str):
    finite = feature_map[np.isfinite(feature_map)]
    if finite.size == 0:
        print(f'[WARN] {feature_name} 无有效像元。')
        return
    q2, q50, q98 = np.percentile(finite, [2, 50, 98])
    print(
        f'[STAT] {feature_name}: min={finite.min():.6f}, max={finite.max():.6f}, '
        f'p2={q2:.6f}, p50={q50:.6f}, p98={q98:.6f}'
    )


def main():
    parser = argparse.ArgumentParser(description='可视化同一样本在不同监测模型下的空间预测图')
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--feature_time_index', type=int, default=-1, help='参考底图时间步，默认最后一个月')
    parser.add_argument('--feature_channel_index', type=int, default=0, help='默认显示 NDVI 通道')
    parser.add_argument('--auto_fallback_nonzero_time', action='store_true', help='若指定时间步某通道全0，自动回退到最近有数据的时间步')
    parser.add_argument(
        '--extra_feature_channels',
        type=str,
        default='',
        help='额外展示的通道索引，逗号分隔，例如 "5,6" 表示 VV 和 VH',
    )
    parser.add_argument('--show_distribution', action='store_true', help='在图中显示真实/预测类别分布摘要')
    parser.add_argument('--distribution_as_panel', action='store_true', help='将分布统计单独作为一个面板显示')
    parser.add_argument('--max_cols', type=int, default=3, help='每行最多展示多少张图')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--data_dirs',
        nargs='+',
        default=[
            '/root/autodl-tmp/zyk_drought_monitor/data/data_proc',
            '/root/autodl-tmp/data_proc',
            '/root/autodl-tmp/data_proc/data_proc',
            '/content/drive/MyDrive/GEE_Drought_Project/data_proc',
            '/content/drive/MyDrive/drought_monitor/data_proc'
        ],
    )
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        required=True,
        help='传入多个监测模型 checkpoint 路径',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/root/autodl-tmp/zyk_drought_monitor/results/monitor_compare/spatial_prediction_compare.png',
    )
    args = parser.parse_args()

    setup_chinese_font(FONT_PATH)

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    x_path, y_path = resolve_paths(args.data_dirs, args.label_mode, args.year)
    print(f'加载特征: {x_path}')
    print(f'加载标签: {y_path}')
    x_all = torch.load(x_path, map_location='cpu')
    y_all = torch.load(y_path, map_location='cpu')
    common_input_steps = min(model_params['convlstm']['core']['window_in'], x_all.shape[1])
    x_all = build_monitor_input(x_all, common_input_steps)
    actual_channels = x_all.shape[2]
    print(f'样本可视化按 {common_input_steps} 个时间步运行，输入通道数: {actual_channels}')

    if args.sample_index < 0 or args.sample_index >= x_all.shape[0]:
        raise IndexError(f'sample_index 越界，当前样本数为 {x_all.shape[0]}')

    x_sample = x_all[args.sample_index]
    y_true = y_all[args.sample_index].numpy()
    extra_channels = parse_channel_list(args.extra_feature_channels)

    selected_time_for_channel: Dict[int, int] = {}
    channels_to_use = [args.feature_channel_index] + extra_channels
    for ch in channels_to_use:
        t_idx = resolve_time_index(args.feature_time_index, x_sample.shape[0])
        if args.auto_fallback_nonzero_time:
            t_idx = find_latest_nonzero_time(x_sample, ch, args.feature_time_index)
        selected_time_for_channel[ch] = t_idx

    ref_time_idx = selected_time_for_channel[args.feature_channel_index]
    ref_map = x_sample[ref_time_idx, args.feature_channel_index].numpy()

    ref_feature_name = safe_feature_name(args.feature_channel_index)
    print_feature_stats(ref_map, f'{ref_feature_name}(t={ref_time_idx})')
    for ch in extra_channels:
        ch_name = safe_feature_name(ch)
        t_idx = selected_time_for_channel[ch]
        feature_map = x_sample[t_idx, ch].numpy()
        print_feature_stats(feature_map, f'{ch_name}(t={t_idx})')

    predictions = []
    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            print(f'[SKIP] 文件不存在: {ckpt}')
            continue
        model, model_type = load_model_from_checkpoint(ckpt, device, actual_channels, common_input_steps)
        pred_map = predict_single(model, x_sample, device)
        predictions.append((model_type, pred_map))
        print(f'[OK] 已生成 {model_type} 的预测图')

    if not predictions:
        raise RuntimeError('没有成功生成任何模型预测图。')

    base_panels = 2 + len(predictions)
    show_dist_panel = args.show_distribution and args.distribution_as_panel
    n_panels = base_panels + len(extra_channels) + (1 if show_dist_panel else 0)
    max_cols = max(1, args.max_cols)
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.9 * nrows), constrained_layout=True)
    if isinstance(axes, np.ndarray):
        axes_list = list(axes.flatten())
    else:
        axes_list = [axes]

    panel_idx = 0
    ref_feature_name = safe_feature_name(args.feature_channel_index)
    ref_im = plot_reference_feature(
        axes_list[panel_idx],
        ref_map,
        f'参考底图 ({ref_feature_name}, t={ref_time_idx})',
    )
    panel_idx += 1

    for ch in extra_channels:
        if ch < 0 or ch >= actual_channels:
            raise IndexError(f'extra_feature_channels 存在越界通道: {ch}，当前通道数={actual_channels}')
        t_idx = selected_time_for_channel[ch]
        feature_map = x_sample[t_idx, ch].numpy()
        ch_name = safe_feature_name(ch)
        plot_reference_feature(axes_list[panel_idx], feature_map, f'特征底图 ({ch_name}, t={t_idx})')
        panel_idx += 1

    label_im = plot_label_map(axes_list[panel_idx], y_true, '真实旱情图')
    panel_idx += 1

    for model_type, pred_map in predictions:
        plot_label_map(axes_list[panel_idx], pred_map, f'{model_type} 预测图')
        panel_idx += 1

    if show_dist_panel:
        text_blocks = [build_distribution_text('真实标签分布:', y_true)]
        for model_type, pred_map in predictions:
            text_blocks.append(build_distribution_text(f'{model_type} 预测分布:', pred_map))
        text = '\n\n'.join(text_blocks)
        info_ax = axes_list[panel_idx]
        info_ax.axis('off')
        info_ax.set_title('统计信息')
        info_ax.text(0.01, 0.98, text, ha='left', va='top', fontsize=10)
        panel_idx += 1

    for ax in axes_list[panel_idx:]:
        ax.axis('off')

    fig.colorbar(ref_im, ax=[axes_list[0]], location='bottom', shrink=0.8, pad=0.06)
    map_axes = [ax for ax in axes_list[:panel_idx] if ax.get_title().endswith('图') and '底图' not in ax.get_title()]
    if map_axes:
        cbar = fig.colorbar(label_im, ax=map_axes, location='bottom', shrink=0.86, pad=0.06)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(CLASS_NAMES)

    title = (
        f'样本 {args.sample_index} 监测结果对比 | year={args.year} | label_mode={args.label_mode} '
        f'| time_steps={common_input_steps} | channels={actual_channels}'
    )
    fig.suptitle(title, fontsize=14)

    if args.show_distribution and (not args.distribution_as_panel):
        text_blocks = [build_distribution_text('真实标签分布:', y_true)]
        for model_type, pred_map in predictions:
            text_blocks.append(build_distribution_text(f'{model_type} 预测分布:', pred_map))
        text = '\n\n'.join(text_blocks)
        fig.text(0.01, 0.01, text, ha='left', va='bottom', fontsize=9)

    plt.savefig(args.output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 已保存空间预测对比图: {args.output_path}')


if __name__ == '__main__':
    main()
