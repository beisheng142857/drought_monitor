import argparse
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
    x_path = find_existing_file(data_dirs, [f'dataset_X_{year}.pt'])
    if label_mode == 'threshold':
        y_candidates = [f'dataset_Y_{year}_threshold.pt', 'dataset_Y_threshold.pt']
    else:
        y_candidates = [f'dataset_Y_{year}.pt', 'dataset_Y.pt']
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


def build_model(model_type: str, device: torch.device) -> torch.nn.Module:
    if model_type in ['convlstm_attn', 'convlstm_no_attn']:
        cfg = model_params['convlstm']['core']
        input_attn_params = cfg['input_attn_params'] if model_type == 'convlstm_attn' else None
        model = ConvLSTM(
            input_size=cfg['input_size'],
            window_in=cfg['window_in'],
            num_layers=cfg['num_layers'],
            encoder_params=cfg['encoder_params'],
            input_attn_params=input_attn_params,
            device=device,
        )
    elif model_type == 'convgru':
        cfg = model_params['convgru']['core']
        model = ConvGRU(
            input_size=cfg['input_size'],
            window_in=cfg['window_in'],
            num_layers=cfg['num_layers'],
            encoder_params=cfg['encoder_params'],
            num_classes=cfg['num_classes'],
            device=device,
        )
    elif model_type == 'traj_gru':
        cfg = model_params['traj_gru']['core']
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


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, str]:
    model_type = infer_model_type_from_ckpt(ckpt_path)
    model = build_model(model_type, device)
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
    im = ax.imshow(feature_map, cmap='YlGn')
    ax.set_title(title)
    ax.axis('off')
    return im


def plot_label_map(ax, label_map: np.ndarray, title: str):
    im = ax.imshow(label_map, cmap=DROUGHT_CMAP, norm=DROUGHT_NORM)
    ax.set_title(title)
    ax.axis('off')
    return im


def main():
    parser = argparse.ArgumentParser(description='可视化同一样本在不同监测模型下的空间预测图')
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--feature_time_index', type=int, default=-1, help='参考底图时间步，默认最后一个月')
    parser.add_argument('--feature_channel_index', type=int, default=0, help='默认显示 NDVI 通道')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--data_dirs',
        nargs='+',
        default=[
            '/root/autodl-tmp/data_proc',
            '/root/autodl-tmp/data_proc/data_proc',
            '/content/drive/MyDrive/GEE_Drought_Project/data_proc',
            '/content/drive/MyDrive/drought_monitor/data_proc',
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

    if args.sample_index < 0 or args.sample_index >= x_all.shape[0]:
        raise IndexError(f'sample_index 越界，当前样本数为 {x_all.shape[0]}')

    x_sample = x_all[args.sample_index]
    y_true = y_all[args.sample_index].numpy()
    ref_map = x_sample[args.feature_time_index, args.feature_channel_index].numpy()

    predictions = []
    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            print(f'[SKIP] 文件不存在: {ckpt}')
            continue
        model, model_type = load_model_from_checkpoint(ckpt, device)
        pred_map = predict_single(model, x_sample, device)
        predictions.append((model_type, pred_map))
        print(f'[OK] 已生成 {model_type} 的预测图')

    if not predictions:
        raise RuntimeError('没有成功生成任何模型预测图。')

    ncols = 2 + len(predictions)
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    ref_im = plot_reference_feature(axes[0], ref_map, '参考底图 (NDVI)')
    label_im = plot_label_map(axes[1], y_true, '真实旱情图')

    for idx, (model_type, pred_map) in enumerate(predictions, start=2):
        plot_label_map(axes[idx], pred_map, f'{model_type} 预测图')

    fig.colorbar(ref_im, ax=[axes[0]], location='bottom', shrink=0.8, pad=0.08)
    cbar = fig.colorbar(label_im, ax=axes[1:], location='bottom', shrink=0.8, pad=0.08)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(CLASS_NAMES)

    fig.suptitle(
        f'样本 {args.sample_index} 的监测结果对比 | year={args.year} | label_mode={args.label_mode}',
        fontsize=14,
    )
    plt.savefig(args.output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 已保存空间预测对比图: {args.output_path}')


if __name__ == '__main__':
    main()
