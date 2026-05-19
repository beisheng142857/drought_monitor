import argparse
import copy
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap

NEW_CODE_DIR = '/root/autodl-tmp/zyk_drought_monitor/proposed_attention_optimization'
ROOT_DIR = '/root/autodl-tmp/zyk_drought_monitor'
if NEW_CODE_DIR not in sys.path:
    sys.path.insert(0, NEW_CODE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.chdir(NEW_CODE_DIR)

from configs.config import model_params
from models.baseline.convgru import ConvGRU
from models.baseline.convlstm import ConvLSTM
from models.baseline.traj_gru import TrajGRU

CLASS_NAMES = ['无旱', '轻旱', '中旱', '重/特旱']
FEATURE_NAMES = ['NDVI', 'EVI', 'NDMI', 'NDWI', 'MSAVI', 'VV', 'VH', 'VV/VH', 'VV-VH', 'RVI']
FONT_PATH = '/root/autodl-tmp/zyk_drought_monitor/SimHei.ttf'
FORECAST_INPUT_STEPS = 4
DROUGHT_CMAP = ListedColormap(['#2ca25f', '#fee08b', '#f46d43', '#a50026'])
DROUGHT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], DROUGHT_CMAP.N)
ERROR_CMAP = ListedColormap(['#f7f7f7', '#542788'])
ERROR_NORM = BoundaryNorm([-0.5, 0.5, 1.5], ERROR_CMAP.N)


def setup_chinese_font(font_path: str):
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.sans-serif'] = [name]
        plt.rcParams['axes.unicode_minus'] = False


def find_existing_file(candidate_dirs: List[str], candidate_names: List[str]) -> str:
    for directory in candidate_dirs:
        for name in candidate_names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f'未找到候选文件: {candidate_names}')


def resolve_paths(data_dirs: List[str], label_mode: str, year: int) -> Tuple[str, str]:
    x_path = find_existing_file(data_dirs, [f'forecast_v2_X_{year}.pt'])
    y_names = [f'forecast_v2_Y_{year}.pt', 'forecast_v2_Y.pt'] if label_mode == 'threshold' else [f'forecast_v2_Y_{year}.pt', 'forecast_v2_Y.pt']
    y_path = find_existing_file(data_dirs, y_names)
    return x_path, y_path


def build_forecast_dataset(x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x_tensor[:, :FORECAST_INPUT_STEPS].contiguous(), y_tensor.contiguous()


def infer_model_type_from_ckpt(ckpt_path: str) -> str:
    lower = os.path.basename(ckpt_path).lower()
    if 'forecast_convlstm_attn' in lower:
        return 'convlstm_attn'
    if 'forecast_convlstm_no_attn' in lower:
        return 'convlstm_no_attn'
    if 'forecast_convgru' in lower:
        return 'convgru'
    if 'forecast_traj_gru' in lower or 'forecast_trajgru' in lower:
        return 'traj_gru'
    raise ValueError(f'无法从文件名推断预测模型类型: {ckpt_path}')


def build_model(model_type: str, device: torch.device, actual_channels: int) -> torch.nn.Module:
    if model_type in ['convlstm_attn', 'convlstm_no_attn']:
        cfg = copy.deepcopy(model_params['convlstm']['core'])
        cfg['window_in'] = FORECAST_INPUT_STEPS
        cfg['encoder_params']['input_dim'] = actual_channels
        attn = copy.deepcopy(cfg['input_attn_params']) if model_type == 'convlstm_attn' else None
        if attn is not None:
            attn['input_dim'] = actual_channels
        model = ConvLSTM(cfg['input_size'], cfg['window_in'], cfg['num_layers'], cfg['encoder_params'], attn, device)
    elif model_type == 'convgru':
        cfg = copy.deepcopy(model_params['convgru']['core'])
        cfg['window_in'] = FORECAST_INPUT_STEPS
        cfg['encoder_params']['input_dim'] = actual_channels
        model = ConvGRU(cfg['input_size'], cfg['window_in'], cfg['num_layers'], cfg['encoder_params'], device, cfg['num_classes'])
    else:
        cfg = copy.deepcopy(model_params['traj_gru']['core'])
        cfg['window_in'] = FORECAST_INPUT_STEPS
        cfg['encoder_params']['input_dim'] = actual_channels
        model = TrajGRU(cfg['input_size'], cfg['window_in'], cfg['window_out'], cfg['encoder_params'], cfg['decoder_params'], device, cfg['num_classes'])
    return model.to(device)


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        return obj['state_dict']
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise ValueError(f'无法识别 checkpoint 格式: {path}')


def load_model_from_checkpoint(ckpt_path: str, device: torch.device, actual_channels: int):
    model_type = infer_model_type_from_ckpt(ckpt_path)
    model = build_model(model_type, device, actual_channels)
    model.load_state_dict(load_state_dict(ckpt_path), strict=False)
    model.eval()
    return model, model_type


def extract_attention_map(model: torch.nn.Module) -> Optional[np.ndarray]:
    for attr_path in ['input_attn.saved_attn_weights', 'encoder.0.input_attn.saved_attn_weights']:
        target = model
        ok = True
        for part in attr_path.split('.'):
            if part.isdigit():
                idx = int(part)
                if hasattr(target, '__getitem__'):
                    try:
                        target = target[idx]
                    except Exception:
                        ok = False
                        break
                else:
                    ok = False
                    break
            else:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    ok = False
                    break
        if ok and target is not None:
            arr = target
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            if isinstance(arr, np.ndarray) and arr.ndim == 4:
                return arr[0].mean(axis=0)
    return None


def predict_single(model, x_sample: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with torch.no_grad():
        x = x_sample.unsqueeze(0).float().to(device)
        hidden = model.init_hidden(batch_size=1) if hasattr(model, 'hidden') else None
        logits = model(x=x, hidden=hidden)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        attn_map = extract_attention_map(model)
        return pred, attn_map


def normalize_feature_map(feature_map: np.ndarray) -> np.ndarray:
    finite = feature_map[np.isfinite(feature_map)]
    if finite.size == 0:
        return np.zeros_like(feature_map, dtype=np.float32)
    vmin = np.percentile(finite, 2)
    vmax = np.percentile(finite, 98)
    if vmax <= vmin:
        return np.zeros_like(feature_map, dtype=np.float32)
    return np.clip((feature_map - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)


def class_ratio_text(label_map: np.ndarray) -> str:
    counts = np.bincount(label_map.reshape(-1), minlength=len(CLASS_NAMES))
    total = max(label_map.size, 1)
    return ' | '.join([f'{name}:{counts[i] / total:.1%}' for i, name in enumerate(CLASS_NAMES)])


def sample_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    pixel_acc = float((y_true == y_pred).mean())
    boundary_like_error = float((y_true != y_pred).mean())
    return pixel_acc, boundary_like_error


def plot_label_map(ax, label_map: np.ndarray, title: str):
    im = ax.imshow(label_map, cmap=DROUGHT_CMAP, norm=DROUGHT_NORM)
    ax.set_title(title)
    ax.axis('off')
    return im


def plot_compare_figure(ref_map: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, attn_map: Optional[np.ndarray], output_path: str, feature_name: str, model_name: str, sample_index: int, year: int):
    error_mask = (y_true != y_pred).astype(np.int32)
    pixel_acc, error_ratio = sample_metrics(y_true, y_pred)

    use_attn = attn_map is not None
    ncols = 5 if use_attn else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    ref_im = axes[0].imshow(normalize_feature_map(ref_map), cmap='YlGn')
    axes[0].set_title(f'参考底图（{feature_name}）')
    axes[0].axis('off')

    drought_im = plot_label_map(axes[1], y_true, '真实监测旱情图')
    plot_label_map(axes[2], y_pred, f'{model_name} 预测图')

    err_im = axes[3].imshow(error_mask, cmap=ERROR_CMAP, norm=ERROR_NORM)
    axes[3].set_title('错分掩膜图')
    axes[3].axis('off')

    if use_attn:
        axes[4].imshow(normalize_feature_map(ref_map), cmap='gray')
        attn_im = axes[4].imshow(attn_map, cmap='jet', alpha=0.55)
        axes[4].set_title('注意力聚焦热力图')
        axes[4].axis('off')
    else:
        attn_im = None

    fig.colorbar(ref_im, ax=[axes[0]], location='bottom', shrink=0.82, pad=0.08)
    drought_cbar = fig.colorbar(drought_im, ax=axes[1:3], location='bottom', shrink=0.82, pad=0.08)
    drought_cbar.set_ticks([0, 1, 2, 3])
    drought_cbar.set_ticklabels(CLASS_NAMES)
    err_cbar = fig.colorbar(err_im, ax=[axes[3]], location='bottom', shrink=0.82, pad=0.08)
    err_cbar.set_ticks([0, 1])
    err_cbar.set_ticklabels(['预测正确', '预测错误'])
    if attn_im is not None:
        fig.colorbar(attn_im, ax=[axes[4]], location='bottom', shrink=0.82, pad=0.08)

    title = (
        f'样本 {sample_index} 干旱图斑预测与真实监测对比 | year={year} | model={model_name}\n'
        f'像元准确率={pixel_acc:.2%} | 错分比例={error_ratio:.2%}\n'
        f'真实类别占比: {class_ratio_text(y_true)}'
    )
    fig.suptitle(title, fontsize=13)
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='生成 6.1.1 用的预测图斑-真实监测环境比对图')
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--feature_time_index', type=int, default=-1)
    parser.add_argument('--feature_channel_index', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dirs', nargs='+', default=['/root/autodl-tmp/zyk_drought_monitor/data_V2', '/root/autodl-tmp/data_proc', '/root/autodl-tmp/data_proc/data_proc'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='/root/autodl-tmp/zyk_drought_monitor/results/forecast_compare_V2/visual_compare_611.png')
    args = parser.parse_args()

    setup_chinese_font(FONT_PATH)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    device = torch.device(args.device)

    x_path, y_path = resolve_paths(args.data_dirs, args.label_mode, args.year)
    x_all_raw = torch.load(x_path, map_location='cpu')
    y_all_raw = torch.load(y_path, map_location='cpu')
    x_all, y_all = build_forecast_dataset(x_all_raw, y_all_raw)
    actual_channels = x_all.shape[2]

    if args.sample_index < 0 or args.sample_index >= x_all.shape[0]:
        raise IndexError(f'sample_index 越界，当前样本数为 {x_all.shape[0]}')

    x_sample = x_all[args.sample_index]
    y_true = y_all[args.sample_index].numpy()
    time_index = args.feature_time_index if args.feature_time_index >= 0 else x_sample.shape[0] + args.feature_time_index
    time_index = max(0, min(time_index, x_sample.shape[0] - 1))
    ref_map = x_sample[time_index, args.feature_channel_index].numpy()
    feature_name = FEATURE_NAMES[args.feature_channel_index] if 0 <= args.feature_channel_index < len(FEATURE_NAMES) else f'Channel-{args.feature_channel_index}'

    model, model_name = load_model_from_checkpoint(args.checkpoint, device, actual_channels)
    y_pred, attn_map = predict_single(model, x_sample, device)

    plot_compare_figure(ref_map, y_true, y_pred, attn_map, args.output_path, feature_name, model_name, args.sample_index, args.year)
    print(f'[OK] 已保存 6.1.1 比对图: {args.output_path}')


if __name__ == '__main__':
    main()
