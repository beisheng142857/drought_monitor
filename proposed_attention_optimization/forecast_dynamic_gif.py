import argparse
import copy
import os
import sys
from typing import Dict, List, Tuple

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    imageio = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

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


def predict_single(model, x_sample: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = x_sample.unsqueeze(0).float().to(device)
        hidden = model.init_hidden(batch_size=1) if hasattr(model, 'hidden') else None
        logits = model(x=x, hidden=hidden)
        return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()


def normalize_feature_map(feature_map: np.ndarray) -> np.ndarray:
    finite = feature_map[np.isfinite(feature_map)]
    if finite.size == 0:
        return np.zeros_like(feature_map, dtype=np.float32)
    vmin = np.percentile(finite, 2)
    vmax = np.percentile(finite, 98)
    if vmax <= vmin:
        vmin = float(finite.min())
        vmax = float(finite.max())
    if vmax <= vmin:
        return np.zeros_like(feature_map, dtype=np.float32)
    return np.clip((feature_map - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)


def summarize_label_distribution(label_map: np.ndarray) -> str:
    counts = np.bincount(label_map.reshape(-1), minlength=len(CLASS_NAMES))
    total = max(label_map.size, 1)
    parts = []
    for i, name in enumerate(CLASS_NAMES):
        parts.append(f'{name}:{counts[i] / total:.1%}')
    return ' | '.join(parts)


def render_frame(ref_map: np.ndarray, drought_map: np.ndarray, frame_title: str, save_path: str, feature_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    ref_im = axes[0].imshow(normalize_feature_map(ref_map), cmap='YlGn')
    axes[0].set_title(f'参考底图（{feature_name}）')
    axes[0].axis('off')

    drought_im = axes[1].imshow(drought_map, cmap=DROUGHT_CMAP, norm=DROUGHT_NORM)
    axes[1].set_title('干旱等级图')
    axes[1].axis('off')

    fig.colorbar(ref_im, ax=[axes[0]], location='bottom', shrink=0.85, pad=0.08)
    cbar = fig.colorbar(drought_im, ax=[axes[1]], location='bottom', shrink=0.85, pad=0.08)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(CLASS_NAMES)

    dist_text = summarize_label_distribution(drought_map)
    fig.suptitle(f'{frame_title}\n{dist_text}', fontsize=13)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def build_gif_from_frames(frame_paths: List[str], gif_path: str, duration: float):
    if imageio is not None:
        images = [imageio.imread(path) for path in frame_paths]
        imageio.mimsave(gif_path, images, duration=duration, loop=0)
        return

    if Image is not None:
        frames = [Image.open(path).convert('P', palette=Image.ADAPTIVE) for path in frame_paths]
        if not frames:
            raise ValueError('没有可用于合成 GIF 的帧。')
        first_frame, rest_frames = frames[0], frames[1:]
        first_frame.save(
            gif_path,
            save_all=True,
            append_images=rest_frames,
            duration=max(int(duration * 1000), 1),
            loop=0,
        )
        return

    raise ModuleNotFoundError('既未安装 imageio，也未安装 Pillow，无法合成 GIF。请执行 pip install imageio 或 pip install pillow。')


def main():
    parser = argparse.ArgumentParser(description='生成连续时空干旱演变动态 GIF（真实序列或预测序列）')
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--start_index', type=int, default=0, help='起始样本索引')
    parser.add_argument('--num_frames', type=int, default=6, help='连续帧数')
    parser.add_argument('--mode', type=str, default='gt', choices=['gt', 'pred'], help='gt=真实标签序列, pred=模型预测序列')
    parser.add_argument('--checkpoint', type=str, default='', help='mode=pred 时必须提供')
    parser.add_argument('--feature_channel_index', type=int, default=0, help='默认使用 NDVI 作为参考底图')
    parser.add_argument('--feature_time_index', type=int, default=-1, help='默认显示输入窗口最后一个时间步')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dirs', nargs='+', default=['/root/autodl-tmp/zyk_drought_monitor/data_V2', '/root/autodl-tmp/data_proc', '/root/autodl-tmp/data_proc/data_proc'])
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/zyk_drought_monitor/drought_outputs/dynamic_forecast')
    parser.add_argument('--gif_name', type=str, default='drought_dynamic_evolution_forecast.gif')
    parser.add_argument('--frame_duration', type=float, default=0.9, help='GIF 每帧持续秒数')
    args = parser.parse_args()

    setup_chinese_font(FONT_PATH)
    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    x_path, y_path = resolve_paths(args.data_dirs, args.label_mode, args.year)
    x_all_raw = torch.load(x_path, map_location='cpu')
    y_all_raw = torch.load(y_path, map_location='cpu')
    x_all, y_all = build_forecast_dataset(x_all_raw, y_all_raw)

    total_samples = x_all.shape[0]
    end_index = min(args.start_index + args.num_frames, total_samples)
    if args.start_index < 0 or args.start_index >= total_samples:
        raise IndexError(f'start_index 越界，当前样本数为 {total_samples}')
    if end_index <= args.start_index:
        raise ValueError('没有可用于生成动画的样本帧')

    model = None
    model_type = None
    device = torch.device(args.device)
    if args.mode == 'pred':
        if not args.checkpoint:
            raise ValueError('mode=pred 时必须提供 --checkpoint')
        actual_channels = x_all.shape[2]
        model, model_type = load_model_from_checkpoint(args.checkpoint, device, actual_channels)

    frame_paths = []
    feature_name = FEATURE_NAMES[args.feature_channel_index] if 0 <= args.feature_channel_index < len(FEATURE_NAMES) else f'Channel-{args.feature_channel_index}'

    for idx in range(args.start_index, end_index):
        x_sample = x_all[idx]
        y_true = y_all[idx].numpy()
        time_idx = args.feature_time_index if args.feature_time_index >= 0 else x_sample.shape[0] + args.feature_time_index
        time_idx = max(0, min(time_idx, x_sample.shape[0] - 1))
        ref_map = x_sample[time_idx, args.feature_channel_index].numpy()

        if args.mode == 'gt':
            drought_map = y_true
            frame_title = f'真实干旱演变 | year={args.year} | frame={idx}'
        else:
            drought_map = predict_single(model, x_sample, device)
            frame_title = f'预测干旱演变 | {model_type} | year={args.year} | frame={idx}'

        frame_path = os.path.join(frames_dir, f'frame_{idx:03d}.png')
        render_frame(ref_map, drought_map, frame_title, frame_path, feature_name)
        frame_paths.append(frame_path)
        print(f'[OK] 已生成帧: {frame_path}')

    gif_path = os.path.join(args.output_dir, args.gif_name)
    build_gif_from_frames(frame_paths, gif_path, args.frame_duration)
    print(f'[OK] 已生成 GIF: {gif_path}')


if __name__ == '__main__':
    main()
