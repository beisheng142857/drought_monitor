import argparse, copy, os, sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap

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
FORECAST_INPUT_STEPS = 4
FORECAST_TARGET_MONTH_INDEX = 4
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
    x_path = find_existing_file(data_dirs, [f'dataset_X_{year}.pt'])
    y_names = [f'dataset_Y_{year}_threshold.pt', 'dataset_Y_threshold.pt'] if label_mode == 'threshold' else [f'dataset_Y_{year}.pt', 'dataset_Y.pt']
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


def build_model(model_type: str, device: torch.device) -> torch.nn.Module:
    if model_type in ['convlstm_attn', 'convlstm_no_attn']:
        cfg = copy.deepcopy(model_params['convlstm']['core'])
        cfg['window_in'] = FORECAST_INPUT_STEPS
        attn = cfg['input_attn_params'] if model_type == 'convlstm_attn' else None
        if attn is not None:
            attn['input_dim'] = 1
        model = ConvLSTM(cfg['input_size'], cfg['window_in'], cfg['num_layers'], cfg['encoder_params'], attn, device)
    elif model_type == 'convgru':
        cfg = copy.deepcopy(model_params['convgru']['core'])
        cfg['window_in'] = FORECAST_INPUT_STEPS
        model = ConvGRU(cfg['input_size'], cfg['window_in'], cfg['num_layers'], cfg['encoder_params'], device, cfg['num_classes'])
    else:
        cfg = copy.deepcopy(model_params['traj_gru']['core'])
        cfg['window_in'] = FORECAST_INPUT_STEPS
        model = TrajGRU(cfg['input_size'], cfg['window_in'], cfg['window_out'], cfg['encoder_params'], cfg['decoder_params'], device, cfg['num_classes'])
    return model.to(device)


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        return obj['state_dict']
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise ValueError(f'无法识别 checkpoint 格式: {path}')


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    model_type = infer_model_type_from_ckpt(ckpt_path)
    model = build_model(model_type, device)
    model.load_state_dict(load_state_dict(ckpt_path), strict=False)
    model.eval()
    return model, model_type


def predict_single(model, x_sample: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = x_sample.unsqueeze(0).float().to(device)
        hidden = model.init_hidden(batch_size=1) if hasattr(model, 'hidden') else None
        logits = model(x=x, hidden=hidden)
        return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()


def plot_reference_feature(ax, feature_map: np.ndarray, title: str):
    im = ax.imshow(feature_map, cmap='YlGn'); ax.set_title(title); ax.axis('off'); return im


def plot_label_map(ax, label_map: np.ndarray, title: str):
    im = ax.imshow(label_map, cmap=DROUGHT_CMAP, norm=DROUGHT_NORM); ax.set_title(title); ax.axis('off'); return im


def main():
    parser = argparse.ArgumentParser(description='可视化同一样本在不同旱情预测模型下的空间预测图')
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--feature_time_index', type=int, default=-1, help='参考底图时间步，默认输入窗口最后一个月')
    parser.add_argument('--feature_channel_index', type=int, default=0, help='默认显示 NDVI 通道')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dirs', nargs='+', default=['/root/autodl-tmp/data_proc', '/root/autodl-tmp/data_proc/data_proc', '/content/drive/MyDrive/GEE_Drought_Project/data_proc', '/content/drive/MyDrive/drought_monitor/data_proc'])
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output_path', type=str, default='/root/autodl-tmp/zyk_drought_monitor/results/forecast_compare/forecast_spatial_prediction_compare.png')
    args = parser.parse_args()
    setup_chinese_font(FONT_PATH)
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    x_path, y_path = resolve_paths(args.data_dirs, args.label_mode, args.year)
    x_all_raw, y_all_raw = torch.load(x_path, map_location='cpu'), torch.load(y_path, map_location='cpu')
    x_all, y_all = build_forecast_dataset(x_all_raw, y_all_raw)
    if args.sample_index < 0 or args.sample_index >= x_all.shape[0]:
        raise IndexError(f'sample_index 越界，当前样本数为 {x_all.shape[0]}')
    x_sample = x_all[args.sample_index]; y_true = y_all[args.sample_index].numpy(); ref_map = x_sample[args.feature_time_index, args.feature_channel_index].numpy()
    preds = []
    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            continue
        model, model_type = load_model_from_checkpoint(ckpt, device)
        preds.append((model_type, predict_single(model, x_sample, device)))
    if not preds: raise RuntimeError('没有成功生成任何预测图。')
    ncols = 2 + len(preds)
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.8), constrained_layout=True)
    if ncols == 1: axes = [axes]
    ref_im = plot_reference_feature(axes[0], ref_map, '参考底图（输入窗口末月 NDVI）')
    label_im = plot_label_map(axes[1], y_true, '真实未来旱情图')
    for idx, (model_type, pred_map) in enumerate(preds, start=2):
        plot_label_map(axes[idx], pred_map, f'{model_type} 预测图')
    fig.colorbar(ref_im, ax=[axes[0]], location='bottom', shrink=0.8, pad=0.08)
    cbar = fig.colorbar(label_im, ax=axes[1:], location='bottom', shrink=0.8, pad=0.08)
    cbar.set_ticks([0, 1, 2, 3]); cbar.set_ticklabels(CLASS_NAMES)
    fig.suptitle(f'样本 {args.sample_index} 的旱情预测结果对比 | year={args.year} | label_mode={args.label_mode}', fontsize=14)
    plt.savefig(args.output_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'[OK] 已保存预测空间对比图: {args.output_path}')


if __name__ == '__main__':
    main()
