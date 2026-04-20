import argparse, copy, os, sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

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


def resolve_test_paths(data_dirs: List[str], label_mode: str, test_year: int) -> Tuple[str, str]:
    # x_path = find_existing_file(data_dirs, [f'dataset_X_{test_year}.pt'])
    # y_names = [f'dataset_Y_{test_year}_threshold.pt', 'dataset_Y_threshold.pt'] if label_mode == 'threshold' else [f'dataset_Y_{test_year}.pt', 'dataset_Y.pt']
    
    x_path = find_existing_file(data_dirs, [f'forecast_v2_X_{test_year}.pt'])
    y_names = [f'forecast_v2_Y_{test_year}.pt', 'forecast_v2_Y.pt'] if label_mode == 'threshold' else [f'forecast_v2_Y_{test_year}.pt', 'forecast_v2_Y.pt']

    y_path = find_existing_file(data_dirs, y_names)
    return x_path, y_path


def build_forecast_dataset(x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_tensor.shape[1] <= FORECAST_TARGET_MONTH_INDEX:
        raise ValueError('时间步不足以构造预测样本')
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
            attn['input_dim'] = cfg['encoder_params']['input_dim']
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


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return cm / rs


def evaluate_checkpoint(ckpt_path: str, test_loader: DataLoader, device: torch.device):
    model, model_type = load_model_from_checkpoint(ckpt_path, device)
    criterion = nn.CrossEntropyLoss()
    all_preds, all_targets, total_loss, total_batches = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.float().to(device), y.long().to(device)
            hidden = model.init_hidden(batch_size=x.shape[0]) if hasattr(model, 'hidden') else None
            logits = model(x=x, hidden=hidden)
            total_loss += criterion(logits, y).item(); total_batches += 1
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_targets.append(y.cpu())
    y_pred = torch.cat(all_preds).numpy().reshape(-1)
    y_true = torch.cat(all_targets).numpy().reshape(-1)
    return {
        'checkpoint': ckpt_path,
        'model_type': model_type,
        'loss': total_loss / max(total_batches, 1),
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))),
        'report': classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, digits=4, zero_division=0),
    }


def plot_metric_bars(results, save_path):
    names = [r['model_type'] for r in results]
    acc = [r['accuracy'] for r in results]; macro = [r['macro_f1'] for r in results]; weighted = [r['weighted_f1'] for r in results]
    x = np.arange(len(names)); width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, acc, width, label='Accuracy'); ax.bar(x, macro, width, label='Macro-F1'); ax.bar(x + width, weighted, width, label='Weighted-F1')
    ax.set_ylim(0.0, 1.05); ax.set_ylabel('分数'); ax.set_title('旱情预测模型对比'); ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right'); ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close(fig)


def plot_confusion_matrices(results, save_path):
    n = len(results); fig, axes = plt.subplots(1, n, figsize=(5.2 * n + 1.2, 5.2), constrained_layout=True)
    if n == 1: axes = [axes]
    im = None
    for ax, result in zip(axes, results):
        cm = normalize_rows(result['confusion_matrix']); im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(result['model_type']); ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别')
        ax.set_xticks(np.arange(len(CLASS_NAMES))); ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right'); ax.set_yticklabels(CLASS_NAMES)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', color='white' if cm[i, j] > 0.5 else 'black', fontsize=9)
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.92, pad=0.02); cbar.set_label('按行归一化比例')
    fig.suptitle('旱情预测混淆矩阵（行归一化）', fontsize=14); plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)


def save_text_results(results, output_dir):
    with open(os.path.join(output_dir, 'forecast_model_summary.txt'), 'w', encoding='utf-8') as f:
        for i, r in enumerate(results, 1):
            f.write('=' * 80 + f'\nRank {i}: {r["model_type"]}\nCheckpoint: {r["checkpoint"]}\nLoss: {r["loss"]:.6f}\nAccuracy: {r["accuracy"]:.6f}\nMacro-F1: {r["macro_f1"]:.6f}\nWeighted-F1: {r["weighted_f1"]:.6f}\n\n{r["report"]}\n')


def main():
    parser = argparse.ArgumentParser(description='统一评估多个旱情预测模型并生成可视化结果')
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--test_year', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dirs', nargs='+', default=['/root/autodl-tmp/zyk_drought_monitor/data_V2', '/root/autodl-tmp/data_proc', '/root/autodl-tmp/data_proc/data_proc', '/content/drive/MyDrive/GEE_Drought_Project/data_proc', '/content/drive/MyDrive/drought_monitor/data_proc'])
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/zyk_drought_monitor/results/forecast_compare_V2/V2_1')      # 新文件保存位置记录
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True); setup_chinese_font(FONT_PATH); device = torch.device(args.device)
    x_path, y_path = resolve_test_paths(args.data_dirs, args.label_mode, args.test_year)
    # x_test, y_test = build_forecast_dataset(torch.load(x_path, map_location='cpu'), torch.load(y_path, map_location='cpu'))
    x_test = torch.load(x_path, map_location='cpu')
    y_test = torch.load(y_path, map_location='cpu')
    
    actual_channels = x_test.shape[2]
    for model_key in model_params:
        if 'encoder_params' in model_params[model_key]['core']:
            model_params[model_key]['core']['encoder_params']['input_dim'] = actual_channels
        if 'attention_params' in model_params[model_key]['core']:
            model_params[model_key]['core']['input_attn_params']['input_dim'] = actual_channels
    print(f'已自动将模型输入通道数(input_dim)自适应调整为: {actual_channels}')
    
    print(f'测试集形状: X={x_test.shape}, Y={y_test.shape}')

    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    results = [evaluate_checkpoint(ckpt, test_loader, device) for ckpt in args.checkpoints if os.path.exists(ckpt)]
    if not results: raise RuntimeError('没有成功评估的预测 checkpoint。')
    results = sorted(results, key=lambda x: x['macro_f1'], reverse=True)
    save_text_results(results, args.output_dir)
    plot_metric_bars(results, os.path.join(args.output_dir, 'forecast_metrics_comparison.png'))
    plot_confusion_matrices(results, os.path.join(args.output_dir, 'forecast_confusion_matrices_normalized.png'))
    print('[OK] 预测任务评估与可视化已保存。')


if __name__ == '__main__':
    main()
