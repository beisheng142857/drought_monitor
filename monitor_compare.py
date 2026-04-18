import argparse
import os
import sys
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


def resolve_test_paths(data_dirs: List[str], label_mode: str, test_year: int) -> Tuple[str, str]:
    x_path = find_existing_file(data_dirs, [f'dataset_X_{test_year}.pt'])
    if label_mode == 'threshold':
        y_candidates = [f'dataset_Y_{test_year}_threshold.pt', 'dataset_Y_threshold.pt']
    else:
        y_candidates = [f'dataset_Y_{test_year}.pt', 'dataset_Y.pt']
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
    raise ValueError(f'无法从文件名推断模型类型，请显式重命名或修改脚本: {ckpt_path}')


def build_model(model_type: str, device: torch.device):
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


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    model_type = infer_model_type_from_ckpt(ckpt_path)
    model = build_model(model_type, device)
    state_dict = load_state_dict(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, model_type


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def evaluate_checkpoint(ckpt_path: str, test_loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model, model_type = load_model_from_checkpoint(ckpt_path, device)
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(device)
            y = y.long().to(device)
            hidden = model.init_hidden(batch_size=x.shape[0]) if hasattr(model, 'hidden') else None
            logits = model(x=x, hidden=hidden)
            loss = criterion(logits, y)

            total_loss += loss.item()
            total_batches += 1
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy().reshape(-1)
    y_true = torch.cat(all_targets, dim=0).numpy().reshape(-1)
    avg_loss = total_loss / max(total_batches, 1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    return {
        'checkpoint': ckpt_path,
        'model_type': model_type,
        'loss': avg_loss,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'report': report,
    }


def plot_metric_bars(results: List[Dict[str, object]], save_path: str):
    model_names = [r['model_type'] for r in results]
    accuracy = [r['accuracy'] for r in results]
    macro_f1 = [r['macro_f1'] for r in results]
    weighted_f1 = [r['weighted_f1'] for r in results]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, accuracy, width, label='Accuracy')
    ax.bar(x, macro_f1, width, label='Macro-F1')
    ax.bar(x + width, weighted_f1, width, label='Weighted-F1')

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Drought Monitoring Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for i in range(len(model_names)):
        for offset, values in [(-width, accuracy), (0, macro_f1), (width, weighted_f1)]:
            val = values[i]
            ax.text(x[i] + offset, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(results: List[Dict[str, object]], save_path: str):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n + 1.2, 5.2), constrained_layout=True)
    if n == 1:
        axes = [axes]

    im = None
    for ax, result in zip(axes, results):
        cm = normalize_rows(result['confusion_matrix'])
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(result['model_type'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(np.arange(len(CLASS_NAMES)))
        ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
        ax.set_yticklabels(CLASS_NAMES)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                txt_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=txt_color, fontsize=9)

    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.92, pad=0.02)
    cbar.set_label('Row-normalized ratio')
    fig.suptitle('Confusion Matrices (Row-normalized)', fontsize=14)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_text_results(results: List[Dict[str, object]], output_dir: str):
    summary_path = os.path.join(output_dir, 'model_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        for rank, result in enumerate(results, 1):
            f.write('=' * 80 + '\n')
            f.write(f"Rank {rank}: {result['model_type']}\n")
            f.write(f"Checkpoint: {result['checkpoint']}\n")
            f.write(f"Loss: {result['loss']:.6f}\n")
            f.write(f"Accuracy: {result['accuracy']:.6f}\n")
            f.write(f"Macro-F1: {result['macro_f1']:.6f}\n")
            f.write(f"Weighted-F1: {result['weighted_f1']:.6f}\n\n")
            f.write(result['report'])
            f.write('\n')

        f.write('=' * 80 + '\n')
        f.write('排序依据: Macro-F1 (降序)\n')


def main():
    parser = argparse.ArgumentParser(description='统一评估多个监测模型并生成可视化结果')
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--test_year', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=16)
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
        '--output_dir',
        type=str,
        default='/root/autodl-tmp/zyk_drought_monitor/results/monitor_compare',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_chinese_font(FONT_PATH)
    device = torch.device(args.device)

    x_path, y_path = resolve_test_paths(args.data_dirs, args.label_mode, args.test_year)
    print(f'加载测试特征: {x_path}')
    print(f'加载测试标签: {y_path}')
    x_test = torch.load(x_path, map_location='cpu')
    y_test = torch.load(y_path, map_location='cpu')

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    results = []
    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            print(f'[SKIP] 文件不存在: {ckpt}')
            continue
        result = evaluate_checkpoint(ckpt, test_loader, device)
        results.append(result)
        print(
            f"[OK] {result['model_type']} | Loss={result['loss']:.6f}, "
            f"Acc={result['accuracy']:.4f}, Macro-F1={result['macro_f1']:.4f}, "
            f"Weighted-F1={result['weighted_f1']:.4f}"
        )

    if not results:
        raise RuntimeError('没有成功评估的 checkpoint。')

    results = sorted(results, key=lambda x: x['macro_f1'], reverse=True)
    save_text_results(results, args.output_dir)
    plot_metric_bars(results, os.path.join(args.output_dir, 'metrics_comparison.png'))
    plot_confusion_matrices(results, os.path.join(args.output_dir, 'confusion_matrices_normalized.png'))

    print(f"[OK] 已保存汇总文本: {os.path.join(args.output_dir, 'model_summary.txt')}")
    print(f"[OK] 已保存指标图: {os.path.join(args.output_dir, 'metrics_comparison.png')}")
    print(f"[OK] 已保存混淆矩阵图: {os.path.join(args.output_dir, 'confusion_matrices_normalized.png')}")


if __name__ == '__main__':
    main()
