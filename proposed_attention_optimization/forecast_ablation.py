import argparse, copy, csv, os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

NEW_CODE_DIR = '/root/autodl-tmp/zyk_drought_monitor/proposed_attention_optimization'
ROOT_DIR = '/root/autodl-tmp/zyk_drought_monitor'
if NEW_CODE_DIR not in sys.path:
    sys.path.insert(0, NEW_CODE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.chdir(NEW_CODE_DIR)

from configs.config import model_params
from models.baseline.convlstm import ConvLSTM

CLASS_NAMES = ['无旱', '轻旱', '中旱', '重/特旱']
FONT_PATH = '/root/autodl-tmp/zyk_drought_monitor/SimHei.ttf'
FORECAST_INPUT_STEPS = 4


def setup_font():
    if os.path.exists(FONT_PATH):
        font_manager.fontManager.addfont(FONT_PATH)
        name = font_manager.FontProperties(fname=FONT_PATH).get_name()
        plt.rcParams['font.sans-serif'] = [name]
        plt.rcParams['axes.unicode_minus'] = False


def find_existing_file(candidate_dirs, candidate_names):
    for directory in candidate_dirs:
        for name in candidate_names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f'未找到候选文件: {candidate_names}')


def resolve_test_paths(data_dirs, label_mode, test_year):
    x_path = find_existing_file(data_dirs, [f'forecast_v2_X_{test_year}.pt'])
    y_names = [f'forecast_v2_Y_{test_year}.pt', 'forecast_v2_Y.pt'] if label_mode == 'threshold' else [f'forecast_v2_Y_{test_year}.pt', 'forecast_v2_Y.pt']
    y_path = find_existing_file(data_dirs, y_names)
    return x_path, y_path


def build_model(model_type, device, actual_channels):
    cfg = copy.deepcopy(model_params['convlstm']['core'])
    cfg['window_in'] = FORECAST_INPUT_STEPS
    cfg['encoder_params']['input_dim'] = actual_channels
    attn = copy.deepcopy(cfg['input_attn_params']) if model_type == 'convlstm_attn' else None
    if attn is not None:
        attn['input_dim'] = actual_channels
    return ConvLSTM(cfg['input_size'], cfg['window_in'], cfg['num_layers'], cfg['encoder_params'], attn, device).to(device)


def load_model(ckpt_path, model_type, device, actual_channels):
    model = build_model(model_type, device, actual_channels)
    obj = torch.load(ckpt_path, map_location='cpu')
    state = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    all_preds, all_targets, total_loss, total_batches = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.long().to(device)
            hidden = model.init_hidden(batch_size=x.shape[0]) if hasattr(model, 'hidden') else None
            logits = model(x=x, hidden=hidden)
            total_loss += criterion(logits, y).item()
            total_batches += 1
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_targets.append(y.cpu())
    y_pred = torch.cat(all_preds).numpy().reshape(-1)
    y_true = torch.cat(all_targets).numpy().reshape(-1)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    return {
        'loss': total_loss / max(total_batches, 1),
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cm': confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))),
        'report': classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, digits=4, zero_division=0),
        'p': p, 'r': r, 'f1': f1, 's': s,
    }


def norm_rows(cm):
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return cm / rs


def plot_metrics(base_res, ours_res, save_path):
    names = ['Accuracy', 'Macro-F1', 'Weighted-F1']
    base_vals = [base_res['accuracy'], base_res['macro_f1'], base_res['weighted_f1']]
    ours_vals = [ours_res['accuracy'], ours_res['macro_f1'], ours_res['weighted_f1']]
    x = np.arange(len(names)); width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    b1 = ax.bar(x - width/2, base_vals, width, label='Base (no attn)', color='#8da0cb')
    b2 = ax.bar(x + width/2, ours_vals, width, label='Ours (with attn)', color='#fc8d62')
    ax.set_ylim(0, 1.05); ax.set_ylabel('分数'); ax.set_title('注意力机制消融实验：总体指标对比')
    ax.set_xticks(x); ax.set_xticklabels(names); ax.grid(axis='y', linestyle='--', alpha=0.3); ax.legend()
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close(fig)


def plot_per_class_f1(base_res, ours_res, save_path):
    x = np.arange(len(CLASS_NAMES)); width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    b1 = ax.bar(x - width/2, base_res['f1'], width, label='Base', color='#8da0cb')
    b2 = ax.bar(x + width/2, ours_res['f1'], width, label='Ours', color='#fc8d62')
    ax.set_ylim(0, 1.05); ax.set_ylabel('F1-score'); ax.set_title('注意力机制消融实验：各类别 F1 对比')
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES); ax.grid(axis='y', linestyle='--', alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close(fig)


def plot_cm(base_res, ours_res, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, cm, title in zip(axes, [base_res['cm'], ours_res['cm']], ['Base (no attn)', 'Ours (with attn)']):
        m = norm_rows(cm)
        im = ax.imshow(m, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(title); ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别')
        ax.set_xticks(np.arange(len(CLASS_NAMES))); ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right'); ax.set_yticklabels(CLASS_NAMES)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f'{m[i,j]:.2f}', ha='center', va='center', color='white' if m[i,j] > 0.5 else 'black', fontsize=9)
    fig.colorbar(im, ax=axes, location='right', shrink=0.92, pad=0.03)
    fig.suptitle('注意力机制消融实验：混淆矩阵对比', fontsize=13)
    plt.savefig(save_path, dpi=220, bbox_inches='tight'); plt.close(fig)


def save_outputs(base_res, ours_res, output_dir, base_ckpt, ours_ckpt):
    with open(os.path.join(output_dir, 'ablation_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('================ 注意力机制消融实验摘要 ================\n')
        f.write(f'Base checkpoint: {base_ckpt}\nOurs checkpoint: {ours_ckpt}\n\n')
        f.write(f'Base -> Loss={base_res["loss"]:.6f}, Accuracy={base_res["accuracy"]:.6f}, Macro-F1={base_res["macro_f1"]:.6f}, Weighted-F1={base_res["weighted_f1"]:.6f}\n')
        f.write(f'Ours -> Loss={ours_res["loss"]:.6f}, Accuracy={ours_res["accuracy"]:.6f}, Macro-F1={ours_res["macro_f1"]:.6f}, Weighted-F1={ours_res["weighted_f1"]:.6f}\n')
        f.write(f'Delta -> Loss={ours_res["loss"] - base_res["loss"]:+.6f}, Accuracy={ours_res["accuracy"] - base_res["accuracy"]:+.6f}, Macro-F1={ours_res["macro_f1"] - base_res["macro_f1"]:+.6f}, Weighted-F1={ours_res["weighted_f1"] - base_res["weighted_f1"]:+.6f}\n\n')
        f.write('[Base report]\n' + base_res['report'] + '\n\n[Ours report]\n' + ours_res['report'] + '\n')
    with open(os.path.join(output_dir, 'ablation_summary.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model', 'loss', 'accuracy', 'macro_f1', 'weighted_f1'])
        w.writerow(['convlstm_no_attn', base_res['loss'], base_res['accuracy'], base_res['macro_f1'], base_res['weighted_f1']])
        w.writerow(['convlstm_attn', ours_res['loss'], ours_res['accuracy'], ours_res['macro_f1'], ours_res['weighted_f1']])
        w.writerow(['delta(ours-base)', ours_res['loss'] - base_res['loss'], ours_res['accuracy'] - base_res['accuracy'], ours_res['macro_f1'] - base_res['macro_f1'], ours_res['weighted_f1'] - base_res['weighted_f1']])
        w.writerow([])
        w.writerow(['class_name', 'base_f1', 'ours_f1', 'delta_f1'])
        for i, name in enumerate(CLASS_NAMES):
            w.writerow([name, base_res['f1'][i], ours_res['f1'][i], ours_res['f1'][i] - base_res['f1'][i]])


def main():
    parser = argparse.ArgumentParser(description='旱情预测注意力机制消融实验脚本')
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--test_year', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dirs', nargs='+', default=['/root/autodl-tmp/zyk_drought_monitor/data_V2', '/root/autodl-tmp/data_proc', '/root/autodl-tmp/data_proc/data_proc'])
    parser.add_argument('--base_ckpt', type=str, required=True)
    parser.add_argument('--ours_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/zyk_drought_monitor/results/forecast_ablation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_font()
    device = torch.device(args.device)
    x_path, y_path = resolve_test_paths(args.data_dirs, args.label_mode, args.test_year)
    x_test = torch.load(x_path, map_location='cpu')
    y_test = torch.load(y_path, map_location='cpu')
    actual_channels = x_test.shape[2]
    loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

    base_model = load_model(args.base_ckpt, 'convlstm_no_attn', device, actual_channels)
    ours_model = load_model(args.ours_ckpt, 'convlstm_attn', device, actual_channels)
    base_res = evaluate(base_model, loader, device)
    ours_res = evaluate(ours_model, loader, device)

    plot_metrics(base_res, ours_res, os.path.join(args.output_dir, 'ablation_metrics_comparison.png'))
    plot_per_class_f1(base_res, ours_res, os.path.join(args.output_dir, 'ablation_per_class_f1.png'))
    plot_cm(base_res, ours_res, os.path.join(args.output_dir, 'ablation_confusion_matrices_normalized.png'))
    save_outputs(base_res, ours_res, args.output_dir, args.base_ckpt, args.ours_ckpt)

    print('[OK] 注意力机制消融实验已完成。')
    print(f"Base Macro-F1 : {base_res['macro_f1']:.6f}")
    print(f"Ours Macro-F1 : {ours_res['macro_f1']:.6f}")
    print(f"Delta Macro-F1: {ours_res['macro_f1'] - base_res['macro_f1']:+.6f}")


if __name__ == '__main__':
    main()
