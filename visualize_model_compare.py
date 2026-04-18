import os
import re
import glob
import argparse
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_chinese_font(font_path: str):
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_prop = font_manager.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"[OK] Loaded Chinese font: {font_name}")
    else:
        print(f"[WARN] Font file not found: {font_path}. Using default font.")


def parse_classification_report(report_path: str) -> Dict[str, float]:
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    metrics = {
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
    }

    acc_match = re.search(r"accuracy\s+([0-9]*\.?[0-9]+)", text)
    if acc_match:
        metrics["accuracy"] = float(acc_match.group(1))

    macro_match = re.search(r"macro avg\s+[0-9]*\.?[0-9]+\s+[0-9]*\.?[0-9]+\s+([0-9]*\.?[0-9]+)", text)
    if macro_match:
        metrics["macro_f1"] = float(macro_match.group(1))

    weighted_match = re.search(r"weighted avg\s+[0-9]*\.?[0-9]+\s+[0-9]*\.?[0-9]+\s+([0-9]*\.?[0-9]+)", text)
    if weighted_match:
        metrics["weighted_f1"] = float(weighted_match.group(1))

    return metrics


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def plot_metric_bars(model_names: List[str], metrics: Dict[str, List[float]], save_path: str):
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, metrics["accuracy"], width, label="Accuracy")
    ax.bar(x, metrics["macro_f1"], width, label="Macro-F1")
    ax.bar(x + width, metrics["weighted_f1"], width, label="Weighted-F1")

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on 2024 Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i in range(len(model_names)):
        for offset, key in [(-width, "accuracy"), (0, "macro_f1"), (width, "weighted_f1")]:
            val = metrics[key][i]
            if not np.isnan(val):
                ax.text(x[i] + offset, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(cms: List[np.ndarray], model_names: List[str], class_names: List[str], save_path: str):
    n = len(cms)

    # 使用 constrained_layout 自动处理子图与色标，避免右侧子图被 colorbar 挤压/遮挡
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5.2 * n + 1.2, 5.2),
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    im = None
    for ax, cm, model_name in zip(axes, cms, model_names):
        cm_norm = normalize_rows(cm)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_yticklabels(class_names)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm_norm[i, j]
                txt_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=9)

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.92, pad=0.02)
    cbar.set_label("Row-normalized ratio")
    fig.suptitle("Confusion Matrices (Row-normalized)", fontsize=14)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize model_compare evaluation outputs")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/autodl-tmp/zyk_drought_monitor/results/model_compare",
        help="Directory containing *_classification_report.txt and *_confusion_matrix.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/zyk_drought_monitor/results/model_compare/figures",
        help="Directory to save visualization PNG files",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="/root/autodl-tmp/zyk_drought_monitor/SimHei.ttf",
        help="Path to a Chinese font file (e.g., SimHei.ttf)",
    )
    args = parser.parse_args()

    setup_chinese_font(args.font_path)

    os.makedirs(args.output_dir, exist_ok=True)

    report_files = sorted(glob.glob(os.path.join(args.input_dir, "*_classification_report.txt")))
    if not report_files:
        raise FileNotFoundError(f"No *_classification_report.txt found in {args.input_dir}")

    model_names = []
    cms = []
    metric_values = {"accuracy": [], "macro_f1": [], "weighted_f1": []}

    for rep_path in report_files:
        base = os.path.basename(rep_path).replace("_classification_report.txt", "")
        cm_path = os.path.join(args.input_dir, f"{base}_confusion_matrix.pt")

        if not os.path.exists(cm_path):
            print(f"[SKIP] Missing confusion matrix for {base}")
            continue

        cm = torch.load(cm_path, map_location="cpu")
        if isinstance(cm, torch.Tensor):
            cm = cm.numpy()
        cm = np.asarray(cm)

        metrics = parse_classification_report(rep_path)

        model_names.append(base)
        cms.append(cm)
        metric_values["accuracy"].append(metrics["accuracy"])
        metric_values["macro_f1"].append(metrics["macro_f1"])
        metric_values["weighted_f1"].append(metrics["weighted_f1"])

    if not model_names:
        raise RuntimeError("No complete report+confusion_matrix pairs found.")

    bar_path = os.path.join(args.output_dir, "metrics_comparison.png")
    cm_path = os.path.join(args.output_dir, "confusion_matrices_normalized.png")

    plot_metric_bars(model_names, metric_values, bar_path)
    plot_confusion_matrices(
        cms,
        model_names,
        class_names=["无旱", "轻旱", "中旱", "重/特旱"],
        save_path=cm_path,
    )

    print(f"[OK] Saved metrics bar chart: {bar_path}")
    print(f"[OK] Saved confusion matrix chart: {cm_path}")


if __name__ == "__main__":
    main()
