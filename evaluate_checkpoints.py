import os
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from configs.config import model_params
from models.baseline.convlstm import ConvLSTM


def build_convlstm(device: torch.device, use_attention: bool) -> ConvLSTM:
    cfg = model_params["convlstm"]["core"]
    input_attn_params = cfg["input_attn_params"] if use_attention else None

    model = ConvLSTM(
        input_size=cfg["input_size"],
        window_in=cfg["window_in"],
        num_layers=cfg["num_layers"],
        encoder_params=cfg["encoder_params"],
        input_attn_params=input_attn_params,
        device=device,
    )
    return model.to(device)


def read_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    raise ValueError(f"无法识别 checkpoint 格式: {ckpt_path}")


def infer_attention_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("input_attn.") for k in state_dict.keys())


def load_model_for_checkpoint(ckpt_path: str, device: torch.device) -> ConvLSTM:
    state_dict = read_state_dict(ckpt_path)
    use_attention = infer_attention_from_state_dict(state_dict)

    model = build_convlstm(device=device, use_attention=use_attention)

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[OK] 严格加载成功: {os.path.basename(ckpt_path)} | attention={use_attention}")
    except RuntimeError as e:
        print(f"[WARN] 严格加载失败，改为 non-strict: {os.path.basename(ckpt_path)}")
        print(f"       原因: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"       missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")

    model.eval()
    return model


def evaluate_single_checkpoint(
    ckpt_path: str,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
) -> Dict[str, float]:
    model = load_model_for_checkpoint(ckpt_path, device)
    criterion = nn.CrossEntropyLoss()

    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(device)
            y = y.long().to(device)

            hidden = model.init_hidden(batch_size=x.shape[0])
            logits = model(x=x, hidden=hidden)  # [B, C, H, W]

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
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    class_names = ["无旱 (0)", "轻旱 (1)", "中旱 (2)", "重/特旱 (3)"]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    print("\n" + "=" * 80)
    print(f"模型: {ckpt_path}")
    print(f"Loss={avg_loss:.6f} | Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f} | Weighted-F1={weighted_f1:.4f}")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return {
        "checkpoint": ckpt_path,
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="评估多个 ConvLSTM checkpoint（支持有/无 attention）")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/zyk_drought_monitor/data")
    parser.add_argument("--x_test", type=str, default="dataset_X_2024.pt")
    parser.add_argument("--y_test", type=str, default="dataset_Y_2024.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "/root/autodl-tmp/zyk_drought_monitor/data/drought_convlstm_best.pth",
            "/root/autodl-tmp/zyk_drought_monitor/data/drought_convlstm_best_2021_2024_413.pth",
            "/root/autodl-tmp/zyk_drought_monitor/data/drought_convlstm_best_2021_2024_20260414_211714.pth",
        ],
    )
    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp/zyk_drought_monitor/results/model_compare")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    x_path = os.path.join(args.data_dir, args.x_test)
    y_path = os.path.join(args.data_dir, args.y_test)

    print(f"加载测试集: {x_path}")
    X_test = torch.load(x_path, map_location="cpu")
    print(f"加载测试标签: {y_path}")
    Y_test = torch.load(y_path, map_location="cpu")

    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_results = []
    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            print(f"[SKIP] 文件不存在: {ckpt}")
            continue

        result = evaluate_single_checkpoint(
            ckpt_path=ckpt,
            test_loader=test_loader,
            device=device,
            num_classes=4,
        )
        all_results.append(result)

        base_name = os.path.splitext(os.path.basename(ckpt))[0]
        report_path = os.path.join(args.save_dir, f"{base_name}_classification_report.txt")
        cm_path = os.path.join(args.save_dir, f"{base_name}_confusion_matrix.pt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result["report"])
            f.write("\n")
            f.write(f"Loss: {result['loss']:.6f}\n")
            f.write(f"Accuracy: {result['accuracy']:.6f}\n")
            f.write(f"Macro-F1: {result['macro_f1']:.6f}\n")
            f.write(f"Weighted-F1: {result['weighted_f1']:.6f}\n")

        torch.save(torch.tensor(result["confusion_matrix"]), cm_path)
        print(f"已保存: {report_path}")
        print(f"已保存: {cm_path}")

    if not all_results:
        print("没有可评估的 checkpoint。")
        return

    all_results = sorted(all_results, key=lambda x: x["macro_f1"], reverse=True)

    print("\n" + "#" * 80)
    print("模型对比（按 Macro-F1 降序）")
    for i, r in enumerate(all_results, 1):
        print(
            f"{i}. {os.path.basename(r['checkpoint'])} | "
            f"Loss={r['loss']:.6f}, Acc={r['accuracy']:.4f}, "
            f"Macro-F1={r['macro_f1']:.4f}, Weighted-F1={r['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
