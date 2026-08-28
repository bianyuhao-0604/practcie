"""
run_eval.py — 独立评估脚本

加载已训练好的 checkpoint，在 test 集上做完整评估 + 绘图。

用法：
  python run_eval.py --ckpt checkpoints/best_gcn_bfs.pt
  python run_eval.py --ckpt checkpoints/best_gat_bfs.pt --plot
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset import get_dataloaders
from models import PPINetwork
from focal_loss import MultiLabelFocalLoss, compute_class_weights
from evaluate import multi_label_metrics, print_class_report
from train import evaluate, plot_training_curves

def load_model_from_ckpt(ckpt_path: str, device: str = DEVICE):
    """从 checkpoint 恢复模型"""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    # 重建模型
    model = PPINetwork(
        in_dim=cfg.get("in_dim", ENC_INPUT_DIM),
        enc_hidden=cfg.get("enc_hidden", ENC_HIDDEN),
        hidden_dim=cfg.get("hidden_dim", HIDDEN_DIM),
        num_layers=cfg.get("num_layers", NUM_LAYERS),
        num_classes=cfg.get("num_classes", NUM_CLASSES),
        gnn_type=cfg.get("gnn_type", "gcn"),
        dropout=cfg.get("dropout", DROPOUT),
        use_bn=not cfg.get("no_bn", False),
        residual=not cfg.get("no_residual", False),
        edge_mlp_hidden=cfg.get("edge_mlp_hidden", EDGE_MLP_HIDDEN),
        predictor_type=cfg.get("predictor", "full"),
        gat_heads=cfg.get("gat_heads", 4),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"📋 加载模型 ← {ckpt_path}")
    print(f"   训练 epoch: {ckpt.get('epoch', '?')}  "
          f"Val F1μ: {ckpt.get('val_f1_micro', '?'):.4f}")
    return model, ckpt


@torch.no_grad()
def run_test_eval(model, data, loaders, device=DEVICE):
    """在 test 集上完整评估"""
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    test_loss, test_probs, test_labels = evaluate(
        model, data, loaders["test_loader"], criterion, device
    )
    print(f"\nTest Loss: {test_loss:.4f}")
    metrics = multi_label_metrics(test_labels, test_probs, threshold=0.5)
    print_class_report(test_labels, test_probs, CLASS_NAMES)
    return metrics, test_probs, test_labels


def plot_test_predictions(test_labels, test_probs, class_names, out_dir=PLOT_DIR):
    """绘制测试集预测分布图"""
    try:
      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
      from matplotlib import rcParams
      rcParams["font.family"] = "Microsoft YaHei"   # 改用 Windows 自带中文字体
      rcParams["axes.unicode_minus"] = False        # 解决负号显示问题
    except ImportError:
      print("⚠️ matplotlib 不可用")
      return

    os.makedirs(out_dir, exist_ok=True)

    # 1. 每类 ROC-AUC 柱状图
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(test_labels.shape[1]):
        if len(np.unique(test_labels[:, c])) < 2:
            aucs.append(0.0)
        else:
            aucs.append(roc_auc_score(test_labels[:, c], test_probs[:, c]))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#55A868", "#8172B2", "#CCB974", "#C44E52",
              "#64B5CD", "#937860"]
    bars = ax.barh(class_names, aucs, color=colors[:len(class_names)])
    ax.set_xlabel("ROC-AUC")
    ax.set_title("SHS148k 各类别 AUC")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    for bar, v in zip(bars, aucs):
        ax.text(max(v + 0.01, 0.52), bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, "test_auc_per_class.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 各类别 AUC 图 → {p}")

    # 2. 预测概率分布（正样本 vs 负样本）
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()
    for c in range(min(test_labels.shape[1], 7)):
        ax = axes[c]
        pos_probs = test_probs[test_labels[:, c] == 1, c]
        neg_probs = test_probs[test_labels[:, c] == 0, c]
        ax.hist(pos_probs, bins=30, alpha=0.6, label="Positive", color="steelblue")
        ax.hist(neg_probs, bins=30, alpha=0.6, label="Negative", color="coral")
        ax.set_title(class_names[c])
        ax.legend(fontsize=8)
        ax.set_xlabel("Predicted Probability")
    axes[-1].axis("off")
    plt.suptitle("预测概率分布（正 vs 负）", fontsize=13)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "test_prob_dist.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 概率分布图 → {p2}")


def main():
    parser = argparse.ArgumentParser(description="SHS148k 模型独立评估")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoints/best_gcn_bfs.pt")
    parser.add_argument("--data_path", type=str, default=None,
                        help="预处理 .pt 路径（默认自动推断）")
    parser.add_argument("--plot", action="store_true", help="绘制额外分析图")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # 推断数据路径
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    split_mode = cfg.get("split_mode", SPLIT_MODE)
    if args.data_path is None:
        args.data_path = os.path.join(PROC_DIR, f"SHS148k_{split_mode}.pt")

    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        sys.exit(1)

    # 加载
    loaders = get_dataloaders(args.data_path)
    model, _ = load_model_from_ckpt(args.ckpt)

    # 评估
    metrics, probs, labels = run_test_eval(model, loaders["data"], loaders)

    # 可选绘图
    if args.plot:
        plot_test_predictions(labels, probs, CLASS_NAMES)


if __name__ == "__main__":
    main()
