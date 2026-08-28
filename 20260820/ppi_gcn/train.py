"""
train.py — 训练主循环

功能：
  - 加载预处理数据
  - 初始化 PPINetwork (GCN/GAT)
  - AdamW + Cosine 调度 + 早停
  - 支持 BCE / Focal Loss
  - 定期在 val 集评估，保存最佳模型
  - 训练曲线可视化

用法：
  python train.py --gnn_type gcn --split_mode bfs --epochs 200
  python train.py --gnn_type gat --split_mode bfs --epochs 200 --loss focal
"""

import os
import sys
import time
import json
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset import get_dataloaders
from models import PPINetwork
from focal_loss import MultiLabelFocalLoss, compute_class_weights
from evaluate import multi_label_metrics, print_class_report
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# ───────────────────── 训练一个 epoch ─────────────────────
def train_epoch(model, data, loaders, optimizer, criterion, device):
    model.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    total_loss = 0.0
    total_pos = 0

    for edges, labels in loaders["train_loader"]:
        edges  = edges.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(x, edge_index, edges)
        loss = criterion(logits, labels)

        loss.backward()
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_pos  += bs

    return total_loss / max(total_pos, 1)


# ───────────────────── 在指定 split 上评估 ─────────────────────
@torch.no_grad()
def evaluate(model, data, loader, criterion, device, threshold=0.5):
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_n = 0

    for edges, labels in loader:
        edges  = edges.to(device)
        labels = labels.to(device)

        logits = model(x, edge_index, edges)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(all_logits).numpy()
    loss_avg = total_loss / max(total_n, 1)

    return loss_avg, probs, all_labels.numpy()


# ───────────────────── 主函数 ─────────────────────
def main(argv=None):
    print(">>> 进入 main()") 
    parser = argparse.ArgumentParser(description="SHS148k GCN/GAT 训练")
    parser.add_argument("--data_path", type=str, default=None,
                        help="预处理 .pt 文件路径")
    parser.add_argument("--gnn_type", type=str, default="gcn",
                        choices=["gcn", "gat"])
    parser.add_argument("--split_mode", type=str, default=SPLIT_MODE,
                        choices=["random", "bfs", "dfs"])
    parser.add_argument("--loss", type=str, default=LOSS_TYPE,
                        choices=["bce", "focal"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--scheduler", type=str, default=SCHEDULER,
                        choices=["cosine", "step", "none"])
    parser.add_argument("--predictor", type=str, default="full",
                        choices=["full", "simple"])
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--no_bn", action="store_true")
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS,
                        help="GNN层数 (默认取 config.NUM_LAYERS)")
    args = parser.parse_args(argv)

    # ── 设置随机种子 ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 数据路径 ──
    if args.data_path is None:
        args.data_path = os.path.join(PROC_DIR, f"SHS148k_{args.split_mode}.pt")
    if not os.path.exists(args.data_path):
        print(f"❌ 预处理文件不存在: {args.data_path}")
        print(f"   请先运行: python data_process.py --split_mode {args.split_mode}")
        sys.exit(1)

    # ── 加载数据 ──
    print(f"\n{'='*60}")
    print(f"  🧬 SHS148k PPI 预测 — {args.gnn_type.upper()} 训练")
    print(f"  数据: {args.data_path}")
    print(f"  划分: {args.split_mode.upper()}  Loss: {args.loss}")
    print(f"{'='*60}\n")
    print(f">>> 开始加载数据: {args.data_path}")
    loaders = get_dataloaders(args.data_path, batch_size=args.batch_size)
    print(">>> 数据加载完成")
    data = loaders["data"].to(DEVICE)
    x = data.x

    num_nodes = x.size(0)
    in_dim = x.size(1)
    print(f"[Data] 节点数={num_nodes}  特征维度={in_dim}  "
          f"GNN边数={data.edge_index.size(1)}  "
          f"预测边数={data.edge_pairs.size(1)}")
# 在构建 loss 之前
    for split_name, loader in [("train", loaders["train_loader"]),
                           ("val", loaders["val_loader"]),
                           ("test", loaders["test_loader"])]:
        all_labels = []
        for edges, labels in loader:
            all_labels.append(labels.cpu().numpy())
        all_labels = np.concatenate(all_labels, axis=0)
        print(f"\n[{split_name}] 标签形状: {all_labels.shape}")
        print(f"  每类正样本数: {all_labels.sum(axis=0).astype(int)}")
        print(f"  总正样本数: {all_labels.sum()}")
    # ── 构建模型 ──
    model = PPINetwork(
        in_dim=in_dim,
        enc_hidden=ENC_HIDDEN,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        gnn_type=args.gnn_type,
        dropout=DROPOUT,
        use_bn=not args.no_bn,
        residual=not args.no_residual,
        edge_mlp_hidden=EDGE_MLP_HIDDEN,
        predictor_type=args.predictor,
        gat_heads=args.gat_heads,
    ).to(DEVICE)

    n_params = model.count_parameters()
    print(f"[Model] 参数量: {n_params:,}")
    print(f"[Model] 设备: {DEVICE}")

    # ── 损失函数 ──
    if args.loss == "focal":
        criterion = MultiLabelFocalLoss(
            gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, reduction="mean"
        )
        print(f"[Loss] MultiLabelFocalLoss (γ={FOCAL_GAMMA}, α={FOCAL_ALPHA})")
    else:
        # 计算类别权重
        train_idx = data.train_idx.cpu().numpy()
        train_labels = data.edge_y[train_idx].cpu().numpy()
        train_labels_t = torch.from_numpy(train_labels)
        pos_weight = compute_class_weights(train_labels_t, NUM_CLASSES, clip_max=10.0)
        pos_weight = pos_weight.to(DEVICE)
        print(f"[Loss] BCE with pos_weight={pos_weight.cpu().numpy().round(2)}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    # ── 优化器 & 调度器 ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        scheduler = None

    # ── 训练记录 ──
    history = {
        "train_loss": [], "val_loss": [], "val_f1_micro": [],
        "val_f1_macro": [], "val_auc": [], "lr": [],
    }
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    # ── 训练循环 ──
    print(f"\n🚀 开始训练 ({args.epochs} epochs) ...\n")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        tr_loss = train_epoch(model, data, loaders, optimizer, criterion, DEVICE)

        # Val
        val_loss, val_probs, val_labels = evaluate(
            model, data, loaders["val_loader"], criterion, DEVICE
        )
        val_metrics = multi_label_metrics(val_labels, val_probs,
                                           threshold=0.5, verbose=False)

        # 记录
        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(round(tr_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_f1_micro"].append(round(val_metrics["f1_micro"], 4))
        history["val_f1_macro"].append(round(val_metrics["f1_macro"], 4))
        history["val_auc"].append(round(val_metrics["auc_macro"], 4))
        history["lr"].append(round(lr_now, 6))

        elapsed = time.time() - t0

        # 日志
        log = (f"Epoch {epoch:4d}/{args.epochs} | "
               f"LR {lr_now:.2e} | "
               f"TrainLoss {tr_loss:.4f} | "
               f"ValLoss {val_loss:.4f} | "
               f"F1μ {val_metrics['f1_micro']:.4f} | "
               f"F1M {val_metrics['f1_macro']:.4f} | "
               f"AUC {val_metrics['auc_macro']:.4f} | "
               f"{elapsed:.1f}s")
        print(log)

        # 调度器
        if scheduler is not None:
            scheduler.step()

        # 早停 & 保存最佳
        if val_metrics["f1_micro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_micro"]
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(CKPT_DIR,
                f"best_{args.gnn_type}_{args.split_mode}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": vars(args),
                "epoch": epoch,
                "val_f1_micro": best_val_f1,
                "history": history,
            }, ckpt_path)
            print(f"  💾 保存最佳模型 → {ckpt_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n⏹️  早停触发 (耐心={args.patience}, "
                  f"最佳 epoch={best_epoch}, F1μ={best_val_f1:.4f})")
            break

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ✅ 训练完成! 总耗时: {total_time/60:.1f} min")
    print(f"  最佳 Val F1μ: {best_val_f1:.4f} @ epoch {best_epoch}")
    print(f"{'='*60}")

    # ── 加载最佳模型，在 Test 集上最终评估 ──
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n📋 加载最佳模型 (epoch {ckpt['epoch']})")

    test_loss, test_probs, test_labels = evaluate(
        model, data, loaders["test_loader"], criterion, DEVICE
    )
    print(f"\n{'='*60}")
    print(f"  🎯 Test 集最终评估")
    print(f"{'='*60}")
    test_metrics = multi_label_metrics(test_labels, test_probs, threshold=0.5)
    print_class_report(test_labels, test_probs, CLASS_NAMES)

    # ── 保存训练历史 ──
    history_path = os.path.join(LOG_DIR,
        f"history_{args.gnn_type}_{args.split_mode}.json")
    with open(history_path, "w") as f:
        json.dump({
            "config": vars(args),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_f1_micro": best_val_f1,
            "test_metrics": test_metrics,
        }, f, indent=2)
    print(f"\n💾 训练历史 → {history_path}")

    # ── 绘制训练曲线 ──
    plot_training_curves(history, args, test_metrics)
    return best_val_f1, test_metrics

# ───────────────────── 绘图 ─────────────────────
def plot_training_curves(history: dict, args, test_metrics: dict):
    """绘制 Loss 和 F1 曲线"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    except ImportError:
        print("⚠️ matplotlib 不可用，跳过绘图")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss 曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1
    ax = axes[1]
    ax.plot(epochs, history["val_f1_micro"], "g-", label="Val F1 Micro")
    ax.plot(epochs, history["val_f1_macro"], "m-", label="Val F1 Macro")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("F1 曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[2]
    ax.plot(epochs, history["val_auc"], "c-", label="Val AUC Macro")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("AUC 曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"SHS148k {args.gnn_type.upper()} ({args.split_mode.upper()}) "
                 f"— Test F1μ={test_metrics['f1_micro']:.4f}", fontsize=13)
    plt.tight_layout()

    plot_path = os.path.join(PLOT_DIR,
        f"curves_{args.gnn_type}_{args.split_mode}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 训练曲线 → {plot_path}")


if __name__ == "__main__":
    main()