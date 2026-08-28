"""
GNNpre/GIN 预训练脚本。

在 HI-II-14 提取的 L3 路径样本上训练 GIN 进行图级二分类：
  - 正样本：已知交互蛋白对之间的 L3 路径（y=1）
  - 负样本：非交互蛋白对之间的 L3 路径（y=0）

预训练完成后保存模型权重，供 L3-PPI 分类头加载并冻结。

用法：
  python train/train_gnn_pre.py \
      --pretrain_dir data/pretrain \
      --embeddings_dir data/embeddings \
      --output_dir checkpoints/gnn_pre \
      --d_gin 64 \
      --num_layers 2 \
      --epochs 50 \
      --lr 1e-3 \
      --batch_size 256
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_pre import GNNpre
from datasets.pretrain_dataset import L3PathPretrainDataset
from utils.metrics import compute_metrics, format_metrics
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="GNNpre/GIN 预训练")

    # 数据路径
    parser.add_argument("--pretrain_dir", type=str,
                        default="data/pretrain",
                        help="预训练样本目录")
    parser.add_argument("--embeddings_path", type=str,
                        default="data/embeddings/esm2_650m_pretrain.pt",
                        help="ESM2 嵌入文件路径")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/gnn_pre",
                        help="模型保存目录")

    # 模型超参数
    parser.add_argument("--d_model", type=int, default=1280,
                        help="ESM2 嵌入维度")
    parser.add_argument("--d_gin", type=int, default=64,
                        help="GIN 隐藏维度")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="GIN 层数")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="批大小")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_labels = []
    all_scores = []

    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        # 前向传播
        y_pre = model(batch_data)  # [batch, 1]
        labels = labels.unsqueeze(1).float()   # (batch,) -> (batch,1)，并转为 float

        # 计算损失
        loss = nn.functional.binary_cross_entropy(y_pre, labels)
        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_data.num_graphs
        all_labels.extend(labels.cpu().numpy().tolist())
        all_scores.extend(y_pre.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    labels_arr = np.array(all_labels).flatten()
    scores_arr = np.array(all_scores).flatten()
    metrics = compute_metrics(labels_arr, scores_arr)

    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, device):
    """评估"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []

    for batch_data in loader:
        batch_data = batch_data.to(device)

        y_pre = model(batch_data)
        labels = batch_data.y.float()

        loss = nn.functional.binary_cross_entropy(y_pre, labels)

        total_loss += loss.item() * batch_data.num_graphs
        all_labels.extend(labels.cpu().numpy().tolist())
        all_scores.extend(y_pre.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    labels_arr = np.array(all_labels).flatten()
    scores_arr = np.array(all_scores).flatten()
    metrics = compute_metrics(labels_arr, scores_arr)

    return avg_loss, metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置日志
    logger = setup_logger(args.output_dir, name="gnn_pre_train")
    logger.info(f"配置: {vars(args)}")
    logger.info(f"设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 加载数据集 ----
    logger.info("加载预训练数据集...")
    dataset = L3PathPretrainDataset(
        pretrain_dir=args.pretrain_dir,
        embeddings_path=args.embeddings_path,
        d_model=args.d_model,
    )

    # 划分训练集和验证集
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    # 创建数据加载器
    train_loader = PyGDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
    )
    val_loader = PyGDataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
    )

    # ---- 创建模型 ----
    logger.info("创建 GNNpre/GIN 模型...")
    model = GNNpre(
        d_in=args.d_model,
        d_gin=args.d_gin,
        num_layers=args.num_layers,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count:,}")

    # 确保预训练阶段所有参数可训练
    model.unfreeze()

    # ---- 优化器 ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    )

    # ---- 训练循环 ----
    logger.info("=" * 60)
    logger.info("开始 GNNpre/GIN 预训练")
    logger.info("=" * 60)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device
        )

        # 验证
        val_loss, val_metrics = evaluate(model, val_loader, device)

        # 学习率调度
        scheduler.step(val_loss)

        # 日志
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )
        logger.info(f"  Train: {format_metrics(train_metrics)}")
        logger.info(f"  Val:   {format_metrics(val_metrics)}")

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            best_model_path = os.path.join(args.output_dir, "best_gnn_pre.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  ✓ 保存最佳模型 (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"  早停：验证 F1 连续 {args.patience} 轮未提升")
                break

    # ---- 最终评估 ----
    logger.info("=" * 60)
    logger.info("预训练完成")
    logger.info(f"最佳验证 F1: {best_val_f1:.4f}")
    logger.info("=" * 60)

    # 加载最佳模型进行最终评估
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_gnn_pre.pt"))
    )
    val_loss, val_metrics = evaluate(model, val_loader, device)
    logger.info(f"最终验证指标: {format_metrics(val_metrics)}")

    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final_gnn_pre.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"最终模型已保存: {final_path}")


if __name__ == "__main__":
    main()