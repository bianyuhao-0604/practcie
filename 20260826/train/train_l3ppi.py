"""
L3-PPI 分类头微调脚本（P → G 两阶段训练策略）。

阶段 1（Phase 1）：仅训练提示嵌入
  - 冻结 GNNgpt 门控网络
  - 仅优化提示嵌入和投影层
  - 所有 K 条路径保持激活

阶段 2（Phase 2）：启用门控网络，联合优化
  - 解冻 GNNgpt 门控网络
  - 联合优化提示嵌入 + GNNgpt
  - Gumbel-Softmax 采样选择性激活路径

用法：
  python train/train_l3ppi.py \
      --benchmark_dir data/benchmark/SHS27k_BFS \
      --embeddings_path data/embeddings/esm2_650m_SHS27k.pt \
      --gnn_pre_checkpoint checkpoints/gnn_pre/best_gnn_pre.pt \
      --output_dir checkpoints/l3ppi/SHS27k_BFS \
      --K 4 \
      --phase1_epochs 20 \
      --phase2_epochs 80 \
      --lr 1e-3
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.l3ppi import L3PPIClassificationHead
from datasets.benchmark_dataset import PPIBenchmarkDataset
from utils.metrics import compute_metrics, format_metrics
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="L3-PPI 分类头微调")

    # 数据路径
    parser.add_argument("--benchmark_dir", type=str,
                        default="data/benchmark/SHS27k_BFS",
                        help="评测基准目录（包含 train.pt 和 test.pt）")
    parser.add_argument("--embeddings_path", type=str,
                        default="data/embeddings/esm2_650m_SHS27k.pt",
                        help="ESM2 嵌入文件路径")
    parser.add_argument("--gnn_pre_checkpoint", type=str,
                        default="checkpoints/gnn_pre/best_gnn_pre.pt",
                        help="预训练 GNNpre 权重路径")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/l3ppi/SHS27k_BFS",
                        help="模型保存目录")

    # 模型超参数
    parser.add_argument("--d_model", type=int, default=1280,
                        help="ESM2 嵌入维度")
    parser.add_argument("--d_prompt", type=int, default=64,
                        help="提示图内部维度")
    parser.add_argument("--d_gpt", type=int, default=64,
                        help="GNNgpt 隐藏维度")
    parser.add_argument("--d_gin", type=int, default=64,
                        help="GNNpre/GIN 隐藏维度")
    parser.add_argument("--gpt_layers", type=int, default=2,
                        help="GNNgpt 层数")
    parser.add_argument("--gin_layers", type=int, default=2,
                        help="GNNpre/GIN 层数")
    parser.add_argument("--K", type=int, default=4,
                        help="候选 L3 路径数")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Gumbel-Softmax 温度")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="LPN 正则化超参数")

    # 训练超参数
    parser.add_argument("--phase1_epochs", type=int, default=20,
                        help="Phase 1 训练轮数（仅训练提示嵌入）")
    parser.add_argument("--phase2_epochs", type=int, default=80,
                        help="Phase 2 训练轮数（联合优化）")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批大小")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--patience", type=int, default=15,
                        help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    return parser.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_phase1(model, train_loader, val_loader, optimizer, device,
                 epochs, logger):
    """
    Phase 1：仅训练提示嵌入。
    冻结 GNNgpt，仅优化提示嵌入和投影层。
    """
    logger.info("=" * 60)
    logger.info("Phase 1：训练提示嵌入（门控网络冻结）")
    logger.info("=" * 60)

    # 冻结 GNNgpt
    for param in model.gnn_gpt.parameters():
        param.requires_grad = False

    # 冻结 GNNpre（已预训练）
    model.freeze_gnn_pre()

    # 仅优化提示嵌入和投影层
    trainable_params = list(model.prompt_embeddings.parameters())
    optimizer.param_groups[0]['params'] = trainable_params

    logger.info(f"  可训练参数: {sum(p.numel() for p in trainable_params):,}")
    logger.info(f"  训练轮数: {epochs}")

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        total_loss = 0
        all_labels, all_scores = [], []

        for embed_u, embed_v, labels in train_loader:
            embed_u = embed_u.to(device)
            embed_v = embed_v.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            y_pre, gate_values, path_probs = model(
                embed_u, embed_v, training=True
            )
            loss, loss_bce, loss_lpn = model.compute_loss(
                y_pre, gate_values, path_probs, labels
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * embed_u.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(y_pre.detach().cpu().numpy().flatten().tolist())

        train_loss = total_loss / len(train_loader.dataset)
        train_metrics = compute_metrics(
            np.array(all_labels), np.array(all_scores)
        )

        # 验证
        val_loss, val_metrics = evaluate(model, val_loader, device)

        logger.info(
            f"  P1 Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        logger.info(f"    Train: {format_metrics(train_metrics)}")
        logger.info(f"    Val:   {format_metrics(val_metrics)}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"    早停：Val F1 连续 {args.patience} 轮未提升")
                break

    logger.info(f"  Phase 1 完成，最佳 Val F1: {best_val_f1:.4f}")
    return best_val_f1


def train_phase2(model, train_loader, val_loader, optimizer, device,
                 epochs, logger, output_dir):
    """
    Phase 2：启用门控网络，联合优化提示嵌入 + GNNgpt。
    """
    logger.info("=" * 60)
    logger.info("Phase 2：联合优化提示嵌入 + 门控网络")
    logger.info("=" * 60)

    # 解冻 GNNgpt
    for param in model.gnn_gpt.parameters():
        param.requires_grad = True

    # GNNpre 保持冻结
    model.freeze_gnn_pre()

    # 优化提示嵌入 + GNNgpt
    trainable_params = (
        list(model.prompt_embeddings.parameters())
        + list(model.gnn_gpt.parameters())
    )
    optimizer.param_groups[0]['params'] = trainable_params

    logger.info(f"  可训练参数: {sum(p.numel() for p in trainable_params):,}")
    logger.info(f"  训练轮数: {epochs}")

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        total_loss = 0
        total_bce = 0
        total_lpn = 0
        all_labels, all_scores = [], []

        for embed_u, embed_v, labels in train_loader:
            embed_u = embed_u.to(device)
            embed_v = embed_v.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            y_pre, gate_values, path_probs = model(
                embed_u, embed_v, training=True
            )
            loss, loss_bce, loss_lpn = model.compute_loss(
                y_pre, gate_values, path_probs, labels
            )

            loss.backward()
            optimizer.step()

            bs = embed_u.size(0)
            total_loss += loss.item() * bs
            total_bce += loss_bce.item() * bs
            total_lpn += loss_lpn.item() * bs
            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(y_pre.detach().cpu().numpy().flatten().tolist())

        train_loss = total_loss / len(train_loader.dataset)
        avg_bce = total_bce / len(train_loader.dataset)
        avg_lpn = total_lpn / len(train_loader.dataset)
        train_metrics = compute_metrics(
            np.array(all_labels), np.array(all_scores)
        )

        # 验证
        val_loss, val_metrics = evaluate(model, val_loader, device)

        logger.info(
            f"  P2 Epoch {epoch:3d}/{epochs} | "
            f"Loss: {train_loss:.4f} (BCE: {avg_bce:.4f}, LPN: {avg_lpn:.4f}) | "
            f"Val Loss: {val_loss:.4f}"
        )
        logger.info(f"    Train: {format_metrics(train_metrics)}")
        logger.info(f"    Val:   {format_metrics(val_metrics)}")

        # 门控统计
        with torch.no_grad():
            sample_embed_u, sample_embed_v, _ = next(iter(train_loader))
            sample_embed_u = sample_embed_u[:16].to(device)
            sample_embed_v = sample_embed_v[:16].to(device)
            _, gv, _ = model(sample_embed_u, sample_embed_v, training=False)
            avg_active = gv.sum(dim=1).mean().item()
            logger.info(f"    平均激活路径数: {avg_active:.2f} / {model.K}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            best_path = os.path.join(output_dir, "best_l3ppi.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"    ✓ 保存最佳模型 (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"    早停：Val F1 连续 {args.patience} 轮未提升")
                break

    logger.info(f"  Phase 2 完成，最佳 Val F1: {best_val_f1:.4f}")
    return best_val_f1


@torch.no_grad()
def evaluate(model, loader, device):
    """评估"""
    model.eval()
    total_loss = 0
    all_labels, all_scores = [], []

    for embed_u, embed_v, labels in loader:
        embed_u = embed_u.to(device)
        embed_v = embed_v.to(device)
        labels = labels.to(device)

        y_pre, gate_values, path_probs = model(
            embed_u, embed_v, training=False
        )
        loss, _, _ = model.compute_loss(y_pre, gate_values, path_probs, labels)

        total_loss += loss.item() * embed_u.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_scores.extend(y_pre.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))

    return avg_loss, metrics


@torch.no_grad()
def test(model, test_loader, device, logger):
    """测试集评估"""
    model.eval()
    all_labels, all_scores = [], []
    all_active_paths = []

    for embed_u, embed_v, labels in test_loader:
        embed_u = embed_u.to(device)
        embed_v = embed_v.to(device)

        predictions, scores, active_paths = model.predict(embed_u, embed_v)

        all_labels.extend(labels.numpy().tolist())
        all_scores.extend(scores.cpu().numpy().flatten().tolist())
        all_active_paths.extend(active_paths)

    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))

    logger.info("=" * 60)
    logger.info("测试集评估结果")
    logger.info("=" * 60)
    logger.info(format_metrics(metrics))
    logger.info(f"混淆矩阵: TP={metrics['tp']}, FP={metrics['fp']}, "
                f"TN={metrics['tn']}, FN={metrics['fn']}")

    # 门控统计
    avg_paths = np.mean([len(p) for p in all_active_paths])
    logger.info(f"平均激活路径数: {avg_paths:.2f}")

    return metrics


def main():
    global args
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir, name="l3ppi_train")
    logger.info(f"配置: {vars(args)}")
    logger.info(f"设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 加载数据集 ----
    logger.info("加载评测基准数据集...")
    train_dataset = PPIBenchmarkDataset(
        data_path=os.path.join(args.benchmark_dir, "train.pt"),
        embeddings_path=args.embeddings_path,
        d_model=args.d_model,
    )
    test_dataset = PPIBenchmarkDataset(
        data_path=os.path.join(args.benchmark_dir, "test.pt"),
        embeddings_path=args.embeddings_path,
        d_model=args.d_model,
    )

    # 从训练集中划分验证集
    val_size = int(len(train_dataset) * 0.1)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
    )

    logger.info(f"训练集: {train_size}, 验证集: {val_size}, "
                f"测试集: {len(test_dataset)}")

    # ---- 创建模型 ----
    logger.info("创建 L3-PPI 分类头...")
    model = L3PPIClassificationHead(
        d_model=args.d_model,
        d_prompt=args.d_prompt,
        d_gpt=args.d_gpt,
        d_gin=args.d_gin,
        gpt_layers=args.gpt_layers,
        gin_layers=args.gin_layers,
        K=args.K,
        temperature=args.temperature,
        gamma=args.gamma,
    ).to(device)

    # 加载预训练的 GNNpre/GIN
    if os.path.exists(args.gnn_pre_checkpoint):
        model.load_pretrained_gnn_pre(args.gnn_pre_checkpoint)
    else:
        logger.warning(
            f"未找到预训练 GNNpre 权重: {args.gnn_pre_checkpoint}"
        )
        logger.warning("GNNpre 将随机初始化，建议先运行 train_gnn_pre.py")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    # ---- 优化器 ----
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- Phase 1：仅训练提示嵌入 ----
    best_p1_f1 = train_phase1(
        model, train_loader, val_loader, optimizer, device,
        epochs=args.phase1_epochs, logger=logger,
    )

    # ---- Phase 2：联合优化 ----
    best_p2_f1 = train_phase2(
        model, train_loader, val_loader, optimizer, device,
        epochs=args.phase2_epochs, logger=logger,
        output_dir=args.output_dir,
    )

    # ---- 测试集评估 ----
    # 加载最佳模型
    best_model_path = os.path.join(args.output_dir, "best_l3ppi.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"已加载最佳模型: {best_model_path}")

    test_metrics = test(model, test_loader, device, logger)

    # 保存测试指标
    import json
    results = {
        "config": vars(args),
        "phase1_best_val_f1": best_p1_f1,
        "phase2_best_val_f1": best_p2_f1,
        "test_metrics": test_metrics,
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存: {results_path}")


if __name__ == "__main__":
    main()