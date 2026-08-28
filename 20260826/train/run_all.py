"""
一键运行完整训练流程：
  1. GNNpre/GIN 预训练
  2. L3-PPI 分类头微调（P → G 两阶段）
  3. 测试集评测

用法：
  python train/run_all.py --dataset SHS27k --split_strategy BFS
  python train/run_all.py --dataset SHS148k --split_strategy DFS
"""

import os
import sys
import subprocess
import argparse

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TRAIN_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="L3-PPI 一键训练")
    parser.add_argument("--dataset", type=str, default="SHS27k",
                        choices=["SHS27k", "SHS148k"])
    parser.add_argument("--split_strategy", type=str, default="BFS",
                        choices=["BFS", "DFS"])
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--d_gpt", type=int, default=64)
    parser.add_argument("--d_gin", type=int, default=64)
    parser.add_argument("--gpt_layers", type=int, default=2)
    parser.add_argument("--gin_layers", type=int, default=2)
    parser.add_argument("--phase1_epochs", type=int, default=20)
    parser.add_argument("--phase2_epochs", type=int, default=80)
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="跳过 GNNpre 预训练（使用已有权重）")
    return parser.parse_args()


def run_command(cmd, desc):
    """运行子命令"""
    print(f"\n{'#' * 70}")
    print(f"# {desc}")
    print(f"# 命令: {' '.join(cmd)}")
    print(f"{'#' * 70}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[错误] {desc} 执行失败")
        sys.exit(1)
    print(f"\n[完成] {desc}")


def main():
    args = parse_args()

    benchmark_dir = os.path.join(
        PROJECT_DIR, "data", "benchmark",
        f"{args.dataset}_{args.split_strategy}"
    )
    embeddings_path = os.path.join(
        PROJECT_DIR, "data", "embeddings",
        f"esm2_650m_{args.dataset}.pt"
    )
    gnn_pre_dir = os.path.join(PROJECT_DIR, "checkpoints", "gnn_pre")
    gnn_pre_ckpt = os.path.join(gnn_pre_dir, "best_gnn_pre.pt")
    l3ppi_dir = os.path.join(
        PROJECT_DIR, "checkpoints", "l3ppi",
        f"{args.dataset}_{args.split_strategy}"
    )

    python = sys.executable

    # ---- 步骤 1：GNNpre 预训练 ----
    if not args.skip_pretrain:
        cmd = [
            python, os.path.join(TRAIN_DIR, "train_gnn_pre.py"),
            "--pretrain_dir", os.path.join(PROJECT_DIR, "data", "pretrain"),
            "--embeddings_path", embeddings_path,
            "--output_dir", gnn_pre_dir,
            "--d_gin", str(args.d_gin),
            "--num_layers", str(args.gin_layers),
            "--epochs", str(args.pretrain_epochs),
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
        ]
        run_command(cmd, "步骤 1/3：GNNpre/GIN 预训练")
    else:
        print(f"\n[跳过] GNNpre 预训练（使用已有权重: {gnn_pre_ckpt}）")

    # ---- 步骤 2：L3-PPI 分类头微调 ----
    cmd = [
        python, os.path.join(TRAIN_DIR, "train_l3ppi.py"),
        "--benchmark_dir", benchmark_dir,
        "--embeddings_path", embeddings_path,
        "--gnn_pre_checkpoint", gnn_pre_ckpt,
        "--output_dir", l3ppi_dir,
        "--K", str(args.K),
        "--d_gpt", str(args.d_gpt),
        "--d_gin", str(args.d_gin),
        "--gpt_layers", str(args.gpt_layers),
        "--gin_layers", str(args.gin_layers),
        "--phase1_epochs", str(args.phase1_epochs),
        "--phase2_epochs", str(args.phase2_epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    run_command(cmd, "步骤 2/3：L3-PPI 分类头微调（P → G 两阶段）")

    # ---- 步骤 3：测试集评测 ----
    best_ckpt = os.path.join(l3ppi_dir, "best_l3ppi.pt")
    cmd = [
        python, os.path.join(TRAIN_DIR, "evaluate.py"),
        "--checkpoint", best_ckpt,
        "--benchmark_dir", benchmark_dir,
        "--embeddings_path", embeddings_path,
        "--split", "test",
        "--K", str(args.K),
        "--d_gpt", str(args.d_gpt),
        "--d_gin", str(args.d_gin),
    ]
    run_command(cmd, "步骤 3/3：测试集评测")

    print(f"\n{'=' * 70}")
    print(f"全部完成！")
    print(f"  数据集: {args.dataset} ({args.split_strategy})")
    print(f"  GNNpre 权重: {gnn_pre_ckpt}")
    print(f"  L3-PPI 权重: {best_ckpt}")
    print(f"  评测结果: {l3ppi_dir}/results.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()