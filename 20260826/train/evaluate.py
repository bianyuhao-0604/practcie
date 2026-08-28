"""
独立评测脚本：加载训练好的 L3-PPI 模型，在测试集上评估。

用法：
  python train/evaluate.py \
      --checkpoint checkpoints/l3ppi/SHS27k_BFS/best_l3ppi.pt \
      --benchmark_dir data/benchmark/SHS27k_BFS \
      --embeddings_path data/embeddings/esm2_650m_SHS27k.pt
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.l3ppi import L3PPIClassificationHead
from datasets.benchmark_dataset import PPIBenchmarkDataset
from utils.metrics import compute_metrics, format_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="L3-PPI 评测")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"])
    parser.add_argument("--d_model", type=int, default=1280)
    parser.add_argument("--d_prompt", type=int, default=64)
    parser.add_argument("--d_gpt", type=int, default=64)
    parser.add_argument("--d_gin", type=int, default=64)
    parser.add_argument("--gpt_layers", type=int, default=2)
    parser.add_argument("--gin_layers", type=int, default=2)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = L3PPIClassificationHead(
        d_model=args.d_model, d_prompt=args.d_prompt,
        d_gpt=args.d_gpt, d_gin=args.d_gin,
        gpt_layers=args.gpt_layers, gin_layers=args.gin_layers,
        K=args.K,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"已加载模型: {args.checkpoint}")

    # 加载数据
    data_path = os.path.join(args.benchmark_dir, f"{args.split}.pt")
    dataset = PPIBenchmarkDataset(
        data_path=data_path,
        embeddings_path=args.embeddings_path,
        d_model=args.d_model,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 评测
    all_labels, all_scores = [], []
    all_active_paths = []

    for embed_u, embed_v, labels in loader:
        embed_u = embed_u.to(device)
        embed_v = embed_v.to(device)

        predictions, scores, active_paths = model.predict(embed_u, embed_v)

        all_labels.extend(labels.numpy().tolist())
        all_scores.extend(scores.cpu().numpy().flatten().tolist())
        all_active_paths.extend(active_paths)

    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))

    print(f"\n{'=' * 60}")
    print(f"评测结果 ({args.split} set)")
    print(f"{'=' * 60}")
    print(format_metrics(metrics))
    print(f"混淆矩阵: TP={metrics['tp']}, FP={metrics['fp']}, "
          f"TN={metrics['tn']}, FN={metrics['fn']}")

    avg_paths = np.mean([len(p) for p in all_active_paths])
    print(f"平均激活路径数: {avg_paths:.2f} / {args.K}")

    # 保存结果
    results_path = args.checkpoint.replace('.pt', f'_eval_{args.split}.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n结果已保存: {results_path}")


if __name__ == "__main__":
    main()