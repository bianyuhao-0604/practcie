"""
ablation.py — 层数消融实验 (独立脚本)

用法:
    python ablation.py
    python ablation.py --gnn_type gat --split_mode bfs --loss focal
    (额外参数会原样透传给 train.py 的 main())

产出:
    ablation_results.json   各层数的 Val / Test Micro-F1
    layer_ablation.png      层数消融折线图
"""
import json
import sys

import numpy as np

# train.py 已暴露 main(argv) 与 --num_layers 参数, 直接导入
from train import main as train_main


LAYER_CANDIDATES = [2, 3, 4, 5, 6]


def run_ablation(extra_args):
    """遍历候选层数, 逐个训练并收集最佳 Val F1 与 Test F1"""
    results = []
    for n_layers in LAYER_CANDIDATES:
        print()
        print("=" * 60)
        print(f"  消融实验: num_layers = {n_layers}")
        print("=" * 60)

        # --num_layers 放在末尾, 保证覆盖用户可能误传的同名参数
        argv = list(extra_args) + ["--num_layers", str(n_layers)]
        best_val_f1, test_metrics = train_main(argv)

        results.append({
            "num_layers": n_layers,
            "val_f1": float(best_val_f1),
            "test_f1": float(test_metrics["f1_micro"]),
            "test_auc": float(test_metrics.get("auc_macro", 0.0)),
        })
        print(f"  [结果] num_layers={n_layers}  "
              f"Val F1={best_val_f1:.4f}  "
              f"Test F1={test_metrics['f1_micro']:.4f}")

    with open("ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print()
    print("消融结果已保存到 ablation_results.json")
    return results


def plot_ablation(results, save_path="layer_ablation.png"):
    """绘制 层数 vs Val/Test Micro-F1 折线图, 并标注最优点"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers  = [r["num_layers"] for r in results]
    val_f1  = [r["val_f1"] for r in results]
    test_f1 = [r["test_f1"] for r in results]

    best_idx = int(np.argmax(val_f1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, val_f1, "o-", color="#2196F3", linewidth=2,
            markersize=8, label="Best Val Micro-F1")
    ax.plot(layers, test_f1, "s--", color="#FF9800", linewidth=2,
            markersize=8, label="Test Micro-F1")



    for x, y in zip(layers, val_f1):
        ax.text(x, y + 0.006, f"{y:.4f}", ha="center", va="bottom", fontsize=9)
    for x, y in zip(layers, test_f1):
        ax.text(x, y - 0.014, f"{y:.4f}", ha="center", va="top", fontsize=9)

    ax.set_xlabel("Number of GNN Layers", fontsize=13)
    ax.set_ylabel("Micro-F1", fontsize=13)
    ax.set_title("Layer Ablation Study on SHS148K", fontsize=15, fontweight="bold")
    ax.set_xticks(layers)
    all_vals = val_f1 + test_f1
    ax.set_ylim(min(all_vals) - 0.05, max(all_vals) + 0.06)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"折线图已保存到 {save_path}")


if __name__ == "__main__":
    # 命令行额外参数原样透传给 train.py, 例如:
    # python ablation.py --gnn_type gat --split_mode bfs
    extra = sys.argv[1:]
    results = run_ablation(extra)
    plot_ablation(results)

    # 打印汇总表
    print()
    print("=" * 44)
    print(f"{'num_layers':>10} | {'Val F1':>10} | {'Test F1':>10}")
    print("-" * 44)
    for r in results:
        print(f"{r['num_layers']:>10} | {r['val_f1']:>10.4f} | {r['test_f1']:>10.4f}")
    print("=" * 44)
