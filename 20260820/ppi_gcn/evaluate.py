"""
evaluate.py — 多标签分类评估指标

提供：
  - multi_label_metrics: 计算综合指标
  - print_class_report : 逐类打印 precision/recall/f1
  - hamming_accuracy   : 逐位准确率
"""

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix,
)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def hamming_accuracy(y_true, y_pred):
    """逐位准确率（所有标签位的平均正确率）"""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return np.mean(y_true == y_pred)


def subset_accuracy(y_true, y_pred):
    """完全匹配率：预测集合与真实集合完全一致的比例"""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return np.mean(np.all(y_true == y_pred, axis=1))


def multi_label_metrics(y_true, y_prob, threshold=0.5,
                         average="macro", verbose=True):
    """
    综合多标签评估。

    y_true : [N, C]  0/1 真实标签
    y_prob : [N, C]  预测概率（sigmoid 后）
    threshold : 二值化阈值

    返回 dict
    """
    y_true = _to_numpy(y_true)
    y_prob = _to_numpy(y_prob)
    y_pred = (y_prob >= threshold).astype(np.float32)

    results = {}

    # ── F1 ──
    for avg in ["micro", "macro", "weighted"]:
        results[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)

    # ── Precision / Recall ──
    results["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    results["recall_macro"]    = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # ── Hamming / Subset ──
    results["hamming_acc"]  = hamming_accuracy(y_true, y_pred)
    results["subset_acc"]   = subset_accuracy(y_true, y_pred)

    # ── AUC (per-class, 再平均) ──
    aucs = []
    aps  = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            aucs.append(np.nan)
            aps.append(np.nan)
        else:
            aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
            aps.append(average_precision_score(y_true[:, c], y_prob[:, c]))
    results["auc_macro"] = np.nanmean(aucs)
    results["ap_macro"]  = np.nanmean(aps)

    # ── MCC（展平）──
    try:
        results["mcc"] = matthews_corrcoef(y_true.ravel(), y_pred.ravel())
    except Exception:
        results["mcc"] = 0.0

    if verbose:
        print("\n" + "="*55)
        print("  📊 多标签评估结果")
        print("="*55)
        print(f"  Micro F1    : {results['f1_micro']:.4f}")
        print(f"  Macro F1    : {results['f1_macro']:.4f}")
        print(f"  Weighted F1 : {results['f1_weighted']:.4f}")
        print(f"  Precision(M): {results['precision_macro']:.4f}")
        print(f"  Recall(M)   : {results['recall_macro']:.4f}")
        print(f"  Hamming Acc : {results['hamming_acc']:.4f}")
        print(f"  Subset Acc  : {results['subset_acc']:.4f}")
        print(f"  AUC(Macro)  : {results['auc_macro']:.4f}")
        print(f"  AP(Macro)   : {results['ap_macro']:.4f}")
        print(f"  MCC         : {results['mcc']:.4f}")
        print("="*55)

    return results


def print_class_report(y_true, y_prob, class_names=None, threshold=0.5):
    """逐类打印 Precision / Recall / F1"""
    y_true = _to_numpy(y_true)
    y_prob = _to_numpy(y_prob)
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)

    C = y_true.shape[1]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(C)]

    print(f"\n{'类别':<14s} {'支持数':>6s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
    print("-" * 52)
    for i in range(C):
        support = int(y_true[:, i].sum())
        if support == 0:
            print(f"{class_names[i]:<14s} {support:>6d} {'N/A':>10s} {'N/A':>8s} {'N/A':>8s}")
            continue
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        print(f"{class_names[i]:<14s} {support:>6d} {p:>10.4f} {r:>8.4f} {f:>8.4f}")
    print("-" * 52)


if __name__ == "__main__":
    # 快速自检
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, (200, 7)).astype(np.float32)
    y_prob = rng.rand(200, 7).astype(np.float32)

    results = multi_label_metrics(y_true, y_prob, verbose=True)
    print_class_report(y_true, y_prob,
                       class_names=["activation","binding","catalysis",
                                    "expression","inhibition","ptmod","reaction"])
    print("\n✅ evaluate.py 自检通过")
