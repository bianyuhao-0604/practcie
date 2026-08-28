"""
评估指标计算。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def compute_metrics(labels: np.ndarray, scores: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """
    计算二分类评估指标。

    Args:
        labels: 真实标签 [N]，值为 0 或 1
        scores: 预测分数 [N]，值为 (0, 1) 之间的概率
        threshold: 分类阈值

    Returns:
        metrics: 包含所有指标的字典
    """
    preds = (scores >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }

    # AUC 需要至少两个类别的样本
    if len(np.unique(labels)) > 1:
        metrics['auc'] = roc_auc_score(labels, scores)
        metrics['aupr'] = average_precision_score(labels, scores)
    else:
        metrics['auc'] = 0.0
        metrics['aupr'] = 0.0

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics['tp'] = int(tp)
    metrics['fp'] = int(fp)
    metrics['tn'] = int(tn)
    metrics['fn'] = int(fn)

    return metrics


def format_metrics(metrics: dict) -> str:
    """格式化指标为字符串"""
    return (
        f"Acc={metrics['accuracy']:.4f} | "
        f"Prec={metrics['precision']:.4f} | "
        f"Rec={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"AUC={metrics['auc']:.4f} | "
        f"AUPR={metrics['aupr']:.4f}"
    )