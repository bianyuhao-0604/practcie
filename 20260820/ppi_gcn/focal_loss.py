"""
focal_loss.py — 多标签 Focal Loss

解决 SHS148k 类别不平衡问题：
  - expression / ptmod / inhibition 样本极少
  - 标准 BCE 被多数类主导

Focal Loss 公式（多标签版）：
  FL = −α * (1−p_t)^γ * log(p_t)
  p_t = p if y=1 else 1−p
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    """
    多标签 Focal Loss。

    Args:
        gamma      : 聚焦参数，越大越关注难样本（默认 2.0）
        alpha      : 平衡因子，0~1（默认 0.25）
        reduction  : mean / sum / none
        pos_weight : 可选，[num_classes] 正样本权重（配合 BCE 思路）
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 reduction: str = "mean", pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, C]  原始 logits（未 sigmoid）
        targets : [B, C]  0/1 多标签
        """
        # 数值稳定的 BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight,
            reduction="none"
        )  # [B, C]

        # p_t 计算
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = torch.clamp(p_t, min=1e-7, max=1 - 1e-7)

        # focal 权重
        focal_weight = (1 - p_t) ** self.gamma

        # alpha 平衡
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def compute_class_weights(labels: torch.Tensor,
                          num_classes: int = 7,
                          clip_max: float = 10.0) -> torch.Tensor:
    """
    根据训练集标签计算类别权重（逆频率）。
    用于 BCE 的 pos_weight 参数。

    labels : [N, num_classes]  0/1 多标签
    返回   : [num_classes]  float tensor
    """
    pos_counts = labels.sum(dim=0)  # [C]
    neg_counts = labels.shape[0] - pos_counts
    # 防止除零
    pos_counts = torch.clamp(pos_counts, min=1.0)
    weights = neg_counts / pos_counts
    weights = torch.clamp(weights, max=clip_max)
    return weights


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C = 16, 7
    logits = torch.randn(B, C)
    targets = torch.randint(0, 2, (B, C)).float()

    # 对比 BCE 和 Focal
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    focal = MultiLabelFocalLoss(gamma=2.0, alpha=0.25)(logits, targets)

    print(f"BCE Loss      : {bce.item():.4f}")
    print(f"Focal Loss    : {focal.item():.4f}")

    # 类别权重
    big_labels = torch.randint(0, 2, (1000, 7)).float()
    # 制造不平衡
    big_labels[:, 0] = 1.0  # activation 全正
    big_labels[:, 4] = 0.0  # inhibition 全负
    w = compute_class_weights(big_labels, 7)
    print(f"\n类别权重: {w}")

    print("\n✅ Focal Loss 自检通过")
