# nt_xent.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent: Normalized Temperature-scaled Cross Entropy (SimCLR, eq.1)
    z1, z2: (N, d) —— 同一批图像两个视图的投影输出
    分母含 2N-1 个候选：1 个正样本 + 2N-2 个批内负样本。
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.t = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # ℓ2 归一化（必须！）

        sim = z @ z.T / self.t                               # (2N, 2N) 余弦相似度 / τ
        sim.fill_diagonal_(float("-inf"))                    # 等价于指示函数 1[k≠i]

        # 行 i∈[0,N) 的正样本是 i+N；行 N+i 的正样本是 i（cat 顺序决定）
        targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)

        return F.cross_entropy(sim, targets)                 # -log softmax = NT-Xent


@torch.no_grad()
def contrastive_accuracy(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """训练监控指标：从 2N-1 个候选中认出正对的 top-1 准确率。"""
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.T
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return (sim.argmax(1) == targets).float().mean().item()
