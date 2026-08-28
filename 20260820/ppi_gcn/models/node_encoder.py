"""
models/node_encoder.py — 节点特征编码器

将每个蛋白质节点的原始特征（22 维氨基酸组成向量）
映射到 GNN 隐藏空间（默认 128 维）。

架构：Linear → BN → ReLU → Dropout → Linear → BN → ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    """
    两层 MLP 节点编码器。

    Input  : [N, in_dim]    — 原始节点特征（如氨基酸组成）
    Output : [N, out_dim]   — GNN 输入节点嵌入
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.bn2(h)
        h = self.relu(h)
        return h


if __name__ == "__main__":
    # 离线自检
    torch.manual_seed(0)
    N, in_d = 16, 22
    x = torch.randn(N, in_d)
    enc = NodeEncoder(in_dim=22, hidden_dim=64, out_dim=128, dropout=0.1)
    out = enc(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    assert out.shape == (N, 128), "输出维度错误"
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"参数量 : {n_params:,}")
    print("✅ NodeEncoder 自检通过")
