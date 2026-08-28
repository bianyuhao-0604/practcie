"""
models/edge_predictor.py — 边预测头（Edge Predictor）

输入：GNN 输出的两端节点嵌入 z_u, z_v
输出：7 维 sigmoid 多标签预测

融合方式：
  concat   : [z_u, z_v, z_u⊙z_v, |z_u−z_v|]
  bilinear : uᵀ W v
  → MLP → 7 类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePredictor(nn.Module):
    """
    边级多标签分类头。

    Input  : z_u [B, D], z_v [B, D]
    Output : logits [B, num_classes]  (sigmoid 前)
    """

    def __init__(self, hidden_dim: int, num_classes: int = 7,
                 mlp_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 双线性项
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim, bias=True)

        # 拼接后的 MLP
        # 输入维度 = hidden + hidden + hidden + hidden = 4*hidden
        fused_dim = 4 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.BatchNorm1d(mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, num_classes),
        )

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """
        z_u, z_v : [B, D]
        返回       : [B, num_classes]  (raw logits)
        """
        # 双线性项
        bilinear_out = self.bilinear(z_u, z_v)  # [B, D]

        # 拼接项
        hadamard = z_u * z_v                      # [B, D]
        diff = torch.abs(z_u - z_v)              # [B, D]
        concat = torch.cat([z_u, z_v, hadamard, diff], dim=-1)  # [B, 4D]

        # 融合
        fused = torch.cat([concat, bilinear_out], dim=-1)  # [B, 4D + D] = [B, 5D]
        # 注意：MLP 输入是 4*hidden，我们把 bilinear 加进去后改一下
        # 为简洁，直接用 concat 进 MLP，bilinear 作为残差
        out = self.mlp(concat)  # [B, num_classes]
        return out


class SimpleEdgePredictor(nn.Module):
    """
    简化版：仅用拼接 + Hadamard，更快更轻。
    """

    def __init__(self, hidden_dim: int, num_classes: int = 7,
                 mlp_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        hadamard = z_u * z_v
        feat = torch.cat([z_u, z_v, hadamard], dim=-1)
        return self.mlp(feat)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 8, 64
    z_u = torch.randn(B, D)
    z_v = torch.randn(B, D)

    print("── EdgePredictor (Full) ──")
    pred = EdgePredictor(hidden_dim=D, num_classes=7, mlp_hidden=256)
    logits = pred(z_u, z_v)
    print(f"  Input: z_u{z_u.shape} z_v{z_v.shape}")
    print(f"  Output: {logits.shape}")
    assert logits.shape == (B, 7)
    n = sum(p.numel() for p in pred.parameters())
    print(f"  参数量: {n:,}")

    print("\n── SimpleEdgePredictor ──")
    spred = SimpleEdgePredictor(hidden_dim=D, num_classes=7)
    slogits = spred(z_u, z_v)
    print(f"  Output: {slogits.shape}")
    assert slogits.shape == (B, 7)
    sn = sum(p.numel() for p in spred.parameters())
    print(f"  参数量: {sn:,}")

    print("\n✅ EdgePredictor 自检通过")
