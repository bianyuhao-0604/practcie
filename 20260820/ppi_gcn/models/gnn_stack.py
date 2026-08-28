"""
models/gnn_stack.py — GNN 消息传递层堆叠

支持 GCN 和 GAT 两种卷积类型，每层带：
  - 残差连接（Residual Connection）
  - BatchNorm
  - ReLU
  - Dropout

GCNConv  : 标准图卷积（Kipf & Welling, 2017）
GATConv  : 图注意力卷积（Veličković et al., 2018）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GNNLayer(nn.Module):
    """
    单个 GNN 层 = Conv + BN + ReLU + Dropout + 残差
    """

    def __init__(self, in_dim: int, out_dim: int,
                 gnn_type: str = "gcn",
                 dropout: float = 0.2,
                 use_bn: bool = True,
                 residual: bool = True,
                 heads: int = 4):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.residual = residual and (in_dim == out_dim)
        self.use_bn = use_bn

        if self.gnn_type == "gcn":
            self.conv = GCNConv(in_dim, out_dim, add_self_loops=True)
        elif self.gnn_type == "gat":
            # GAT 输出维度 = out_dim，多头拼接后除以 heads
            assert out_dim % heads == 0, "GAT out_dim 必须能被 heads 整除"
            self.conv = GATConv(in_dim, out_dim // heads, heads=heads,
                                concat=True, add_self_loops=True)
        else:
            raise ValueError(f"不支持的 GNN 类型: {gnn_type}")

        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        if self.residual:
            h = h + x
        return h


class GNNStack(nn.Module):
    """
    多层 GNN 堆叠。

    第一层：in_dim → hidden_dim（可能无残差，因为维度变化）
    中间层：hidden_dim → hidden_dim（带残差）
    最后一层：hidden_dim → hidden_dim（带残差）

    支持 GCN / GAT 切换。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3,
                 gnn_type: str = "gcn",
                 dropout: float = 0.2,
                 use_bn: bool = True,
                 residual: bool = True,
                 heads: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()

        layers = []
        for i in range(num_layers):
            if i == 0:
                # 第一层：in_dim → hidden_dim，不一定能残差
                layer_in = in_dim
                layer_residual = residual and (in_dim == hidden_dim)
            else:
                layer_in = hidden_dim
                layer_residual = residual

            layers.append(GNNLayer(
                in_dim=layer_in,
                out_dim=hidden_dim,
                gnn_type=gnn_type,
                dropout=dropout,
                use_bn=use_bn,
                residual=layer_residual,
                heads=heads,
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


if __name__ == "__main__":
    torch.manual_seed(0)
    N, F_in = 20, 22
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0],
    ], dtype=torch.long)

    x = torch.randn(N, F_in)
    hidden = 64

    print("── GCNStack ──")
    gnn_gcn = GNNStack(in_dim=F_in, hidden_dim=hidden, num_layers=3,
                       gnn_type="gcn", dropout=0.2, residual=True)
    out_gcn = gnn_gcn(x, edge_index)
    print(f"  Input : {x.shape}  Output: {out_gcn.shape}")
    assert out_gcn.shape == (N, hidden)
    n_gcn = sum(p.numel() for p in gnn_gcn.parameters())
    print(f"  参数量: {n_gcn:,}")

    print("\n── GATStack ──")
    gnn_gat = GNNStack(in_dim=F_in, hidden_dim=hidden, num_layers=3,
                       gnn_type="gat", dropout=0.2, residual=True, heads=4)
    out_gat = gnn_gat(x, edge_index)
    print(f"  Input : {x.shape}  Output: {out_gat.shape}")
    assert out_gat.shape == (N, hidden)
    n_gat = sum(p.numel() for p in gnn_gat.parameters())
    print(f"  参数量: {n_gat:,}")

    print("\n✅ GNNStack 自检通过 (GCN + GAT)")
