"""
models/__init__.py — 完整 PPI-GCN 模型组装

PPINetwork:
  NodeEncoder → GNNStack → 取出边两端节点 → EdgePredictor → 7类预测
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ENC_INPUT_DIM, ENC_HIDDEN, ENC_OUTPUT,
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, USE_BN, RESIDUAL,
    NUM_CLASSES, EDGE_MLP_HIDDEN,
)

from .node_encoder import NodeEncoder
from .gnn_stack import GNNStack
from .edge_predictor import EdgePredictor, SimpleEdgePredictor


class PPINetwork(nn.Module):
    """
    完整 PPI 预测 GNN 模型。

    流程：
      x [N, 22] ──NodeEncoder──▶ h [N, H]
                  ──GNNStack──▶ z [N, H]   (结构感知嵌入)
      对每条边 (u,v): z_u, z_v
                  ──EdgePredictor──▶ logits [B, 7]
    """

    def __init__(self, in_dim: int = ENC_INPUT_DIM,
                 enc_hidden: int = ENC_HIDDEN,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 num_classes: int = NUM_CLASSES,
                 gnn_type: str = "gcn",
                 dropout: float = DROPOUT,
                 use_bn: bool = USE_BN,
                 residual: bool = RESIDUAL,
                 edge_mlp_hidden: int = EDGE_MLP_HIDDEN,
                 predictor_type: str = "full",  # full / simple
                 gat_heads: int = 4):
        super().__init__()

        # 1. 节点编码器
        self.node_encoder = NodeEncoder(
            in_dim=in_dim,
            hidden_dim=enc_hidden,
            out_dim=hidden_dim,
            dropout=dropout * 0.5,  # 编码器处 dropout 小一些
            use_bn=use_bn,
        )

        # 2. GNN 堆叠
        self.gnn = GNNStack(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            use_bn=use_bn,
            residual=residual,
            heads=gat_heads,
        )

        # 3. 边预测头
        if predictor_type == "full":
            self.edge_predictor = EdgePredictor(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                mlp_hidden=edge_mlp_hidden,
                dropout=dropout,
            )
        else:
            self.edge_predictor = SimpleEdgePredictor(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                mlp_hidden=edge_mlp_hidden // 2,
                dropout=dropout,
            )

        # 4. 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """编码所有节点 → [N, H]"""
        h = self.node_encoder(x)
        z = self.gnn(h, edge_index)
        return z

    def predict_edges(self, z: torch.Tensor,
                      edges: torch.Tensor) -> torch.Tensor:
        """对给定边做预测 → [B, num_classes]"""
        u_idx = edges[:, 0]
        v_idx = edges[:, 1]
        z_u = z[u_idx]
        z_v = z[v_idx]
        logits = self.edge_predictor(z_u, z_v)
        return logits

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        """
        x          : [N, in_dim]   全图节点特征
        edge_index : [2, E_gnn]   GNN 消息传递边（无向）
        edges      : [B, 2]       待预测边

        返回: logits [B, num_classes]
        """
        z = self.encode(x, edge_index)
        logits = self.predict_edges(z, edges)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    N, F_in = 50, 22
    E_gnn = 120  # 无向边（已翻倍）
    B = 16

    x = torch.randn(N, F_in)
    edge_index = torch.randint(0, N, (2, E_gnn), dtype=torch.long)
    edges = torch.randint(0, N, (B, 2), dtype=torch.long)

    print("═══ PPINetwork (GCN) ═══")
    model_gcn = PPINetwork(gnn_type="gcn", predictor_type="full")
    logits_gcn = model_gcn(x, edge_index, edges)
    print(f"  Input : x={x.shape} edge_index={edge_index.shape} edges={edges.shape}")
    print(f"  Output: {logits_gcn.shape}")
    assert logits_gcn.shape == (B, NUM_CLASSES)
    n_gcn = model_gcn.count_parameters()
    print(f"  参数量: {n_gcn:,}")

    print("\n═══ PPINetwork (GAT) ═══")
    model_gat = PPINetwork(gnn_type="gat", predictor_type="full", gat_heads=4)
    logits_gat = model_gat(x, edge_index, edges)
    print(f"  Output: {logits_gat.shape}")
    assert logits_gat.shape == (B, NUM_CLASSES)
    n_gat = model_gat.count_parameters()
    print(f"  参数量: {n_gat:,}")

    print("\n═══ PPINetwork (Simple Predictor) ═══")
    model_sp = PPINetwork(gnn_type="gcn", predictor_type="simple")
    logits_sp = model_sp(x, edge_index, edges)
    print(f"  Output: {logits_sp.shape}")
    n_sp = model_sp.count_parameters()
    print(f"  参数量: {n_sp:,}")

    print(f"\n✅ PPINetwork 自检通过")
    print(f"   GCN+Full: {n_gcn:,}  GAT+Full: {n_gat:,}  GCN+Simple: {n_sp:,}")
