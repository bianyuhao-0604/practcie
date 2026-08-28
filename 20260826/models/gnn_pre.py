"""
预训练替代模型（GIN），对筛选后的提示图进行图级二分类。
架构：GIN 编码器 → Sum Pooling ReadOut → Linear → Sigmoid
预训练完成后参数冻结。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, Batch


class GNNpre(nn.Module):
    def __init__(self, d_in: int, d_gin: int = 64, num_layers: int = 2):
        """
        Args:
            d_in: 输入节点特征维度
            d_gin: GIN 隐藏维度（搜索范围 {32, 64, 128}）
            num_layers: GIN 层数（搜索范围 {1, 2, 4}）
        """
        super().__init__()
        self.d_in = d_in
        self.d_gin = d_gin
        self.num_layers = num_layers

        # GIN 编码器
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_dim = d_in
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, d_gin),
                nn.ReLU(),
                nn.Linear(d_gin, d_gin),
            )
            self.gin_layers.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(d_gin))
            in_dim = d_gin

        # 图级输出层
        self.output_layer = nn.Linear(d_gin, 1)

    def forward(self, batch_data):
        """
        Args:
            batch_data: 提示图（PyG Batch），包含 x, edge_index, batch

        Returns:
            y_pre: L3 模式有效性分数 [batch, 1]
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        batch_vec = batch_data.batch

        # GIN 消息传递
        h = x
        for gin_layer, bn in zip(self.gin_layers, self.batch_norms):
            h = gin_layer(h, edge_index)
            h = bn(h)
            h = F.relu(h)

        # Sum Pooling ReadOut
        h_G = global_add_pool(h, batch_vec)  # [batch, d_gin]

        # 输出层
        z = self.output_layer(h_G)
        y_pre = torch.sigmoid(z)

        return y_pre

    def freeze(self):
        """冻结所有参数（预训练完成后调用）"""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self):
        """解冻所有参数（仅在预训练阶段使用）"""
        for param in self.parameters():
            param.requires_grad = True
        self.train()