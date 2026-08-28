#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3-PPI 论文中的 GIN (Graph Isomorphism Network) PyTorch Geometric 实现

参考论文: Xu et al., 2018 "How Powerful are Graph Neural Networks?"
论文中角色: GNNpre (预训练替代模型/判别器), 预训练后冻结

架构: GIN编码器(多层GINConv堆叠) -> ReadOut(Sum Pooling) -> 图级输出层(Linear + Sigmoid)
参数: theta (GIN编码器权重) + phi (ReadOut + 输出层参数)
超参数: 层数 in {1, 2, 4}, 维度 in {32, 64, 128}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from typing import Optional, List, Tuple
import numpy as np


# ============================================================================
# GINConv (基于 Xu et al., 2018 原始架构)
# ============================================================================

class GINConv(MessagePassing):
    """
    GIN 卷积层, 计算公式:
        h_v^(k) = MLP^(k)( (1 + epsilon^(k)) * h_v^(k-1) + Sum_{u in N(v)} h_u^(k-1) )

    其中:
        - Sum 聚合是单射(injective)的, 能够完整保留邻居多重集信息
        - epsilon^(k) 是可学习标量参数, 控制自身特征的权重
        - MLP 由 Linear -> ReLU -> Linear 组成
    """

    def __init__(self, in_dim: int, hidden_dim: int, eps: float = 0.0,
                 train_eps: bool = False, dropout: float = 0.0):
        """
        Args:
            in_dim:     输入特征维度
            hidden_dim: 隐藏层维度
            eps:        初始 epsilon 值 (可被 train_eps 覆盖)
            train_eps:  是否训练 epsilon (论文中为可学习参数)
            dropout:    Dropout 率
        """
        super(GINConv, self).__init__(aggr='add')  # 'add' 即 Sum 聚合

        # 可学习标量 epsilon
        self.eps = nn.Parameter(torch.Tensor([eps]))
        if train_eps:
            self.eps.requires_grad = True
        else:
            self.eps.requires_grad = False

        # MLP: Linear -> ReLU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 可选 Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        Args:
            x:          节点特征矩阵 [num_nodes, in_dim]
            edge_index: 边索引矩阵 [2, num_edges]
        Returns:
            输出节点特征 [num_nodes, hidden_dim]
        """
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # (1 + eps) * h_v
        x_self = (1 + self.eps) * x

        # 执行 Sum 聚合 + 自环
        out = x_self + self.propagate(edge_index, x=x)

        # MLP 变换
        out = self.mlp(out)
        out = self.dropout(out)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


# ============================================================================
# GINEConv (带边特征的 GINConv, 论文扩展)
# ============================================================================

class GINEConv(MessagePassing):
    """
    GIN 卷积层 (带边特征版本)
    计算公式:
        h_v^(k) = MLP^(k)( (1 + epsilon^(k)) * h_v^(k-1) + Sum_{u in N(v)} (h_u^(k-1) + e_uv) )
    """

    def __init__(self, in_dim: int, hidden_dim: int, eps: float = 0.0,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 dropout: float = 0.0):
        super(GINEConv, self).__init__(aggr='add')

        self.eps = nn.Parameter(torch.Tensor([eps]))
        if train_eps:
            self.eps.requires_grad = True
        else:
            self.eps.requires_grad = False

        if edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.edge_encoder = None
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x_self = (1 + self.eps) * x

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            out = x_self + self.propagate(edge_index, x=x, edge_attr=edge_attr)
        else:
            out = x_self + self.propagate(edge_index, x=x)

        out = self.mlp(out)
        out = self.dropout(out)
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return x_j + edge_attr
        return x_j


# ============================================================================
# GIN 编码器 (多层 GINConv 堆叠)
# ============================================================================

class GINEncoder(nn.Module):
    """
    GIN 编码器: 由多层 GINConv 堆叠而成

    输入: 节点特征矩阵 + 边索引
    输出: 每个节点的最终嵌入向量
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int,
                 eps: float = 0.0, train_eps: bool = False,
                 dropout: float = 0.0):
        """
        Args:
            in_dim:       输入特征维度
            hidden_dim:   隐藏层维度 (所有 GINConv 共享)
            num_layers:   GINConv 层数 (论文搜索范围: {1, 2, 4})
            eps:          初始 epsilon 值
            train_eps:    是否训练 epsilon
            dropout:      Dropout 率
        """
        super(GINEncoder, self).__init__()

        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer_in_dim = in_dim if layer_idx == 0 else hidden_dim
            conv = GINConv(
                in_dim=layer_in_dim,
                hidden_dim=hidden_dim,
                eps=eps,
                train_eps=train_eps,
                dropout=dropout
            )
            bn = nn.BatchNorm1d(hidden_dim)
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)

    def forward(self, x, edge_index):
        """
        Args:
            x:          节点特征矩阵 [num_nodes, in_dim]
            edge_index: 边索引矩阵 [2, num_edges]
        Returns:
            节点嵌入矩阵 [num_nodes, hidden_dim]
        """
        h = x
        for layer_idx in range(self.num_layers):
            h = self.conv_layers[layer_idx](h, edge_index)
            h = self.bn_layers[layer_idx](h)
            h = F.relu(h)
            h = F.dropout(h, p=0.1, training=self.training)
        return h


# ============================================================================
# ReadOut (全局图级池化)
# ============================================================================

class ReadOut(nn.Module):
    """
    全局图级池化: 对图中所有节点嵌入做 Sum Pooling (全局求和)

    将节点级表示 h_v 聚合为图级表示 h_G:
        h_G = Sum_{v in G} h_v

    输出维度 = GIN 隐藏维度
    """

    def __init__(self, mode: str = 'sum'):
        super(ReadOut, self).__init__()
        self.mode = mode

    def forward(self, x, batch=None):
        """
        Args:
            x:     节点特征矩阵 [num_nodes, hidden_dim]
            batch: 节点-图分配向量 [num_nodes], 用于批量处理
        Returns:
            图级表示矩阵 [num_graphs, hidden_dim]
        """
        if batch is None:
            return torch.sum(x, dim=0, keepdim=True)

        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        out = out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        return out


# ============================================================================
# 图级输出层 (Task Head)
# ============================================================================

class GraphClassifierHead(nn.Module):
    """
    图级分类头: 将图级表示映射为二分类输出

    输入维度 = GIN 隐藏维度
    输出维度 = 1 (标量分数, 表示 L3 模式有效性)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super(GraphClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph_repr):
        """
        Args:
            graph_repr: 图级表示 [num_graphs, hidden_dim]
        Returns:
            分类 logits [num_graphs, 1]
        """
        return self.classifier(graph_repr)


# ============================================================================
# 完整模型: GNNpre (GNN-based Pretrained Model)
# ============================================================================

class GNNpre(nn.Module):
    """
    L3-PPI 论文中的预训练替代模型 GNNpre

    完整架构:
        输入图 Gpre -> GIN编码器(theta) -> ReadOut(Sum Pooling) -> 图级输出层(phi) -> 标量分数

    参数组成:
        theta: GIN 编码器的网络权重
        phi:   ReadOut + 分类头的参数

    预训练目标:
        min L_pre(y_pre, GNNpre(G_pre; theta, phi))
        二元交叉熵损失

    微调阶段:
        theta* 和 phi* 被冻结, 不再更新
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int,
                 eps: float = 0.0, train_eps: bool = False,
                 dropout: float = 0.1):
        """
        Args:
            in_dim:       输入节点特征维度
            hidden_dim:   GIN 隐藏维度 (论文搜索范围: {32, 64, 128})
            num_layers:   GIN 层数 (论文搜索范围: {1, 2, 4})
            eps:          初始 epsilon 值
            train_eps:    是否训练 epsilon
            dropout:      Dropout 率 (论文搜索范围: {0.1, 0.2})
        """
        super(GNNpre, self).__init__()

        # 第1部分: GIN 编码器
        self.encoder = GINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            eps=eps,
            train_eps=train_eps,
            dropout=dropout
        )

        # 第2部分: ReadOut (Sum Pooling)
        self.readout = ReadOut(mode='sum')

        # 第3部分: 图级输出层
        self.classifier = GraphClassifierHead(hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x, edge_index, batch=None):
        """
        前向传播

        Args:
            x:          节点特征矩阵 [num_nodes, in_dim]
            edge_index: 边索引矩阵 [2, num_edges]
            batch:      节点-图分配向量 [num_nodes] (批量处理时使用)
        Returns:
            logits:     分类 logits [num_graphs, 1]
        """
        node_repr = self.encoder(x, edge_index)
        graph_repr = self.readout(node_repr, batch)
        logits = self.classifier(graph_repr)
        return logits

    def predict(self, x, edge_index, batch=None):
        """
        预测模式: 返回 Sigmoid 概率

        Args:
            x:          节点特征矩阵 [num_nodes, in_dim]
            edge_index: 边索引矩阵 [2, num_edges]
            batch:      节点-图分配向量 [num_nodes]
        Returns:
            prob:       L3 模式有效性概率 [num_graphs, 1]
        """
        logits = self.forward(x, edge_index, batch)
        return torch.sigmoid(logits)

    def freeze(self):
        """微调阶段: 冻结所有参数 (预训练后不再更新)"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """预训练阶段: 解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

    def count_params(self):
        """统计可训练参数数量"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total


# ============================================================================
# L3 路径数据集构造器
# ============================================================================

class L3PathDataset:
    """
    L3 路径数据集构造器

    预训练数据构造方式:
        正样本: 交互蛋白对之间的所有 L3 路径 (通过深度优先搜索获取)
        负样本: 非交互蛋白对之间的所有 L3 路径
    """

    def __init__(self, ppi_network, positive_pairs, negative_pairs, l3_paths):
        self.ppi_network = ppi_network
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs
        self.l3_paths = l3_paths

    def construct_samples(self):
        """构造预训练样本"""
        positive_graphs = []
        for u, v in self.positive_pairs:
            if (u, v) in self.l3_paths:
                for path in self.l3_paths[(u, v)]:
                    positive_graphs.append((path, 1))

        negative_graphs = []
        for u, v in self.negative_pairs:
            if (u, v) in self.l3_paths:
                for path in self.l3_paths[(u, v)]:
                    negative_graphs.append((path, 0))

        return positive_graphs, negative_graphs


# ============================================================================
# 预训练训练循环
# ============================================================================

def train_gnnpre(model, train_loader, optimizer, device):
    """
    GNNpre 预训练训练循环

    Args:
        model:       GNNpre 模型
        train_loader: 训练数据加载器
        optimizer:   优化器 (Adam)
        device:      计算设备
    Returns:
        avg_loss: 平均训练损失
    """
    model.train()
    total_loss = 0

    for batch in train_loader:
        x, edge_index, batch_idx, labels = batch
        x, edge_index, labels = x.to(device), edge_index.to(device), labels.to(device)

        logits = model(x, edge_index, batch_idx)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(train_loader.dataset)


def evaluate_gnnpre(model, val_loader, device):
    """
    GNNpre 评估函数

    Args:
        model:      GNNpre 模型
        val_loader: 验证数据加载器
        device:     计算设备
    Returns:
        accuracy:   验证准确率
        f1:         验证 F1 分数
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x, edge_index, batch_idx, labels = batch
            x, edge_index, labels = x.to(device), edge_index.to(device), labels.to(device)

            logits = model(x, edge_index, batch_idx)
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return accuracy, f1


# ============================================================================
# 超参数搜索配置
# ============================================================================

HYPARAM_GRID = {
    'num_layers': [1, 2, 4],
    'hidden_dim': [32, 64, 128],
    'dropout': [0.1, 0.2],
    'lr': [1e-4, 1e-3],
    'optimizer': ['adam'],
    'weight_decay': [0, 1e-5],
}


def create_model(in_dim, num_layers, hidden_dim, dropout=0.1):
    """
    创建 GNNpre 模型实例

    Args:
        in_dim:       输入特征维度
        num_layers:   GIN 层数
        hidden_dim:   GIN 隐藏维度
        dropout:      Dropout 率
    Returns:
        model: GNNpre 模型
    """
    model = GNNpre(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        eps=0.0,
        train_eps=True,
        dropout=dropout
    )
    return model


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GIN (GNNpre) 模型使用示例")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 示例: 5个节点, 6条边
    num_nodes = 5
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 4],
    ], dtype=torch.long)

    node_features = torch.randn(num_nodes, 32)

    # 创建模型 (层数=2, 维度=64)
    model = create_model(in_dim=32, num_layers=2, hidden_dim=64, dropout=0.1)
    model.to(device)

    print(f"模型可训练参数: {model.count_params()}")
    print(f"模型结构:")
    print(model)

    # 前向传播 (单图)
    model.train()
    logits = model(node_features, edge_index)
    print(f"\n前向传播输出 logits: {logits.detach().cpu().numpy()}")

    prob = model.predict(node_features, edge_index)
    print(f"预测概率: {prob.detach().cpu().numpy()}")

    # 冻结参数 (模拟预训练后冻结)
    model.freeze()
    print(f"\n参数已冻结 (requires_grad={next(model.parameters()).requires_grad})")

    # 统计各层参数
    print("\n--- 各层参数统计 ---")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.numel()} 参数, requires_grad={param.requires_grad}")

    print("\n示例完成!")
