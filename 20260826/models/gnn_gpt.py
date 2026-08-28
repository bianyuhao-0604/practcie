"""
门控网络：从 K 条候选 L3 路径中选择性激活有效路径。
架构：GNN 编码器 → 路径级特征提取 → 门控评分 → Gumbel-Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch


class GNNgpt(nn.Module):
    def __init__(self, d_prompt: int, d_gpt: int = 64, num_layers: int = 2,
                 K: int = 4, temperature: float = 0.5):
        """
        Args:
            d_prompt: 输入节点特征维度
            d_gpt: GNNgpt 隐藏维度（搜索范围 {64, 128}）
            num_layers: GNN 层数（搜索范围 {1, 2, 4}）
            K: 候选 L3 路径数
            temperature: Gumbel-Softmax 温度参数
        """
        super().__init__()
        self.d_prompt = d_prompt
        self.d_gpt = d_gpt
        self.K = K
        self.num_layers = num_layers
        self.temperature = temperature

        # GNN 编码器（多层 GCN）
        self.gnn_layers = nn.ModuleList()
        in_dim = d_prompt
        for _ in range(num_layers):
            self.gnn_layers.append(GCNConv(in_dim, d_gpt))
            in_dim = d_gpt

        # 路径级特征提取：拼接 3 个节点嵌入 → d_gpt * 3
        d_path = d_gpt * 3
        self.d_path = d_path

        # 门控评分层
        self.gate_layer = nn.Linear(d_path, 1)

    def forward(self, batch_data: Batch, training: bool = True):
        """
        Args:
            batch_data: 初始提示图 Gpre（PyG Batch）
            training: 是否处于训练模式

        Returns:
            gated_batch_data: 最终提示图 Gpre'（边被掩码）
            gate_values: 门控值 [batch, K]
            path_probs: 原始激活概率 [batch, K]
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        batch_vec = batch_data.batch

        K = self.K

        # ---- 步骤 2.1：GNN 消息传递（L 层）----
        h = x
        for layer in self.gnn_layers:
            h = layer(h, edge_index)
            h = F.relu(h)
        # h: [total_nodes, d_gpt]

        # ---- 步骤 2.2：路径级特征提取 ----
        h_dense, _ = to_dense_batch(h, batch_vec)
        # h_dense: [batch, max_nodes, d_gpt]

        h_u = h_dense[:, 0, :]       # [batch, d_gpt]
        h_vP0 = h_dense[:, 2, :]     # [batch, d_gpt]

        path_features = []
        for i in range(1, K + 1):
            h_vPi = h_dense[:, i + 2, :]
            z_i = torch.cat([h_u, h_vPi, h_vP0], dim=-1)
            path_features.append(z_i)

        Z = torch.stack(path_features, dim=1)  # [batch, K, d_path]

        # ---- 步骤 2.3：门控评分 ----
        logits = self.gate_layer(Z).squeeze(-1)  # [batch, K]
        path_probs = torch.sigmoid(logits)

        # ---- 步骤 2.4：Gumbel-Softmax 采样 ----
        if training:
            gate_values = F.gumbel_softmax(
                logits, tau=self.temperature, hard=True
            )
        else:
            gate_values = (torch.sigmoid(logits) > 0.5).float()

        # ---- 步骤 2.5：构造最终提示图 Gpre' ----
        gated_edge_index = self._mask_edges(
            edge_index, batch_vec, gate_values,
            batch_size=h_dense.size(0)
        )

        gated_batch_data = Data(
            x=x,
            edge_index=gated_edge_index,
            batch=batch_vec,
        )

        return gated_batch_data, gate_values, path_probs

    def _mask_edges(self, edge_index, batch_vec, gate_values, batch_size):
        """根据门控值移除被抑制路径的边"""
        K = self.K
        device = edge_index.device
        num_edges_per_graph = 3 * K

        # 构建边到路径的映射
        edge_to_path = []
        for i in range(1, K + 1):
            edge_to_path.extend([i - 1] * 3)
        edge_to_path = torch.tensor(
            edge_to_path, dtype=torch.long, device=device
        )

        total_edges = edge_index.size(1)
        edge_graph_idx = torch.arange(total_edges, device=device) // num_edges_per_graph
        edge_path_idx = edge_to_path.repeat(batch_size)
        edge_gate = gate_values[edge_graph_idx, edge_path_idx]

        keep_mask = edge_gate.bool()
        gated_edge_index = edge_index[:, keep_mask]

        return gated_edge_index