"""
可学习提示嵌入模块。
维护 K+1 个虚拟节点嵌入（全局共享），
并与外部预测器输出的蛋白嵌入拼接，构造初始提示图 Gpre。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


class PromptEmbeddings(nn.Module):
    def __init__(self, d_model: int, d_prompt: int, K: int):
        """
        Args:
            d_model: 外部预测器输出嵌入维度（如 ESM2-650M 为 1280）
            d_prompt: 提示图内部统一维度（如 64 或 128）
            K: 候选 L3 路径数（超参数搜索范围 {4, 16, 64}）
        """
        super().__init__()
        self.d_model = d_model
        self.d_prompt = d_prompt
        self.K = K

        # 投影层：将外部预测器嵌入映射到统一维度
        self.proj = nn.Linear(d_model, d_prompt)

        # K+1 个可学习虚拟节点嵌入（全局共享）
        # x_P0: 中心节点, x_P1~x_PK: 路径节点
        self.virtual_node_embeddings = nn.Parameter(
            torch.randn(K + 1, d_prompt) * 0.01
        )

    def forward(self, embed_u: torch.Tensor, embed_v: torch.Tensor) -> Batch:
        """
        构造初始提示图 Gpre。

        Args:
            embed_u: 外部预测器输出的蛋白 u 嵌入 [batch, d_model]
            embed_v: 外部预测器输出的蛋白 v 嵌入 [batch, d_model]

        Returns:
            batch_data: PyG Batch 对象
                每个图: K+3 个节点, 3K 条边
                节点顺序: [u, v, vP0, vP1, ..., vPK]
                索引:      0   1   2    3    ...   K+2
        """
        batch_size = embed_u.size(0)
        K = self.K
        device = embed_u.device

        # 步骤 1.1：投影外部嵌入到统一维度
        x_u = self.proj(embed_u)  # [batch, d_prompt]
        x_v = self.proj(embed_v)  # [batch, d_prompt]

        # 步骤 1.2：获取虚拟节点嵌入（全局共享，广播到 batch）
        x_vp = self.virtual_node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        # x_vp: [batch, K+1, d_prompt]

        # 步骤 1.3：组装节点特征矩阵
        x_u = x_u.unsqueeze(1)  # [batch, 1, d_prompt]
        x_v = x_v.unsqueeze(1)  # [batch, 1, d_prompt]
        node_features = torch.cat([x_u, x_v, x_vp], dim=1)
        # node_features: [batch, K+3, d_prompt]

        # 步骤 1.4：构造边集（K 条 L3 路径，每条 3 条边）
        # 路径 i: u(0) → vP_i(i+2) → vP0(2) → v(1)
        edge_list = []
        for i in range(1, K + 1):
            edge_list.append([0, i + 2])      # u → vP_i
            edge_list.append([i + 2, 2])      # vP_i → vP0
            edge_list.append([i + 2, 1])      # vP_i → v

        edge_index_single = torch.tensor(
            edge_list, dtype=torch.long, device=device
        ).t().contiguous()  # [2, 3K]

        # 构建 batch 级别的图
        data_list = []
        for b in range(batch_size):
            data = Data(
                x=node_features[b],
                edge_index=edge_index_single.clone(),
            )
            data_list.append(data)

        batch_data = Batch.from_data_list(data_list)
        return batch_data