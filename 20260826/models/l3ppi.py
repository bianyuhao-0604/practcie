"""
L3-PPI 分类头完整实现。
工作流程：
  ① 构造初始提示图 Gpre（可学习嵌入 + 固定 L3 拓扑）
  ② GNNgpt 门控网络 → 路径筛选 → 最终提示图 Gpre'
  ③ GNNpre/GIN（冻结）→ 图级二分类 → L3 有效性分数
  ④ 联合决策 → 最终 PPI 分类结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prompt import PromptEmbeddings
from .gnn_gpt import GNNgpt
from .gnn_pre import GNNpre


class L3PPIClassificationHead(nn.Module):
    def __init__(self, d_model: int = 1280, d_prompt: int = 64,
                 d_gpt: int = 64, d_gin: int = 64,
                 gpt_layers: int = 2, gin_layers: int = 2,
                 K: int = 4, temperature: float = 0.5,
                 gamma: float = 2.0):
        """
        Args:
            d_model: 外部预测器嵌入维度
            d_prompt: 提示图内部统一维度
            d_gpt: GNNgpt 隐藏维度
            d_gin: GNNpre/GIN 隐藏维度
            gpt_layers: GNNgpt 层数
            gin_layers: GNNpre/GIN 层数
            K: 候选 L3 路径数
            temperature: Gumbel-Softmax 温度
            gamma: LPN 正则化超参数
        """
        super().__init__()
        self.K = K
        self.gamma = gamma

        # 组件 ①：可学习提示嵌入
        self.prompt_embeddings = PromptEmbeddings(d_model, d_prompt, K)

        # 组件 ②：GNNgpt 门控网络
        self.gnn_gpt = GNNgpt(
            d_prompt=d_prompt, d_gpt=d_gpt,
            num_layers=gpt_layers, K=K, temperature=temperature,
        )

        # 组件 ③：GNNpre/GIN（预训练后冻结）
        self.gnn_pre = GNNpre(d_in=d_prompt, d_gin=d_gin, num_layers=gin_layers)

    def forward(self, embed_u: torch.Tensor, embed_v: torch.Tensor,
                training: bool = True):
        """
        前向传播。

        Args:
            embed_u: 外部预测器输出的蛋白 u 嵌入 [batch, d_model]
            embed_v: 外部预测器输出的蛋白 v 嵌入 [batch, d_model]
            training: 是否训练模式

        Returns:
            y_pre: L3 模式有效性分数 [batch, 1]
            gate_values: 门控值 [batch, K]
            path_probs: 路径激活概率 [batch, K]
        """
        # 阶段 ①：构造初始提示图
        Gpre = self.prompt_embeddings(embed_u, embed_v)

        # 阶段 ②：GNNgpt 门控筛选
        Gpre_prime, gate_values, path_probs = self.gnn_gpt(Gpre, training=training)

        # 阶段 ③：GNNpre/GIN 图级二分类
        y_pre = self.gnn_pre(Gpre_prime)

        return y_pre, gate_values, path_probs

    def compute_loss(self, y_pre, gate_values, path_probs, labels):
        """
        计算总损失 L_total = L_BCE + L_LPN

        Args:
            y_pre: L3 有效性分数 [batch, 1]
            gate_values: 门控值 [batch, K]
            path_probs: 路径激活概率 [batch, K]
            labels: 真实标签 [batch]

        Returns:
            loss_total, loss_bce, loss_lpn
        """
        # L_BCE：二元交叉熵
        loss_bce = F.binary_cross_entropy(
            y_pre.squeeze(-1), labels.float()
        )

        # L_LPN：路径数量正则化
        K = self.K
        gamma = self.gamma

        pos_mask = (labels == 1).float()
        pos_threshold = K * (1 - 1.0 / gamma)
        pos_sum = path_probs.sum(dim=1)
        pos_penalty = F.relu(pos_threshold - pos_sum) * pos_mask

        neg_mask = (labels == 0).float()
        neg_threshold = K / gamma
        neg_penalty = F.relu(pos_sum - neg_threshold) * neg_mask

        loss_lpn = (pos_penalty.sum() + neg_penalty.sum()) / labels.size(0)

        loss_total = loss_bce + loss_lpn

        return loss_total, loss_bce, loss_lpn

    def freeze_gnn_pre(self):
        """冻结 GNNpre/GIN（预训练完成后调用）"""
        self.gnn_pre.freeze()

    def load_pretrained_gnn_pre(self, checkpoint_path: str):
        """加载预训练的 GNNpre/GIN 权重并冻结"""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.gnn_pre.load_state_dict(state_dict)
        self.freeze_gnn_pre()
        print(f"[L3-PPI] 已加载预训练 GNNpre 权重: {checkpoint_path}")
        print(f"[L3-PPI] GNNpre 参数已冻结")

    @torch.no_grad()
    def predict(self, embed_u: torch.Tensor, embed_v: torch.Tensor):
        """
        推理预测。

        Returns:
            predictions: PPI 预测结果 [batch]（0 或 1）
            scores: L3 有效性分数 [batch, 1]
            active_paths: 被激活的路径索引列表
        """
        self.eval()
        y_pre, gate_values, path_probs = self(embed_u, embed_v, training=False)

        predictions = (y_pre.squeeze(-1) > 0.5).long()

        active_paths = []
        for b in range(gate_values.size(0)):
            active = torch.where(gate_values[b] > 0.5)[0].tolist()
            active_paths.append(active)

        return predictions, y_pre, active_paths