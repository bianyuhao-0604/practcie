"""
评测基准数据集：SHS27k / SHS148k。

每个样本是一个蛋白对 (u, v)，标签为是否相互作用。
蛋白特征来自外部预测器（ESM2）的嵌入。

数据格式：
  train.pt / test.pt: {
      'protein1_idx': tensor [N],
      'protein2_idx': tensor [N],
      'binary_labels': tensor [N],
      'action_labels': tensor [N, 7],
  }
  esm2_embeddings.pt: {protein_id: tensor [d_model]}
"""

import torch
from torch.utils.data import Dataset
import os


class PPIBenchmarkDataset(Dataset):
    """
    PPI 评测基准数据集。

    每个样本是一个蛋白对 (u, v)：
      - embed_u: [d_model] 蛋白 u 的 ESM2 嵌入
      - embed_v: [d_model] 蛋白 v 的 ESM2 嵌入
      - label: 0 或 1（是否相互作用）
    """

    def __init__(self, data_path: str, embeddings_path: str,
                 d_model: int = 1280):
        """
        Args:
            data_path: train.pt 或 test.pt 的路径
            embeddings_path: ESM2 嵌入文件路径
            d_model: 嵌入维度
        """
        self.d_model = d_model

        # 加载样本数据
        data = torch.load(data_path)
        self.p1_idx = data['protein1_idx']
        self.p2_idx = data['protein2_idx']
        self.labels = data['binary_labels']

        # 加载嵌入
        self.embeddings = torch.load(embeddings_path)

        # 构建索引到嵌入的映射
        if isinstance(self.embeddings, dict):
            first_key = list(self.embeddings.keys())[0]
            if isinstance(first_key, str):
                self._use_str_keys = True
            else:
                self._use_str_keys = False
        else:
            self._use_str_keys = False

        print(f"[BenchmarkDataset] 样本数: {len(self.p1_idx)}, "
              f"正样本: {self.labels.sum().item()}, "
              f"负样本: {(1 - self.labels).sum().item()}")

    def __len__(self):
        return len(self.p1_idx)

    def __getitem__(self, idx):
        """
        返回:
          embed_u: [d_model]
          embed_v: [d_model]
          label: scalar
        """
        p1 = self.p1_idx[idx].item()
        p2 = self.p2_idx[idx].item()
        label = self.labels[idx]

        # 获取嵌入
        embed_u = self._get_embedding(p1)
        embed_v = self._get_embedding(p2)

        return embed_u, embed_v, label

    def _get_embedding(self, protein_idx):
        """根据蛋白索引获取嵌入向量"""
        if self._use_str_keys:
            key = str(protein_idx)
        else:
            key = protein_idx

        if isinstance(self.embeddings, dict):
            emb = self.embeddings.get(key)
            if emb is None:
                emb = torch.zeros(self.d_model)
            return emb
        else:
            return self.embeddings[protein_idx]