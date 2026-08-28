import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os

class L3PathPretrainDataset(Dataset):
    def __init__(self, pretrain_dir: str, embeddings_path: str, d_model: int = 1280):
        self.d_model = d_model

        # 加载路径和标签
        pos_paths = torch.load(os.path.join(pretrain_dir, "positive_L3_paths.pt"))
        neg_paths = torch.load(os.path.join(pretrain_dir, "negative_L3_paths.pt"))
        self.paths = torch.cat([pos_paths, neg_paths], dim=0)
        self.labels = torch.cat([
            torch.ones(pos_paths.size(0), dtype=torch.long),
            torch.zeros(neg_paths.size(0), dtype=torch.long),
        ])

        # 加载嵌入
        self.embeddings = torch.load(embeddings_path)

        # 加载映射（如果存在）
        mapping_path = os.path.join(pretrain_dir, "protein_id_mapping.pt")
        if os.path.exists(mapping_path):
            mapping = torch.load(mapping_path)
            self.idx_to_id = mapping.get("idx_to_node", None)
            print("已加载 protein_id_mapping.pt，使用字符串 ID 作为键")
        else:
            self.idx_to_id = None
            print("未找到 protein_id_mapping.pt，假设嵌入键为整数索引")

        # 统计缺失的 ID 数量（仅用于信息，不过滤）
        missing_ids = set()
        for i in range(self.paths.size(0)):
            for node_idx in self.paths[i]:
                idx_int = node_idx.item()
                if self.idx_to_id is not None:
                    pid = self.idx_to_id[idx_int]
                    if pid not in self.embeddings:
                        missing_ids.add(pid)
                else:
                    if idx_int not in self.embeddings and str(idx_int) not in self.embeddings:
                        missing_ids.add(idx_int)

        print(f"数据集总样本数: {self.paths.size(0)}")
        print(f"缺失嵌入的蛋白质 ID 数量: {len(missing_ids)}")
        if missing_ids:
            print(f"缺失 ID 示例: {list(missing_ids)[:10]}")

    def __len__(self):
        return self.paths.size(0)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        node_features = []
        for node_idx in path:
            idx_int = node_idx.item()
            if self.idx_to_id is not None:
                protein_id = self.idx_to_id[idx_int]
                feat = self.embeddings.get(protein_id)
                if feat is None:
                    feat = torch.zeros(self.d_model)   # 缺失时用零向量
            else:
                feat = self.embeddings.get(idx_int)
                if feat is None:
                    feat = self.embeddings.get(str(idx_int))
                if feat is None:
                    feat = torch.zeros(self.d_model)
            node_features.append(feat)

        x = torch.stack(node_features)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=label.unsqueeze(0))
        return data