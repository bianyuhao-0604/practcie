"""
dataset.py — PyTorch Geometric Dataset / DataLoader 封装

从预处理好的 .pt 文件加载数据，提供：
  - PPIDataset: 按边索引返回 (node_features, edge_pair, label) 的 Dataset
  - get_dataloaders: 一键返回 train/val/test 三个 DataLoader
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


class PPIDataset(Dataset):
    """
    每条样本 = 一条边（一对蛋白质）的标签。
    返回：
      edge_index : LongTensor [2]    — (u, v) 节点编号
      label      : FloatTensor [7]   — 多标签
    """

    def __init__(self, data_obj, indices: np.ndarray):
        """
        data_obj : PyG Data 对象（含 x, edge_index, edge_pairs, edge_y）
        indices  : 该 split 的边索引数组
        """
        self.x = data_obj.x           # [N, F]
        self.edge_pairs = data_obj.edge_pairs  # [E_total, 2]
        self.edge_y = data_obj.edge_y        # [E_total, 7]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        edge = self.edge_pairs[:,real_idx]   # [2]
        label = self.edge_y[real_idx]      # [7]
        return edge, label


def collate_fn(batch):
    """
    自定义 collate：
      输入 batch 是 list of (edge [2], label [7])
      输出：
        edges  : LongTensor [B, 2]
        labels : FloatTensor [B, 7]
        node_indices : LongTensor [2*B]  — 去重后的节点列表（用于子图提取）
    """
    edges_list = []
    labels_list = []
    for edge, label in batch:
        edges_list.append(edge)
        labels_list.append(label)
    edges = torch.stack(edges_list, dim=0)    # [B, 2]
    labels = torch.stack(labels_list, dim=0)  # [B, 7]
    return edges, labels


def get_dataloaders(data_path: str, batch_size: int = BATCH_SIZE,
                    num_workers: int = 0):
    """
    加载预处理文件，返回三个 DataLoader。
    新格式：文件是字典，内含 PyG Data 对象
    """
    ckpt = torch.load(data_path, map_location="cpu", weights_only=False)
    data = ckpt["data"]

    # 确保 edge_pairs 形状为 [2, E]
    if data.edge_pairs.dim() == 2 and data.edge_pairs.size(0) != 2:
        data.edge_pairs = data.edge_pairs.t()

    train_idx = data.train_idx.numpy()
    val_idx   = data.val_idx.numpy()
    test_idx  = data.test_idx.numpy()

    train_set = PPIDataset(data, train_idx)
    val_set   = PPIDataset(data, val_idx)
    test_set  = PPIDataset(data, test_idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    print(f"[DataLoader] Train={len(train_set)}  Val={len(val_set)}  Test={len(test_set)}")
    print(f"             Batch={batch_size}  NumWorkers={num_workers}")

    return {
        "data": data,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
    }


if __name__ == "__main__":
    # 快速自检（需先跑 data_process.py）
    test_path = os.path.join(PROC_DIR, f"SHS148k_{SPLIT_MODE}.pt")
    if os.path.exists(test_path):
        loaders = get_dataloaders(test_path)
        for name in ["train_loader", "val_loader", "test_loader"]:
            dl = loaders[name]
            edges, labels = next(iter(dl))
            print(f"{name}: edges={edges.shape}, labels={labels.shape}, "
                  f"pos_rate={labels.sum()/labels.numel():.4f}")
        print("✅ Dataset 自检通过")
    else:
        print(f"⚠️ 预处理文件不存在: {test_path}")
        print("   请先运行: python data_process.py")
