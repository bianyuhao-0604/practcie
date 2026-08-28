import torch
import numpy as np
from torch_geometric.datasets import Planetoid,TUDataset
from torch_geometric.data import Data,Batch
from torch_geometric.transforms import Compose,NormalizeFeatures
from sklearn.model_selection import StratifiedKFold
from config import (
    DATA_DIR,DATASET_NAME,TASK_TYPE,
    N_SPLITS,TRAIN_RATTO,VAL_RATIO,RANDOM_SEED,
)
#数据变换
def get_transforms():
    return Compose([
        NormalizeFeatures(),# 行归一化特征
    ])
#加载 PyG 内置数据集
def load_builtin_dataset():
    transform = get_transforms()
    if TASK_TYPE == 'node':
        dataset = Planetoid(
            root=str(DATA_DIR / 'Planetoid'),#数据存储路径，这里为 DATA_DIR/Planetoid
            name=DATASET_NAME,
            transform=transform,
        )
        data = dataset[0]
        print(f"[数据] {DATASET_NAME} | 节点: {data.num_nodes} | "
              f"边: {data.num_edges} | 特征维度: {dataset.num_node_features} | "
              f"类别: {dataset.num_classes}")
        return data,dataset
    elif TASK_TYPE == 'graph':
        dataset = TUDataset(
            root=str(DATA_DIR / "TUDataset"),
            name=DATASET_NAME,
            transform=transform,
        )
        print(f"[数据] {DATASET_NAME} | 图数量: {len(dataset)} | "
              f"特征维度: {dataset.num_node_features} | "
              f"类别: {dataset.num_classes}")
        return dataset,dataset
    else:
        raise ValueError(f"不支持的任务类型: {TASK_TYPE}")
#加载自定义图数据
def load_custom_graph(path):
    data = torch.load(path)#使用 PyTorch 加载序列化的对象（通常是 Data 对象）
    print(f"[数据] 自定义图 | 节点: {data.num_nodes} | "
          f"边: {data.num_edges} | 特征维度: data.x.shape[1]")
    return data
#节点分类: 随机划分掩码
def generate_node_masks(num_nodes,labels,train_ratio=TRAIN_RATTO,
                        val_ratio=VAL_RATIO,seed=RANDOM_SEED):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_nodes,generator=g)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)
    train_mask = torch.zeros(num_nodes,dtype=torch.bool)
    val_mask = torch.zeros(num_nodes,dtype=torch.bool)
    test_mask = torch.zeros(num_nodes,dtype=torch.bool)
    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train+n_val]] = True
    test_mask[indices[n_train+n_val:]] = True
    return train_mask,val_mask,test_mask
#图分类: K-Fold 划分
def generate_graph_folds(dataset,n_splits=N_SPLITS,seed=RANDOM_SEED):
    labels = np.array([data.y.item() for data in dataset])
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    folds = []
    for train_val_idx,test_idx in skf.split(np.zeros(len(labels)),labels):
        n_val = int(len(train_val_idx) * VAL_RATIO / (TRAIN_RATTO + VAL_RATIO))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train_val_idx),generator=g)
        val_idx = train_val_idx[perm[:n_val]]
        train_idx = train_val_idx[perm[n_val:]]
        folds.append((train_idx,val_idx,test_idx))
    return folds


