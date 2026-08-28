"""
data_process.py — SHS148k 原始数据 → 图结构 + 特征矩阵 + 划分

功能：
  1. 读取原始序列文件和 PPI 注释
  2. 构建蛋白质节点特征矩阵（氨基酸组成）
  3. 构建 PPI 边（无向图）
  4. 生成多标签 Y 矩阵（N×7）
  5. 按 random / bfs / dfs 三种方式划分边
  6. 负采样（1:1）
  7. 保存为 .pt 文件供 DataLoader 使用

用法：
  python data_process.py --seq_path data/xxx.tsv --ppi_path data/yyy.txt \
                         --out_dir data/processed --split_mode bfs
"""

import argparse
import os
import sys
import json
import random
import itertools
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch_geometric.data import Data

# 允许直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from aa_utils import (
    load_sequences, load_ppi_actions,
    amino_acid_composition, AA_ALPHABET, NUM_AA,
)

# ───────────────────── 类别映射 ─────────────────────
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES  = len(CLASS_NAMES)


# ───────────────────── 边划分策略 ─────────────────────
def split_edges_random(edges, y, train_r=0.6, val_r=0.2, seed=42):
    """随机划分边为 train/val/test"""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(edges))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def split_edges_bfs(edges, y, train_r=0.6, val_r=0.2, seed=42):
    """BFS 划分（支持多连通分量）"""
    rng = random.Random(seed)
    n_edges = len(edges)

    neighbors = defaultdict(set)
    for (u, v) in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    all_nodes = list(neighbors.keys())
    rng.shuffle(all_nodes)

    visited_global = set()
    bfs_edge_order = []

    for start in all_nodes:
        if start in visited_global:
            continue
        queue = [start]
        visited_global.add(start)
        while queue:
            node = queue.pop(0)
            for nb in sorted(neighbors[node]):
                if nb not in visited_global:
                    visited_global.add(nb)
                    queue.append(nb)
                    e_tuple = tuple(sorted([node, nb]))
                    bfs_edge_order.append(e_tuple)

    # 建立边→索引映射
    edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    ordered_indices = []
    seen = set()
    for e in bfs_edge_order:
        if e in edge_to_idx and e not in seen:
            ordered_indices.append(edge_to_idx[e])
            seen.add(e)
    # 补齐未覆盖的边
    for i in range(n_edges):
        if i not in ordered_indices:
            ordered_indices.append(i)

    n = n_edges
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    train_idx = np.array(ordered_indices[:n_train])
    val_idx   = np.array(ordered_indices[n_train:n_train + n_val])
    test_idx  = np.array(ordered_indices[n_train + n_val:])
    return train_idx, val_idx, test_idx


def split_edges_dfs(edges, y, train_r=0.6, val_r=0.2, seed=42):
    """
    DFS 划分（迭代版，避免递归深度溢出）
    使用显式栈进行深度优先遍历。
    """
    rng = random.Random(seed)
    n_edges = len(edges)

    # 构建邻接表
    neighbors = defaultdict(set)
    for (u, v) in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    all_nodes = list(neighbors.keys())
    rng.shuffle(all_nodes)

    visited_nodes = set()
    dfs_edge_order = []

    # 遍历所有连通分量（防止不连通图）
    for start in all_nodes:
        if start in visited_nodes:
            continue
        # 显式栈，元素为 (node, iterator_index)
        stack = [(start, iter(sorted(neighbors[start])))]
        visited_nodes.add(start)
        while stack:
            node, nb_iter = stack[-1]
            try:
                nb = next(nb_iter)
                if nb not in visited_nodes:
                    visited_nodes.add(nb)
                    e_tuple = tuple(sorted([node, nb]))
                    dfs_edge_order.append(e_tuple)
                    stack.append((nb, iter(sorted(neighbors[nb]))))
            except StopIteration:
                stack.pop()

    # 建立边→索引映射
    edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    ordered_indices = []
    seen = set()
    for e in dfs_edge_order:
        if e in edge_to_idx and e not in seen:
            ordered_indices.append(edge_to_idx[e])
            seen.add(e)
    # 补齐未覆盖的边（孤立边或跨连通分量边）
    for i in range(n_edges):
        if i not in ordered_indices:
            ordered_indices.append(i)

    n = n_edges
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    train_idx = np.array(ordered_indices[:n_train])
    val_idx   = np.array(ordered_indices[n_train:n_train + n_val])
    test_idx  = np.array(ordered_indices[n_train + n_val:])
    return train_idx, val_idx, test_idx

# ───────────────────── 负采样 ─────────────────────
def generate_negative_edges(positive_edges, num_nodes, n_neg, seed=42):
    """
    从不存在的边中采样 n_neg 条负样本。
    使用 set 加速查重。
    """
    rng = np.random.RandomState(seed)
    pos_set = set(tuple(sorted(e)) for e in positive_edges)
    neg_set = set()
    max_attempts = n_neg * 20
    attempts = 0
    while len(neg_set) < n_neg and attempts < max_attempts:
        u = rng.randint(0, num_nodes - 1)
        v = rng.randint(0, num_nodes - 1)
        if u == v:
            attempts += 1
            continue
        e = tuple(sorted([u, v]))
        if e not in pos_set and e not in neg_set:
            neg_set.add(e)
        attempts += 1
    return [list(e) for e in neg_set]


# ───────────────────── 主流程 ─────────────────────
def process(seq_path: str, ppi_path: str, out_dir: str, split_mode: str = "bfs"):
    print(f"\n{'='*60}")
    print(f"  SHS148k 数据预处理  |  划分模式: {split_mode.upper()}")
    print(f"{'='*60}\n")

    # 1. 加载序列
    seq_dict = load_sequences(seq_path)
    protein_ids = list(seq_dict.keys())
    num_nodes = len(protein_ids)
    print(f"[1] 蛋白质节点数: {num_nodes}")

    # 2. 编码节点特征
    print("[2] 编码氨基酸组成特征 ...")
    x = np.zeros((num_nodes, NUM_AA), dtype=np.float32)
    for i, pid in enumerate(protein_ids):
        x[i] = amino_acid_composition(seq_dict[pid])
    print(f"    特征矩阵 X: {x.shape}")

    # 3. 加载 PPI 注释
    ppi_records = load_ppi_actions(ppi_path)

    # 建立 protein_id → node_index 映射
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    # 4. 构建正样本边 + 多标签 Y
    print("[3] 构建 PPI 边和多标签矩阵 ...")
    edge_set = set()          # 去重后的正边
    edge_labels = defaultdict(set)  # edge → set of class indices

    skipped = 0
    for id1, id2, action, score in ppi_records:
        if id1 not in id_to_idx or id2 not in id_to_idx:
            skipped += 1
            continue
        u, v = id_to_idx[id1], id_to_idx[id2]
        if u == v:
            continue
        e = tuple(sorted([u, v]))
        edge_set.add(e)
        if action in CLASS_TO_IDX:
            edge_labels[e].add(CLASS_TO_IDX[action])

    if skipped:
        print(f"    ⚠️ 跳过 {skipped} 条含未知蛋白的 PPI 记录")

    pos_edges = [list(e) for e in edge_set]
    num_pos = len(pos_edges)
    print(f"    正样本边数: {num_pos}")

    # 构建 Y 矩阵（多标签，N_edges × 7）
    y = np.zeros((num_pos, NUM_CLASSES), dtype=np.float32)
    for i, e in enumerate(pos_edges):
        for cidx in edge_labels[tuple(sorted(e))]:
            y[i, cidx] = 1.0

    # 5. 类别分布统计
    print("\n[4] 类别分布统计:")
    for i, name in enumerate(CLASS_NAMES):
        cnt = int(y[:, i].sum())
        pct = cnt / num_pos * 100
        print(f"    {name:12s}: {cnt:6d} ({pct:5.2f}%)")
    print(f"    {'TOTAL':12s}: {num_pos:6d}")
    # 在第5步（类别统计）之后，第6步（负采样）之前，先划分正样本
    print("\n[5] 划分正样本边 ...")
    if split_mode == "random":
        train_pos_idx, val_pos_idx, test_pos_idx = split_edges_random(
        edges=[tuple(e) for e in pos_edges],  # 只传正样本边
        y=y,
        train_r=TRAIN_RATIO,
        val_r=VAL_RATIO,
        seed=SEED,
    )
    elif split_mode == "bfs":
        train_pos_idx, val_pos_idx, test_pos_idx = split_edges_bfs(
        edges=[tuple(e) for e in pos_edges],
        y=y,
        train_r=TRAIN_RATIO,
        val_r=VAL_RATIO,
        seed=SEED,
    )
    # 6. 负采样
    print(f"\n[5] 负采样 (比例 1:{NEGATIVE_RATIO}) ...")
    n_neg = int(num_pos * NEGATIVE_RATIO)
    neg_edges = generate_negative_edges(pos_edges, num_nodes, n_neg)
    print(f"    负样本边数: {len(neg_edges)}")

    # 合并正负边
    all_edges = np.array(pos_edges + neg_edges, dtype=np.int64)  # [E, 2]
    all_labels = np.vstack([y, np.zeros((len(neg_edges), NUM_CLASSES), dtype=np.float32)])
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(all_edges))
    all_edges = all_edges[perm]
    all_labels = all_labels[perm]

    # 7. 划分边
    print(f"\n[6] 按 {split_mode.upper()} 划分边 ...")
    if split_mode == "random":
      split_fn = split_edges_random
      train_idx, val_idx, test_idx = split_fn(
        edges=[tuple(e) for e in all_edges],
        y=all_labels,
        train_r=TRAIN_RATIO,
        val_r=VAL_RATIO,
        seed=SEED,
    )
    elif split_mode == "bfs":
      split_fn = split_edges_bfs
      train_idx, val_idx, test_idx = split_fn(
        edges=[tuple(e) for e in all_edges],
        y=all_labels,
        train_r=TRAIN_RATIO,
        val_r=VAL_RATIO,
        seed=SEED,
    )
    elif split_mode == "dfs":
      split_fn = split_edges_dfs
      train_idx, val_idx, test_idx = split_fn(
        edges=[tuple(e) for e in all_edges],
        y=all_labels,
        train_r=TRAIN_RATIO,
        val_r=VAL_RATIO,
        seed=SEED,
    )
    print(f"    Train: {len(train_idx)}  Test: {len(test_idx)}  Val: {len(val_idx)}")

    # 8. 统计每个 split 中正负比例
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        n_pos_split = int(all_labels[idx].sum(axis=1).any(axis=0) if False else
                          np.any(all_labels[idx] != 0, axis=1).sum())
        n_total = len(idx)
        print(f"    {name}: 正样本≈{n_pos_split}/{n_total} "
              f"({n_pos_split/n_total*100:.1f}%)")

    # 9. 构建 PyG Data 对象
    print("\n[7] 构建 PyG Data 对象 ...")
    edge_index = torch.from_numpy(all_edges.T).long()  # [2, E]
    edge_y     = torch.from_numpy(all_labels).float()  # [E, 7]

    # 构建邻接表（用于 GNN 消息传递）
    # 无向图：正反边都加
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    gnn_src = np.concatenate([src, dst])
    gnn_dst = np.concatenate([dst, src])
    gnn_edge_index = torch.from_numpy(np.stack([gnn_src, gnn_dst])).long()

    data = Data(
        x=torch.from_numpy(x).float(),
        edge_index=gnn_edge_index,
        edge_pairs=edge_index,       # 预测任务的边
        edge_y=edge_y,
        train_idx=torch.from_numpy(train_idx).long(),
        val_idx=torch.from_numpy(val_idx).long(),
        test_idx=torch.from_numpy(test_idx).long(),
    )

    # 10. 保存
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"SHS148k_{split_mode}.pt")
    torch.save({
        "data": data,
        "protein_ids": protein_ids,
        "num_nodes": num_nodes,
        "num_edges": len(all_edges),
        "num_pos": num_pos,
        "num_neg": len(neg_edges),
        "split_mode": split_mode,
        "class_names": CLASS_NAMES,
    }, save_path)
    print(f"\n✅ 预处理完成 → {save_path}")
    print(f"   节点特征: {data.x.shape}")
    print(f"   GNN 边:  {data.edge_index.shape}")
    print(f"   预测边:  {data.edge_pairs.shape[1]}")
    print(f"   标签:    {data.edge_y.shape}")

    # 保存划分元数据（JSON）
    meta = {
        "num_nodes": num_nodes,
        "num_edges": len(all_edges),
        "num_pos": num_pos,
        "num_neg": len(neg_edges),
        "split_mode": split_mode,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(out_dir, f"meta_{split_mode}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return data


# ───────────────────── CLI ─────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHS148k 数据预处理")
    parser.add_argument("--seq_path",  type=str, default=SEQ_FILE)
    parser.add_argument("--ppi_path",  type=str, default=PPI_FILE)
    parser.add_argument("--out_dir",   type=str, default=PROC_DIR)
    parser.add_argument("--split_mode",type=str, default=SPLIT_MODE,
                        choices=["random", "bfs", "dfs"])
    args = parser.parse_args()

    process(
        seq_path=args.seq_path,
        ppi_path=args.ppi_path,
        out_dir=args.out_dir,
        split_mode=args.split_mode,
    )
