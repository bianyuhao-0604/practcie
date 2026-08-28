"""
02_build_pretrain_samples.py
从 HI-II-14 / HuRI 高质量二元互作网络中提取 L3 路径，
构造 GNNpre/GIN 预训练的正负样本。

HI-II-14 文件格式（TSV，无表头或带表头）：
  列: Protein_A  Protein_B  (Ensembl Gene ID 或 Gene Symbol)

L3 路径定义：
  u → w1 → w2 → v（4 个节点，3 条边，长度恰好为 3）
  
正样本：u 和 v 在 PPI 网络中存在直接相互作用
负样本：u 和 v 在 PPI 网络中不存在直接相互作用
"""

import os
import json
import random
import networkx as nx
import torch
import numpy as np
from collections import defaultdict
from itertools import combinations

# ============================================================
# 配置
# ============================================================
RAW_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw')
PRETRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pretrain')
os.makedirs(PRETRAIN_DIR, exist_ok=True)

RANDOM_SEED = 42
MAX_L3_PATHS = 200000       # 每种样本最大数量
NEGATIVE_RATIO = 1.0        # 负样本与正样本的比例


def load_ppi_network(filepath: str) -> nx.Graph:
    """
    加载 PPI 网络（TSV 格式）。
    
    HI-II-14 文件格式：
      - 制表符分隔
      - 前两列为相互作用的蛋白对（Ensembl Gene ID 或 Gene Symbol）
      - 可能有表头行（以 # 开头或包含非蛋白 ID 的文本）
    
    Returns:
        G: 无向图，节点为蛋白 ID，边为二元相互作用
    """
    G = nx.Graph()
    edge_count = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            cols = line.split('\t')
            if len(cols) < 2:
                continue
            
            protein_a = cols[0].strip()
            protein_b = cols[1].strip()
            
            # 跳过表头行
            if protein_a.lower() in ('protein_a', 'protein1', 'interactor_a'):
                continue
            
            # 跳过自环
            if protein_a == protein_b:
                continue
            
            G.add_edge(protein_a, protein_b)
            edge_count += 1
    
    print(f"  加载完成: {G.number_of_nodes()} 个蛋白, {G.number_of_edges()} 条边")
    return G


def extract_L3_paths(G: nx.Graph, positive_edges: set, max_paths: int = MAX_L3_PATHS):
    """
    从 PPI 网络中提取所有 L3 路径，并标记正负样本。
    
    L3 路径: u → w1 → w2 → v（4 个不同节点，3 条边）
    
    正样本: (u, v) 在 positive_edges 中（存在直接相互作用）
    负样本: (u, v) 不在 positive_edges 中（不存在直接相互作用）
    
    Args:
        G: PPI 网络图
        positive_edges: 已知相互作用的蛋白对集合
        max_paths: 每种样本的最大数量
    
    Returns:
        positive_paths: 正样本 L3 路径列表
        negative_paths: 负样本 L3 路径列表
    """
    positive_paths = []
    negative_paths = []
    
    nodes = list(G.nodes())
    total_pairs = 0
    sampled = 0
    
    print(f"  开始提取 L3 路径（最大 {max_paths} 条/类）...")
    
    for u in nodes:
        if len(positive_paths) >= max_paths and len(negative_paths) >= max_paths:
            break
        
        u_neighbors = list(G.neighbors(u))
        
        for w1 in u_neighbors:
            if len(positive_paths) >= max_paths and len(negative_paths) >= max_paths:
                break
            
            w1_neighbors = list(G.neighbors(w1))
            
            for w2 in w1_neighbors:
                if w2 == u or w2 == w1:
                    continue
                
                w2_neighbors = list(G.neighbors(w2))
                
                for v in w2_neighbors:
                    if v == u or v == w1 or v == w2:
                        continue
                    
                    total_pairs += 1
                    
                    # 规范化蛋白对（排序后比较）
                    pair = tuple(sorted([u, v]))
                    
                    path = [u, w1, w2, v]
                    
                    if pair in positive_edges:
                        if len(positive_paths) < max_paths:
                            positive_paths.append(path)
                    else:
                        if len(negative_paths) < max_paths:
                            negative_paths.append(path)
    
    print(f"  扫描了 {total_pairs} 个蛋白对")
    print(f"  正样本 L3 路径: {len(positive_paths)}")
    print(f"  负样本 L3 路径: {len(negative_paths)}")
    
    return positive_paths, negative_paths


def build_protein_id_mapping(G: nx.Graph) -> dict:
    """
    构建蛋白 ID 到连续整数索引的映射。
    """
    nodes = sorted(G.nodes())
    return {node: idx for idx, node in enumerate(nodes)}


def paths_to_tensors(paths: list, node_to_idx: dict) -> torch.Tensor:
    """
    将 L3 路径列表转换为张量。
    
    Args:
        paths: [[u, w1, w2, v], ...]
        node_to_idx: 蛋白 ID -> 索引映射
    
    Returns:
        tensor: [num_paths, 4] 的整数张量
    """
    indices = []
    for path in paths:
        idx_path = [node_to_idx[node] for node in path]
        indices.append(idx_path)
    return torch.tensor(indices, dtype=torch.long)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    print("=" * 60)
    print("L3-PPI 预训练样本构建")
    print("=" * 60)
    
    # ---- 步骤 1：加载 PPI 网络 ----
    print("\n[步骤 1] 加载 PPI 网络")
    
    # 优先使用 HI-II-14（最高质量的二元互作数据）
    huri_file = os.path.join(RAW_DIR, "HI-II-14.tsv")
    if not os.path.exists(huri_file):
        print(f"错误: 未找到 {huri_file}")
        print("请先运行 01_download_data.py 下载数据")
        return
    
    G = load_ppi_network(huri_file)
    
    # 获取所有已知相互作用的蛋白对（用于区分正负样本）
    positive_edges = set()
    for u, v in G.edges():
        pair = tuple(sorted([u, v]))
        positive_edges.add(pair)
    
    print(f"  已知相互作用蛋白对: {len(positive_edges)}")
    
    # ---- 步骤 2：提取 L3 路径 ----
    print("\n[步骤 2] 提取 L3 路径并标记正负样本")
    positive_paths, negative_paths = extract_L3_paths(G, positive_edges)
    
    # 平衡正负样本
    min_count = min(len(positive_paths), len(negative_paths))
    if NEGATIVE_RATIO == 1.0:
        positive_paths = positive_paths[:min_count]
        negative_paths = negative_paths[:min_count]
    else:
        neg_count = int(len(positive_paths) * NEGATIVE_RATIO)
        negative_paths = negative_paths[:neg_count]
    
    print(f"\n  最终正样本: {len(positive_paths)}")
    print(f"  最终负样本: {len(negative_paths)}")
    
    # ---- 步骤 3：构建蛋白 ID 映射 ----
    print("\n[步骤 3] 构建蛋白 ID 映射")
    node_to_idx = build_protein_id_mapping(G)
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    print(f"  蛋白总数: {len(node_to_idx)}")
    
    # ---- 步骤 4：转换为张量 ----
    print("\n[步骤 4] 转换为张量")
    pos_tensor = paths_to_tensors(positive_paths, node_to_idx)
    neg_tensor = paths_to_tensors(negative_paths, node_to_idx)
    
    print(f"  正样本张量形状: {pos_tensor.shape}")
    print(f"  负样本张量形状: {neg_tensor.shape}")
    
    # ---- 步骤 5：保存 ----
    print("\n[步骤 5] 保存预训练样本")
    
    torch.save(pos_tensor, os.path.join(PRETRAIN_DIR, "positive_L3_paths.pt"))
    torch.save(neg_tensor, os.path.join(PRETRAIN_DIR, "negative_L3_paths.pt"))
    
    # 保存元数据
    metadata = {
        "source": "HI-II-14",
        "num_proteins": len(node_to_idx),
        "num_edges": G.number_of_edges(),
        "num_positive_paths": len(positive_paths),
        "num_negative_paths": len(negative_paths),
        "path_length": 3,
        "num_nodes_per_path": 4,
    }
    
    with open(os.path.join(PRETRAIN_DIR, "pretrain_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 保存蛋白 ID 映射
    torch.save({
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
    }, os.path.join(PRETRAIN_DIR, "protein_id_mapping.pt"))
    
    print(f"\n  保存目录: {PRETRAIN_DIR}")
    print("  文件列表:")
    for f in os.listdir(PRETRAIN_DIR):
        fpath = os.path.join(PRETRAIN_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    {f} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("预训练样本构建完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()