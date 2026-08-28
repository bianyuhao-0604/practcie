"""
03_build_benchmark.py
构建 SHS27k / SHS148k 评测基准数据集。

SHS27k/SHS148k 文件格式（TSV）：
  列: protein1  protein2  action  score  ...
  - protein1/protein2: Ensembl Protein ID（如 ENSP00000xxxxxx）
  - action: 交互类型（reaction/binding/ptmod/activation/inhibition/catalysis/expression）
  - score: STRING combined score

数据划分策略（BFS / DFS）：
  - BFS: 广度优先，测试集蛋白与训练集有较多重叠
  - DFS: 深度优先，测试集蛋白与训练集重叠较少（更难）
  - 80% 训练 / 20% 测试
"""

import os
import json
import random
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import networkx as nx

# ============================================================
# 配置
# ============================================================
RAW_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw')
BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'benchmark')
os.makedirs(BENCHMARK_DIR, exist_ok=True)

RANDOM_SEED = 42
TEST_RATIO = 0.2

# 7 种交互类型
ACTION_TYPES = ['reaction', 'binding', 'ptmod', 'activation', 
                'inhibition', 'catalysis', 'expression']
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTION_TYPES)}


def load_shs_data(filepath: str) -> pd.DataFrame:
    """
    加载 SHS27k 或 SHS148k 数据。
    
    文件格式: TSV
    列: protein1, protein2, action, score, ...
    """
    df = pd.read_csv(filepath, sep='\t')
    
    # 标准化列名
    col_mapping = {}
    for col in df.columns:
      col_lower = col.lower().strip()
      if col_lower in ('protein1', 'protein_a', 'interactor_a', 'item_id_a'):
        col_mapping[col] = 'protein1'
      elif col_lower in ('protein2', 'protein_b', 'interactor_b', 'item_id_b'):
        col_mapping[col] = 'protein2'
      elif col_lower in ('action', 'interaction_type', 'type'):
        col_mapping[col] = 'action'
      elif col_lower in ('score', 'combined_score', 'confidence'):
        col_mapping[col] = 'score'

    df = df.rename(columns=col_mapping)
    
    # 确保必要列存在
    required_cols = ['protein1', 'protein2']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}。实际列: {list(df.columns)}")
    
    # 处理 action 列
    if 'action' not in df.columns:
        df['action'] = 'binding'  # 默认
    
    # 过滤有效的 action 类型
    df = df[df['action'].isin(ACTION_TYPES)]
    
    print(f"  加载完成: {len(df)} 条交互, "
          f"{df['protein1'].nunique() + df['protein2'].nunique()} 个蛋白（去重前）")
    
    return df


def build_protein_graph(df: pd.DataFrame) -> nx.Graph:
    """从交互数据构建蛋白图"""
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['protein1'], row['protein2'])
    return G


def bfs_split(df: pd.DataFrame, test_ratio: float = TEST_RATIO, seed: int = RANDOM_SEED):
    """
    BFS（广度优先搜索）数据划分策略。
    
    从随机种子节点出发，BFS 遍历图，将先访问到的蛋白归入训练集，
    后访问到的归入测试集。测试集蛋白与训练集有较多重叠。
    """
    G = build_protein_graph(df)
    nodes = list(G.nodes())
    random.seed(seed)
    
    # 随机选择种子节点
    seed_node = random.choice(nodes)
    
    # BFS 遍历
    visited = []
    queue = [seed_node]
    visited_set = {seed_node}
    
    while queue:
        next_queue = []
        for node in queue:
            visited.append(node)
            for neighbor in G.neighbors(node):
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    next_queue.append(neighbor)
        queue = next_queue
    
    # 按 BFS 顺序划分
    split_idx = int(len(visited) * (1 - test_ratio))
    train_proteins = set(visited[:split_idx])
    test_proteins = set(visited[split_idx:])
    
    # 划分交互
    train_edges = []
    test_edges = []
    
    for _, row in df.iterrows():
        p1, p2 = row['protein1'], row['protein2']
        if p1 in train_proteins and p2 in train_proteins:
            train_edges.append(row)
        elif p1 in test_proteins or p2 in test_proteins:
            test_edges.append(row)
        else:
            train_edges.append(row)  # 默认归入训练集
    
    return pd.DataFrame(train_edges), pd.DataFrame(test_edges)


def dfs_split(df: pd.DataFrame, test_ratio: float = TEST_RATIO, seed: int = RANDOM_SEED):
    """
    DFS（深度优先搜索）数据划分策略。
    
    从随机种子节点出发，DFS 遍历图，将先访问到的蛋白归入训练集，
    后访问到的归入测试集。测试集蛋白与训练集重叠较少（更难）。
    """
    G = build_protein_graph(df)
    nodes = list(G.nodes())
    random.seed(seed)
    
    # 随机选择种子节点
    seed_node = random.choice(nodes)
    
    # DFS 遍历
    visited = []
    visited_set = set()
    stack = [seed_node]
    
    while stack:
        node = stack.pop()
        if node in visited_set:
            continue
        visited_set.add(node)
        visited.append(node)
        
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited_set:
                stack.append(neighbor)
    
    # 按 DFS 顺序划分
    split_idx = int(len(visited) * (1 - test_ratio))
    train_proteins = set(visited[:split_idx])
    test_proteins = set(visited[split_idx:])
    
    # 划分交互
    train_edges = []
    test_edges = []
    
    for _, row in df.iterrows():
        p1, p2 = row['protein1'], row['protein2']
        if p1 in train_proteins and p2 in train_proteins:
            train_edges.append(row)
        elif p1 in test_proteins or p2 in test_proteins:
            test_edges.append(row)
        else:
            train_edges.append(row)
    
    return pd.DataFrame(train_edges), pd.DataFrame(test_edges)


def build_protein_id_mapping(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """构建蛋白 ID 到索引的映射（训练集和测试集共享）"""
    all_proteins = set()
    for df in [train_df, test_df]:
        all_proteins.update(df['protein1'].unique())
        all_proteins.update(df['protein2'].unique())
    
    sorted_proteins = sorted(all_proteins)
    return {p: idx for idx, p in enumerate(sorted_proteins)}


def dataframe_to_samples(df: pd.DataFrame, node_to_idx: dict) -> dict:
    """
    将 DataFrame 转换为模型可用的样本格式。
    
    Returns:
        {
            'protein1_idx': tensor [N],
            'protein2_idx': tensor [N],
            'action_labels': tensor [N, 7] (multi-hot),
            'binary_labels': tensor [N] (是否相互作用),
        }
    """
    p1_indices = []
    p2_indices = []
    action_labels = []
    
    for _, row in df.iterrows():
        p1 = row['protein1']
        p2 = row['protein2']
        action = row['action']
        
        if p1 not in node_to_idx or p2 not in node_to_idx:
            continue
        
        p1_indices.append(node_to_idx[p1])
        p2_indices.append(node_to_idx[p2])
        
        # 多标签编码
        label_vec = [0] * len(ACTION_TYPES)
        if action in ACTION_TO_IDX:
            label_vec[ACTION_TO_IDX[action]] = 1
        action_labels.append(label_vec)
    
    return {
        'protein1_idx': torch.tensor(p1_indices, dtype=torch.long),
        'protein2_idx': torch.tensor(p2_indices, dtype=torch.long),
        'action_labels': torch.tensor(action_labels, dtype=torch.float),
        'binary_labels': torch.ones(len(p1_indices), dtype=torch.long),  # 全部为正样本
    }


def process_dataset(dataset_name: str, filepath: str):
    """处理单个数据集"""
    print(f"\n{'=' * 60}")
    print(f"处理 {dataset_name}")
    print(f"{'=' * 60}")
    
    # 加载数据
    print(f"\n[加载] {filepath}")
    df = load_shs_data(filepath)
    
    # 统计交互类型分布
    print(f"\n[统计] 交互类型分布:")
    for action in ACTION_TYPES:
        count = len(df[df['action'] == action])
        pct = count / len(df) * 100
        print(f"  {action:12s}: {count:6d} ({pct:5.2f}%)")
    
    # BFS 划分
    print(f"\n[划分] BFS 策略 (test_ratio={TEST_RATIO})")
    train_bfs, test_bfs = bfs_split(df)
    print(f"  训练集: {len(train_bfs)} 条交互")
    print(f"  测试集: {len(test_bfs)} 条交互")
    
    # DFS 划分
    print(f"\n[划分] DFS 策略 (test_ratio={TEST_RATIO})")
    train_dfs, test_dfs = dfs_split(df)
    print(f"  训练集: {len(train_dfs)} 条交互")
    print(f"  测试集: {len(test_dfs)} 条交互")
    
    # 构建蛋白 ID 映射
    node_to_idx = build_protein_id_mapping(train_bfs, test_bfs)
    print(f"\n[映射] 蛋白总数: {len(node_to_idx)}")
    
    # 转换为样本格式
    for split_name, train_df, test_df in [
        ("BFS", train_bfs, test_bfs),
        ("DFS", train_dfs, test_dfs),
    ]:
        split_dir = os.path.join(BENCHMARK_DIR, f"{dataset_name}_{split_name}")
        os.makedirs(split_dir, exist_ok=True)
        
        train_samples = dataframe_to_samples(train_df, node_to_idx)
        test_samples = dataframe_to_samples(test_df, node_to_idx)
        
        torch.save(train_samples, os.path.join(split_dir, "train.pt"))
        torch.save(test_samples, os.path.join(split_dir, "test.pt"))
        
        # 保存元数据
        metadata = {
            "dataset": dataset_name,
            "split_strategy": split_name,
            "num_proteins": len(node_to_idx),
            "num_train_interactions": len(train_samples['protein1_idx']),
            "num_test_interactions": len(test_samples['protein1_idx']),
            "action_types": ACTION_TYPES,
        }
        with open(os.path.join(split_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  [保存] {split_dir}")
        print(f"    train.pt: {len(train_samples['protein1_idx'])} samples")
        print(f"    test.pt:  {len(test_samples['protein1_idx'])} samples")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    print("L3-PPI 评测基准构建")
    
    # 处理 SHS27k
    shs27k_file = os.path.join(RAW_DIR, "protein.actions.SHS27k.STRING.pro2.txt")
    if os.path.exists(shs27k_file):
        process_dataset("SHS27k", shs27k_file)
    else:
        print(f"\n[跳过] SHS27k 文件不存在: {shs27k_file}")
    
    # 处理 SHS148k
    shs148k_file = os.path.join(RAW_DIR, "protein.actions.SHS148k.STRING.txt")
    if os.path.exists(shs148k_file):
        process_dataset("SHS148k", shs148k_file)
    else:
        print(f"\n[跳过] SHS148k 文件不存在: {shs148k_file}")
    
    print("\n" + "=" * 60)
    print("评测基准构建完成！")
    print(f"输出目录: {BENCHMARK_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()