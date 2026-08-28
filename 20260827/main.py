#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3-PPI 完整实现（单文件版）
============================
包含：数据下载、预处理、ESM2嵌入提取、GNNpre预训练、L3-PPI微调、评测

用法:
  python main.py --steps all
  python main.py --steps download preprocess
  python main.py --steps pretrain finetune evaluate --dataset SHS27k --split BFS
"""

import os
import sys
import json
import gzip
import random
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
import networkx as nx
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# PyG imports
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import GCNConv, GINConv, global_add_pool
    from torch_geometric.utils import to_dense_batch
except ImportError:
    print("[ERROR] 请安装 torch-geometric: pip install torch-geometric")
    sys.exit(1)


# ============================================================
# 工具函数
# ============================================================

def setup_logger(log_dir: str, name: str = "l3ppi") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"))
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    d = cfg.get("training", {}).get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
    preds = (scores >= threshold).astype(int)
    m = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }
    if len(np.unique(labels)) > 1:
        m['auc'] = roc_auc_score(labels, scores)
        m['aupr'] = average_precision_score(labels, scores)
    else:
        m['auc'] = m['aupr'] = 0.0
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    m.update({'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)})
    return m


def fmt_metrics(m: dict) -> str:
    return f"Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}"


# ============================================================
# STEP 1: 数据下载
# ============================================================

def download_file(url: str, save_path: str, desc: str = "") -> bool:
    if os.path.exists(save_path):
        print(f"  [跳过] {desc}: {save_path}")
        return True
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=f"下载 {desc}") as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [失败] {desc}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def step_download(raw_dir: str):
    print("\n" + "=" * 60)
    print("STEP 1: 数据下载")
    print("=" * 60)
    os.makedirs(raw_dir, exist_ok=True)

    files = {
        "HI-II-14.tsv": "http://interactome.dfci.harvard.edu/H_sapiens/download/HI-II-14.tsv",
        "HuRI.tsv": "http://interactome.dfci.harvard.edu/H_sapiens/download/HuRI.tsv",
        "protein.actions.SHS27k.STRING.pro2.txt": "https://zenodo.org/records/7213401/files/protein.actions.SHS27k.STRING.pro2.txt",
        "protein.actions.SHS148k.STRING.txt": "https://zenodo.org/records/7213401/files/protein.actions.SHS148k.STRING.txt",
        "protein.SHS27k.sequences.dictionary.pro3.tsv": "https://zenodo.org/records/7213401/files/protein.SHS27k.sequences.dictionary.pro3.tsv",
        "protein.SHS148k.sequences.dictionary.tsv": "https://zenodo.org/records/7213401/files/protein.SHS148k.sequences.dictionary.tsv",
    }

    results = {}
    for fname, url in files.items():
        results[fname] = download_file(url, os.path.join(raw_dir, fname), desc=fname)

    ok = sum(results.values())
    print(f"\n下载完成: {ok}/{len(results)}")
    return results


# ============================================================
# STEP 2: 数据预处理
# ============================================================

ACTION_TYPES = ['reaction', 'binding', 'ptmod', 'activation', 'inhibition', 'catalysis', 'expression']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}


def load_ppi_graph(filepath: str) -> nx.Graph:
    G = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split('\t')
            if len(cols) < 2:
                continue
            a, b = cols[0].strip(), cols[1].strip()
            if a.lower() in ('protein_a', 'protein1', 'interactor_a', 'official_interactor_a'):
                continue
            if a != b:
                G.add_edge(a, b)
    print(f"  图: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def extract_l3_paths(G: nx.Graph, pos_edges: Set[Tuple], max_paths: int = 200000):
    pos_paths, neg_paths = [], []
    nodes = list(G.nodes())
    scanned = 0
    for u in nodes:
        if len(pos_paths) >= max_paths and len(neg_paths) >= max_paths:
            break
        for w1 in G.neighbors(u):
            if len(pos_paths) >= max_paths and len(neg_paths) >= max_paths:
                break
            for w2 in G.neighbors(w1):
                if w2 in (u, w1):
                    continue
                for v in G.neighbors(w2):
                    if v in (u, w1, w2):
                        continue
                    scanned += 1
                    pair = tuple(sorted([u, v]))
                    path = [u, w1, w2, v]
                    if pair in pos_edges:
                        if len(pos_paths) < max_paths:
                            pos_paths.append(path)
                    else:
                        if len(neg_paths) < max_paths:
                            neg_paths.append(path)
    print(f"  扫描 {scanned} 条候选, 正={len(pos_paths)}, 负={len(neg_paths)}")
    return pos_paths, neg_paths
def convert(d, pn2i):
    p1, p2, al = [], [], []
    for _, row in d.iterrows():
        if row['protein1'] in pn2i and row['protein2'] in pn2i:
            p1.append(pn2i[row['protein1']])
            p2.append(pn2i[row['protein2']])
            lv = [0] * len(ACTION_TYPES)
            if row['action'] in ACTION_TO_IDX:
                lv[ACTION_TO_IDX[row['action']]] = 1
            al.append(lv)
    return {
        'protein1_idx': torch.tensor(p1, dtype=torch.long),
        'protein2_idx': torch.tensor(p2, dtype=torch.long),
        'action_labels': torch.tensor(al, dtype=torch.float),
        'binary_labels': torch.ones(len(p1), dtype=torch.long),
    }


def generate_negative_samples(df, all_prots, pn2i, pos_pairs, num_neg):
    neg_p1, neg_p2, neg_al = [], [], []
    prots = list(all_prots)
    random.seed(42)  # 可改为传入 seed
    while len(neg_p1) < num_neg:
        a, b = random.sample(prots, 2)
        if (a, b) not in pos_pairs and (b, a) not in pos_pairs:
            if a in pn2i and b in pn2i:
                neg_p1.append(pn2i[a])
                neg_p2.append(pn2i[b])
                lv = [0] * len(ACTION_TYPES)
                neg_al.append(lv)
    return neg_p1, neg_p2, neg_al


def save_benchmark_data(train_df, test_df, pn2i, all_prots, out_dir):
    # 训练集
    pos_train = convert(train_df, pn2i)
    pos_pairs_train = set(zip(train_df['protein1'], train_df['protein2']))
    neg_train = generate_negative_samples(train_df, all_prots, pn2i,
                                          pos_pairs_train, len(pos_train['protein1_idx']))
    train_data = {
        'protein1_idx': torch.cat([pos_train['protein1_idx'],
                                   torch.tensor(neg_train[0], dtype=torch.long)]),
        'protein2_idx': torch.cat([pos_train['protein2_idx'],
                                   torch.tensor(neg_train[1], dtype=torch.long)]),
        'binary_labels': torch.cat([torch.ones(len(pos_train['protein1_idx'])),
                                    torch.zeros(len(neg_train[0]))]).long(),
        'action_labels': torch.cat([pos_train['action_labels'],
                                    torch.tensor(neg_train[2], dtype=torch.float)]),
    }
    torch.save(train_data, os.path.join(out_dir, "train.pt"))

    # 测试集
    pos_test = convert(test_df, pn2i)
    pos_pairs_test = set(zip(test_df['protein1'], test_df['protein2']))
    neg_test = generate_negative_samples(test_df, all_prots, pn2i,
                                         pos_pairs_test, len(pos_test['protein1_idx']))
    test_data = {
        'protein1_idx': torch.cat([pos_test['protein1_idx'],
                                   torch.tensor(neg_test[0], dtype=torch.long)]),
        'protein2_idx': torch.cat([pos_test['protein2_idx'],
                                   torch.tensor(neg_test[1], dtype=torch.long)]),
        'binary_labels': torch.cat([torch.ones(len(pos_test['protein1_idx'])),
                                    torch.zeros(len(neg_test[0]))]).long(),
        'action_labels': torch.cat([pos_test['action_labels'],
                                    torch.tensor(neg_test[2], dtype=torch.float)]),
    }
    torch.save(test_data, os.path.join(out_dir, "test.pt"))

    # 保存蛋白质映射（可选）
    torch.save({"node_to_idx": pn2i}, os.path.join(out_dir, "protein_id_mapping.pt"))

def step_preprocess(raw_dir: str, pretrain_dir: str, benchmark_dir: str, seed: int = 42):
    print("\n" + "=" * 60)
    print("STEP 2: 数据预处理")
    print("=" * 60)
    set_seed(seed)
    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(benchmark_dir, exist_ok=True)

    # --- 2a: L3 路径提取 ---
    huri = os.path.join(raw_dir, "HI-II-14.tsv")
    if not os.path.exists(huri):
        huri = os.path.join(raw_dir, "HuRI.tsv")
    if not os.path.exists(huri):
        print("[ERROR] 未找到 HI-II-14.tsv 或 HuRI.tsv，请先运行 download")
        return

    print(f"\n[2a] 从 {os.path.basename(huri)} 提取 L3 路径")
    G = load_ppi_graph(huri)
    pos_edges = {tuple(sorted(e)) for e in G.edges()}
    pos_paths, neg_paths = extract_l3_paths(G, pos_edges)

    mn = min(len(pos_paths), len(neg_paths))
    pos_paths, neg_paths = pos_paths[:mn], neg_paths[:mn]

    nodes = sorted(G.nodes())
    n2i = {n: i for i, n in enumerate(nodes)}
    i2n = {i: n for n, i in n2i.items()}

    def to_tensor(paths):
        return torch.tensor([[n2i[n] for n in p] for p in paths], dtype=torch.long)

    torch.save(to_tensor(pos_paths), os.path.join(pretrain_dir, "positive_L3_paths.pt"))
    torch.save(to_tensor(neg_paths), os.path.join(pretrain_dir, "negative_L3_paths.pt"))
    torch.save({"node_to_idx": n2i, "idx_to_node": i2n}, os.path.join(pretrain_dir, "protein_id_mapping.pt"))
    json.dump({"num_proteins": len(n2i), "num_pos": len(pos_paths), "num_neg": len(neg_paths)},
              open(os.path.join(pretrain_dir, "metadata.json"), 'w'), indent=2)
    print(f"  保存至 {pretrain_dir}")

    # --- 2b: 评测基准构建 ---
    print(f"\n[2b] 构建评测基准")
    shs_files = {
        "SHS27k": "protein.actions.SHS27k.STRING.pro2.txt",
        "SHS148k": "protein.actions.SHS148k.STRING.txt",
    }

    for ds_name, fname in shs_files.items():
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [跳过] {ds_name}: {fpath} 不存在")
            continue

        df = pd.read_csv(fpath, sep='\t')
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ('item_id_a','protein1', 'protein_a'): col_map[c] = 'protein1'
            elif cl in ('item_id_b','protein2', 'protein_b'): col_map[c] = 'protein2'
            elif cl in ('action', 'action', 'interaction_type'): col_map[c] = 'action'
        df = df.rename(columns=col_map)
        if 'action' not in df.columns:
            df['action'] = 'binding'
        df = df[df['action'].isin(ACTION_TYPES)]

        all_prots = sorted(set(df['protein1']) | set(df['protein2']))
        pn2i = {p: i for i, p in enumerate(all_prots)}

        for strat, split_fn in [("BFS", _bfs_split), ("DFS", _dfs_split)]:
            train_df, test_df = split_fn(df, seed=seed)
            out = os.path.join(benchmark_dir, f"{ds_name}_{strat}")
            os.makedirs(out, exist_ok=True)
            # 调用新函数保存正负样本
            save_benchmark_data(train_df, test_df, pn2i, all_prots, out)
            print(f"  {ds_name}_{strat}: train={len(train_df)}正, test={len(test_df)}正")



def _bfs_split(df, test_ratio=0.2, seed=42):
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r['protein1'], r['protein2'])
    random.seed(seed)
    start = random.choice(list(G.nodes()))
    visited, vis_set, q = [], {start}, [start]
    while q:
        nq = []
        for n in q:
            visited.append(n)
            for nb in G.neighbors(n):
                if nb not in vis_set:
                    vis_set.add(nb)
                    nq.append(nb)
        q = nq
    sp = int(len(visited) * (1 - test_ratio))
    tr, te = set(visited[:sp]), set(visited[sp:])
    train = df[(df['protein1'].isin(tr)) & (df['protein2'].isin(tr))]
    test = df[~((df['protein1'].isin(tr)) & (df['protein2'].isin(tr)))]
    return train, test

def _dfs_split(df, test_ratio=0.2, seed=42):
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r['protein1'], r['protein2'])
    random.seed(seed)
    start = random.choice(list(G.nodes()))
    visited, vis_set, stack = [], set(), [start]
    while stack:
        n = stack.pop()
        if n in vis_set:
            continue
        vis_set.add(n)
        visited.append(n)
        nbs = list(G.neighbors(n))
        random.shuffle(nbs)
        for nb in nbs:
            if nb not in vis_set:
                stack.append(nb)
    sp = int(len(visited) * (1 - test_ratio))
    tr, te = set(visited[:sp]), set(visited[sp:])
    train = df[(df['protein1'].isin(tr)) & (df['protein2'].isin(tr))]
    test = df[~((df['protein1'].isin(tr)) & (df['protein2'].isin(tr)))]
    return train, test


# ============================================================
# STEP 3: ESM2 嵌入提取
# ============================================================

def step_embeddings(raw_dir: str, emb_dir: str, pretrain_dir:str,model_name: str = "facebook/esm2_t33_650M_UR50D", batch_size: int = 16):
    print("\n" + "=" * 60)
    print("STEP 3: ESM2 嵌入提取")
    print("=" * 60)
    os.makedirs(emb_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}, 模型: {model_name}")

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("[ERROR] 请安装 transformers: pip install transformers")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    seq_files = {
        "SHS27k": ("protein.SHS27k.sequences.dictionary.pro3.tsv", "esm2_650m_SHS27k.pt"),
        "SHS148k": ("protein.SHS148k.sequences.dictionary.tsv", "esm2_650m_SHS148k.pt"),
    }

    for ds, (sf, ef) in seq_files.items():
        sp = os.path.join(raw_dir, sf)
        ep = os.path.join(emb_dir, ef)
        if not os.path.exists(sp):
            print(f"  [跳过] {ds}: {sp}")
            continue
        if os.path.exists(ep):
            print(f"  [跳过] {ds}: {ep} 已存在")
            continue

        df = pd.read_csv(sp, sep='\t')
        seq_col = next((c for c in df.columns if 'seq' in c.lower()), df.columns[-1])
        id_col = next((c for c in df.columns if 'protein' in c.lower() or 'id' in c.lower()), df.columns[0])
        seqs = {str(r[id_col]).strip(): str(r[seq_col]).strip() for _, r in df.iterrows() if str(r[seq_col]).strip() != 'nan'}
        print(f"  {ds}: {len(seqs)} 条序列")

        embs = {}
        ids = list(seqs.keys())
        for bs in tqdm(range(0, len(ids), batch_size), desc=f"ESM2 {ds}"):
            batch_ids = ids[bs:bs+batch_size]
            batch_seqs = [seqs[k][:1022] for k in batch_ids]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]
            for i, k in enumerate(batch_ids):
                embs[k] = cls[i].cpu()

        torch.save(embs, ep)
        print(f"  保存: {ep} ({len(embs)} embeddings)")
     # 新增：为预训练数据集（HI-II-14）生成嵌入
    print("\n  处理预训练数据集 (HI-II-14)...")
    pretrain_mapping_path = os.path.join(pretrain_dir, "protein_id_mapping.pt")
    if not os.path.exists(pretrain_mapping_path):
        print("  [警告] 未找到预训练蛋白质映射文件，跳过预训练嵌入生成")
        return

    # 加载预训练蛋白质 ID
    pretrain_mapping = torch.load(pretrain_mapping_path)
    idx_to_node = pretrain_mapping["idx_to_node"]
    needed_ids = set(idx_to_node.values())
    print(f"  预训练所需蛋白质数量: {len(needed_ids)}")

    # 预训练序列文件（需要根据实际情况修改文件名）
    pretrain_seq_file = os.path.join(raw_dir, "protein.HI-II-14_sequences.tsv")
    if not os.path.exists(pretrain_seq_file):
        print(f"  [错误] 预训练序列文件不存在: {pretrain_seq_file}")
        print("  请先获取 HI-II-14 的蛋白质序列文件。")
        return

    # 读取序列文件
    df = pd.read_csv(pretrain_seq_file, sep='\t', header=None, names=['protein', 'sequence'])
    seq_col = 'sequence'
    id_col = 'protein'
    all_seqs = {str(r[id_col]).strip(): str(r[seq_col]).strip() for _, r in df.iterrows()
                if str(r[seq_col]).strip() != 'nan'}

    # 只保留预训练需要的蛋白质
    seqs = {k: v for k, v in all_seqs.items() if k in needed_ids}
    print(f"  序列覆盖: {len(seqs)}/{len(needed_ids)}")

    if len(seqs) == 0:
        print("  [错误] 没有任何蛋白质的序列，无法生成预训练嵌入")
        return

    # 生成嵌入
    pretrain_emb_path = os.path.join(emb_dir, "esm2_650m_pretrain.pt")
    if os.path.exists(pretrain_emb_path):
        print(f"  [跳过] 预训练嵌入已存在: {pretrain_emb_path}")
        return

    embs = {}
    ids = list(seqs.keys())
    for bs in tqdm(range(0, len(ids), batch_size), desc="ESM2 pretrain"):
        batch_ids = ids[bs:bs+batch_size]
        batch_seqs = [seqs[k][:1022] for k in batch_ids]  # ESM2 最大长度限制
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]
        for i, k in enumerate(batch_ids):
            embs[k] = cls[i].cpu()

    torch.save(embs, pretrain_emb_path)
    print(f"  保存: {pretrain_emb_path} ({len(embs)} embeddings)")


# ============================================================
# 模型定义
# ============================================================

class PromptEmbeddings(nn.Module):
    def __init__(self, d_model: int, d_prompt: int, K: int):
        super().__init__()
        self.K = K
        self.proj = nn.Identity()
        self.virtual = nn.Parameter(torch.randn(K + 1, d_model) * 0.01)

    def forward(self, eu: torch.Tensor, ev: torch.Tensor) -> Batch:
        B = eu.size(0)
        K = self.K
        dev = eu.device
        xu = self.proj(eu).unsqueeze(1)
        xv = self.proj(ev).unsqueeze(1)
        xvp = self.virtual.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([xu, xv, xvp], dim=1)  # [B, K+3, d]
        el = []
        for i in range(1, K + 1):
            el.extend([[0, i+2], [i+2, 2], [i+2, 1]])
        ei = torch.tensor(el, dtype=torch.long, device=dev).t().contiguous()
        return Batch.from_data_list([Data(x=x[b], edge_index=ei.clone()) for b in range(B)])


class GNNgpt(nn.Module):
    def __init__(self, d_in: int, d_hid: int, n_layers: int, K: int, temp: float = 0.5):
        super().__init__()
        self.K, self.temp = K, temp
        self.layers = nn.ModuleList()
        d = d_in
        for _ in range(n_layers):
            self.layers.append(GCNConv(d, d_hid))
            d = d_hid
        self.gate = nn.Linear(d_hid * 3, 1)

    def forward(self, bd: Batch, training: bool = True):
        h = bd.x
        for l in self.layers:
            h = F.relu(l(h, bd.edge_index))
        hd, _ = to_dense_batch(h, bd.batch)
        hu, hv0 = hd[:, 0], hd[:, 2]
        Z = torch.stack([torch.cat([hu, hd[:, i+2], hv0], dim=-1) for i in range(1, self.K+1)], dim=1)
        logits = self.gate(Z).squeeze(-1)
        probs = torch.sigmoid(logits)
        gv = probs

        # mask edges
        K = self.K
        dev = bd.edge_index.device
        nepg = 3 * K
        e2p = torch.tensor([i//3 for i in range(nepg)], dtype=torch.long, device=dev)
        teg = bd.edge_index.size(1)
        egi = torch.arange(teg, device=dev) // nepg
        epi = e2p.repeat(teg // nepg) if teg >= nepg else e2p[:teg]
        eg = gv[egi, epi]
        gei = bd.edge_index[:, eg.bool()]
        return Data(x=bd.x, edge_index=gei, batch=bd.batch), gv, probs


class GNNpre(nn.Module):
    def __init__(self, d_in: int, d_hid: int, n_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_hid)

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        d = d_hid
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(d, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid))
            self.layers.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(d_hid))
        self.out = nn.Linear(d_hid, 1)

    def forward(self, bd):
        h = self.input_proj(bd.x)
        bd = Data(x=h, edge_index=bd.edge_index, batch=bd.batch)
        for l, bn in zip(self.layers, self.bns):
            h = F.relu(bn(l(h, bd.edge_index)))
        hg = global_add_pool(h, bd.batch)
        return torch.sigmoid(self.out(hg))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.train()


class L3PPIClassificationHead(nn.Module):
    def __init__(self, d_model=1280, d_prompt=64, d_gpt=64, d_gin=64,
                 gpt_layers=2, gin_layers=2, K=4, temp=0.5, gamma=2.0):
        super().__init__()
        self.K, self.gamma = K, gamma
        self.prompt = PromptEmbeddings(d_model, d_prompt, K)
        self.gnn_gpt = GNNgpt(d_prompt, d_gpt, gpt_layers, K, temp)
        self.gnn_pre = GNNpre(d_model, d_gin, gin_layers)

    def forward(self, eu, ev, training=True):
        g = self.prompt(eu, ev)
        gp, gv, pp = self.gnn_gpt(g, training)
        yp = self.gnn_pre(gp)
        return yp, gv, pp

    def loss(self, yp, gv, pp, y):
        lbce = F.binary_cross_entropy(yp.squeeze(-1), y.float())
        K, g = self.K, self.gamma
        pm = (y == 1).float()
        nm = (y == 0).float()
        ps = pp.sum(dim=1)
        lp = F.relu(K*(1-1/g) - ps) * pm + F.relu(ps - K/g) * nm
        llpn = lp.sum() / y.size(0)
        lambda_path = 0.01
        return lbce + lambda_path * llpn, lbce, llpn

    def freeze_gnn_pre(self):
        self.gnn_pre.freeze()

    @torch.no_grad()
    def predict(self, eu, ev):
        self.eval()
        yp, gv, pp = self(eu, ev, training=False)
        pred = (yp.squeeze(-1) > 0.5).long()
        ap = [torch.where(gv[b] > 0.5)[0].tolist() for b in range(gv.size(0))]
        return pred, yp, ap


# ============================================================
# 数据集
# ============================================================

class PretrainDS(Dataset):
    def __init__(self, pretrain_dir, emb_path, d_model=1280):
        pp = torch.load(os.path.join(pretrain_dir, "positive_L3_paths.pt"))
        np_ = torch.load(os.path.join(pretrain_dir, "negative_L3_paths.pt"))
        self.paths = torch.cat([pp, np_], dim=0)
        self.labels = torch.cat([torch.ones(pp.size(0)), torch.zeros(np_.size(0))])
        self.embs = torch.load(emb_path)
        mp = os.path.join(pretrain_dir, "protein_id_mapping.pt")
        self.i2n = torch.load(mp).get("idx_to_node") if os.path.exists(mp) else None
        self.d = d_model
        print(f"  [PretrainDS] pos={pp.size(0)}, neg={np_.size(0)}")

    def __len__(self): return self.paths.size(0)

    def __getitem__(self, idx):
        path = self.paths[idx]
        feats = []
        for ni in path:
            k = self.i2n.get(ni.item(), str(ni.item())) if self.i2n else ni.item()
            f = self.embs.get(k) if isinstance(self.embs, dict) else self.embs[ni.item()]
            feats.append(f if f is not None else torch.zeros(self.d))
        x = torch.stack(feats)
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        return Data(x=x, edge_index=ei, y=self.labels[idx].unsqueeze(0))


class BenchmarkDS(Dataset):
    def __init__(self, data_path, emb_path, d_model=1280):
        d = torch.load(data_path)
        self.p1, self.p2, self.y = d['protein1_idx'], d['protein2_idx'], d['binary_labels']
        self.embs = torch.load(emb_path)
        self.sk = isinstance(list(self.embs.keys())[0], str) if isinstance(self.embs, dict) else False
        self.d = d_model
        print(f"  [BenchmarkDS] n={len(self.p1)}, pos={self.y.sum().item()}")

    def __len__(self): return len(self.p1)

    def __getitem__(self, idx):
        k1 = str(self.p1[idx].item()) if self.sk else self.p1[idx].item()
        k2 = str(self.p2[idx].item()) if self.sk else self.p2[idx].item()
        e1 = self.embs.get(k1, torch.zeros(self.d)) if isinstance(self.embs, dict) else self.embs[self.p1[idx].item()]
        e2 = self.embs.get(k2, torch.zeros(self.d)) if isinstance(self.embs, dict) else self.embs[self.p2[idx].item()]
        return e1, e2, self.y[idx]


# ============================================================
# STEP 4: GNNpre 预训练
# ============================================================

def step_pretrain(pretrain_dir, emb_path, out_dir, mc, tc, device, logger):
    print("\n" + "=" * 60)
    print("STEP 4: GNNpre/GIN 预训练")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    ds = PretrainDS(pretrain_dir, emb_path, mc['d_model'])
    vs = int(len(ds) * 0.1)
    tr_ds, va_ds = random_split(ds, [len(ds)-vs, vs], generator=torch.Generator().manual_seed(tc['seed']))
    tr_dl = PyGDataLoader(tr_ds, batch_size=tc['gnn_pre']['batch_size'], shuffle=True)
    va_dl = PyGDataLoader(va_ds, batch_size=tc['gnn_pre']['batch_size'], shuffle=False)

    model = GNNpre(mc['d_model'], mc['d_gin'], mc['gin_layers']).to(device)
    model.unfreeze()
    opt = torch.optim.Adam(model.parameters(), lr=tc['gnn_pre']['lr'], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    best_f1, pat = 0.0, 0
    for ep in range(1, tc['gnn_pre']['epochs']+1):
        model.train()
        tl, als, asc = 0, [], []
        for bd in tr_dl:
            bd = bd.to(device)
            opt.zero_grad()
            yp = model(bd)
            loss = F.binary_cross_entropy(yp.squeeze(-1), bd.y.float().view(-1))
            loss.backward()
            opt.step()
            tl += loss.item() * bd.num_graphs
            als.extend(bd.y.cpu().numpy().flatten().tolist())
            asc.extend(yp.detach().cpu().numpy().flatten().tolist())
        sch.step(tl / len(tr_ds))

        model.eval()
        vl, vls, vsc = 0, [], []
        with torch.no_grad():
            for bd in va_dl:
                bd = bd.to(device)
                yp = model(bd)
                loss = F.binary_cross_entropy(yp.squeeze(-1), bd.y.float().view(-1))
                vl += loss.item() * bd.num_graphs
                vls.extend(bd.y.cpu().numpy().flatten().tolist())
                vsc.extend(yp.cpu().numpy().flatten().tolist())

        vm = compute_metrics(np.array(vls), np.array(vsc))
        logger.info(f"GNNpre Ep{ep:3d} | TrL={tl/len(tr_ds):.4f} VaL={vl/len(va_ds):.4f} | VaF1={vm['f1']:.4f}")

        if vm['f1'] > best_f1:
            best_f1 = vm['f1']
            pat = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_gnn_pre.pt"))
        else:
            pat += 1
            if pat >= tc['gnn_pre']['patience']:
                logger.info(f"  早停 Ep{ep}")
                break

    logger.info(f"GNNpre 完成, Best Val F1={best_f1:.4f}")
    return best_f1


# ============================================================
# STEP 5: L3-PPI 微调
# ============================================================

def step_finetune(bench_dir, emb_path, gnn_pre_ckpt, out_dir, mc, tc, device, logger):
    print("\n" + "=" * 60)
    print("STEP 5: L3-PPI 微调 (P→G)")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    tr_ds = BenchmarkDS(os.path.join(bench_dir, "train.pt"), emb_path, mc['d_model'])
    te_ds = BenchmarkDS(os.path.join(bench_dir, "test.pt"), emb_path, mc['d_model'])
    vs = int(len(tr_ds) * 0.1)
    tr_sub, va_sub = random_split(tr_ds, [len(tr_ds)-vs, vs], generator=torch.Generator().manual_seed(tc['seed']))
    tr_dl = DataLoader(tr_sub, batch_size=tc['l3ppi']['batch_size'], shuffle=True)
    va_dl = DataLoader(va_sub, batch_size=tc['l3ppi']['batch_size'], shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=tc['l3ppi']['batch_size'], shuffle=False)

    model = L3PPIClassificationHead(**mc).to(device)
    if os.path.exists(gnn_pre_ckpt):
        sd = torch.load(gnn_pre_ckpt, map_location=device)
        #sd_filtered = {k: v for k, v in sd.items() if not k.startswith('input_proj')}
        model.gnn_pre.load_state_dict(torch.load(gnn_pre_ckpt, map_location=device))#sd_filtered, strict=False)
        model.freeze_gnn_pre()
        logger.info(f"  加载 GNNpre: {gnn_pre_ckpt}")
    else:
        logger.warning(f"  GNNpre 权重不存在: {gnn_pre_ckpt}")

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=tc['l3ppi']['lr'], weight_decay=1e-4)
    def eval_epoch(m, dl):
        m.eval()
        tl, als, asc = 0, [], []
        with torch.no_grad():
            for eu, ev, y in dl:
                eu, ev, y = eu.to(device), ev.to(device), y.to(device)
                yp, gv, pp = m(eu, ev, training=False)
                l, _, _ = m.loss(yp, gv, pp, y)
                tl += l.item() * eu.size(0)
                als.extend(y.cpu().numpy().tolist())
                asc.extend(yp.cpu().numpy().flatten().tolist())
        return tl / len(dl.dataset), compute_metrics(np.array(als), np.array(asc))

    # Phase 1
    logger.info("--- Phase 1: 仅训练提示嵌入 ---")
    for p in model.gnn_gpt.parameters():
        p.requires_grad = False
    model.freeze_gnn_pre()
    opt.param_groups[0]['params'] = list(model.prompt.parameters())

    bf1, pat = 0, 0
    for ep in range(1, tc['l3ppi']['phase1_epochs']+1):
        model.train()
        tl, als, asc = 0, [], []
        for eu, ev, y in tr_dl:
            eu, ev, y = eu.to(device), ev.to(device), y.to(device)
            opt.zero_grad()
            yp, gv, pp = model(eu, ev, training=True)
            l, _, _ = model.loss(yp, gv, pp, y)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tl += l.item() * eu.size(0)
            als.extend(y.cpu().numpy().tolist())
            asc.extend(yp.detach().cpu().numpy().flatten().tolist())
        vl, vm = eval_epoch(model, va_dl)
        logger.info(f"P1 Ep{ep:3d} | TrL={tl/len(tr_ds):.4f} VaF1={vm['f1']:.4f}")
        if vm['f1'] > bf1:
            bf1 = vm['f1']; pat = 0
        else:
            pat += 1
            if pat >= tc['l3ppi']['patience']: break

    # Phase 2
    logger.info("--- Phase 2: 联合优化 ---")
    for p in model.gnn_gpt.parameters():
        p.requires_grad = True
    model.freeze_gnn_pre()
    opt.param_groups[0]['params'] = list(model.prompt.parameters()) + list(model.gnn_gpt.parameters())

    bf2, pat = 0, 0
    for ep in range(1, tc['l3ppi']['phase2_epochs']+1):
        model.train()
        tl, als, asc = 0, [], []
        for eu, ev, y in tr_dl:
            eu, ev, y = eu.to(device), ev.to(device), y.to(device)
            opt.zero_grad()
            yp, gv, pp = model(eu, ev, training=True)
            l, lb, ll = model.loss(yp, gv, pp, y)
            l.backward()
            opt.step()
            tl += l.item() * eu.size(0)
            als.extend(y.cpu().numpy().tolist())
            asc.extend(yp.detach().cpu().numpy().flatten().tolist())
        vl, vm = eval_epoch(model, va_dl)
        logger.info(f"P2 Ep{ep:3d} | TrL={tl/len(tr_ds):.4f} VaF1={vm['f1']:.4f}")
        if vm['f1'] > bf2:
            bf2 = vm['f1']; pat = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_l3ppi.pt"))
        else:
            pat += 1
            if pat >= tc['l3ppi']['patience']: break

    # Test
    ckpt = os.path.join(out_dir, "best_l3ppi.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    _, tm = eval_epoch(model, te_dl)
    logger.info(f"Test: {fmt_metrics(tm)}")
    json.dump({"p1_f1": bf1, "p2_f1": bf2, "test": tm}, open(os.path.join(out_dir, "results.json"), 'w'), indent=2)
    return tm


# ============================================================
# STEP 6: 独立评测
# ============================================================

def step_evaluate(ckpt, bench_dir, emb_path, mc, device):
    print("\n" + "=" * 60)
    print("STEP 6: 评测")
    print("=" * 60)
    model = L3PPIClassificationHead(**mc).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ds = BenchmarkDS(os.path.join(bench_dir, "test.pt"), emb_path, mc['d_model'])
    dl = DataLoader(ds, batch_size=128, shuffle=False)

    als, asc, aap = [], [], []
    with torch.no_grad():
        for eu, ev, y in dl:
            eu, ev = eu.to(device), ev.to(device)
            pred, sc, ap = model.predict(eu, ev)
            als.extend(y.numpy().tolist())
            asc.extend(sc.cpu().numpy().flatten().tolist())
            aap.extend(ap)

    m = compute_metrics(np.array(als), np.array(asc))
    print(f"\n{'='*60}")
    print(f"测试结果: {fmt_metrics(m)}")
    print(f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"平均激活路径: {np.mean([len(a) for a in aap]):.2f}/{mc['K']}")
    print(f"{'='*60}")
    return m


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="L3-PPI 全流程")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", nargs="+", default=["all"],
                        choices=["all", "download", "preprocess", "embeddings", "pretrain", "finetune", "evaluate"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    # 加载配置
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config,'r',encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    mc = cfg.get("model", {})
    tc = cfg.get("training", {})

    raw_dir = paths.get("raw_dir", "data/raw")
    pretrain_dir = paths.get("pretrain_dir", "data/pretrain")
    bench_dir_base = paths.get("benchmark_dir", "data/benchmark")
    emb_dir = paths.get("embeddings_dir", "data/embeddings")
    ckpt_dir = paths.get("checkpoint_dir", "checkpoints")
    log_dir = paths.get("log_dir", "logs")

    ds_name = args.dataset or cfg.get("dataset", {}).get("name", "SHS27k")
    split = args.split or cfg.get("dataset", {}).get("split", "BFS")
    bench_dir = os.path.join(bench_dir_base, f"{ds_name}_{split}")
    emb_path = os.path.join(emb_dir, f"esm2_650m_{ds_name}.pt")
    gnn_pre_dir = os.path.join(ckpt_dir, "gnn_pre")
    gnn_pre_ckpt = os.path.join(gnn_pre_dir, "best_gnn_pre.pt")
    l3ppi_dir = os.path.join(ckpt_dir, "l3ppi", f"{ds_name}_{split}")

    steps = args.steps
    if "all" in steps:
        steps = ["download", "preprocess", "embeddings", "pretrain", "finetune", "evaluate"]

    set_seed(tc.get("seed", 42))
    device = get_device(cfg)
    logger = setup_logger(log_dir)
    logger.info(f"Steps: {steps}")
    logger.info(f"Dataset: {ds_name}_{split}")
    logger.info(f"Device: {device}")

    if "download" in steps:
        step_download(raw_dir)

    if "preprocess" in steps:
        step_preprocess(raw_dir, pretrain_dir, bench_dir_base, seed=tc.get("seed", 42))

    if "embeddings" in steps:
        step_embeddings(raw_dir, emb_dir,pretrain_dir)

    if "pretrain" in steps:
        # 预训练嵌入：优先用专门的预训练嵌入，否则回退到评测基准嵌入
        pt_emb = os.path.join(emb_dir, "esm2_650m_pretrain.pt")
        if not os.path.exists(pt_emb):
            pt_emb = emb_path
        step_pretrain(pretrain_dir, pt_emb, gnn_pre_dir, mc, tc, device, logger)

    if "finetune" in steps:
        step_finetune(bench_dir, emb_path, gnn_pre_ckpt, l3ppi_dir, mc, tc, device, logger)

    if "evaluate" in steps:
        ckpt = os.path.join(l3ppi_dir, "best_l3ppi.pt")
        if os.path.exists(ckpt):
            step_evaluate(ckpt, bench_dir, emb_path, mc, device)
        else:
            print(f"[WARN] 模型不存在: {ckpt}")

    print("\n✅ 全部完成!")


if __name__ == "__main__":
    main()