"""
aa_utils.py — 氨基酸序列编码工具

提供两种编码方式：
  1. amino_acid_composition：22 维氨基酸组成向量（快速、轻量）
  2. one_hot_encode：按位置 one-hot 编码（保留序列顺序信息）

同时提供从 SHS148k 原始 .tsv 文件加载序列字典的接口。
"""

import numpy as np
import pandas as pd
from collections import OrderedDict


# ───────────────────── 氨基酸字母表 ─────────────────────
# 20 标准氨基酸 + gap(-) + unknown(?)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-?"
AA_TO_IDX   = {aa: i for i, aa in enumerate(AA_ALPHABET)}
NUM_AA      = len(AA_ALPHABET)  # 22


# ───────────────────── 编码函数 ─────────────────────
def amino_acid_composition(seq: str) -> np.ndarray:
    """
    将氨基酸序列编码为 22 维组成向量（归一化频率）。
    顺序对应 AA_ALPHABET。
    """
    vec = np.zeros(NUM_AA, dtype=np.float32)
    length = max(len(seq), 1)
    for ch in seq.upper():
        if ch in AA_TO_IDX:
            vec[AA_TO_IDX[ch]] += 1
    return vec / length


def one_hot_encode(seq: str, max_len: int = 1024) -> np.ndarray:
    """
    将氨基酸序列按位置做 one-hot 编码。
    返回形状 (max_len, 22)，超过 max_len 截断，不足补零。
    """
    arr = np.zeros((max_len, NUM_AA), dtype=np.float32)
    for i, ch in enumerate(seq.upper()[:max_len]):
        if ch in AA_TO_IDX:
            arr[i, AA_TO_IDX[ch]] = 1.0
    return arr


def encode_sequences(seq_dict: "dict[str, str]") -> np.ndarray:
    """
    对 {protein_id: sequence} 字典批量编码为组成向量。
    返回 (num_proteins, 22) 的 float32 矩阵。
    """
    matrix = np.zeros((len(seq_dict), NUM_AA), dtype=np.float32)
    for i, (_, seq) in enumerate(seq_dict.items()):
        matrix[i] = amino_acid_composition(seq)
    return matrix


# ───────────────────── 数据加载 ─────────────────────
def load_sequences(path: str) -> "OrderedDict[str, str]":
    """
    加载 SHS148k 原始序列文件。
    文件格式：TSV，每行  protein_id \\t  sequence
    返回 OrderedDict 保证编号稳定。
    """
    seq_dict: "OrderedDict[str, str]" = OrderedDict()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pid, seq = parts[0].strip(), parts[1].strip()
            seq_dict[pid] = seq
    print(f"[load_sequences] 加载 {len(seq_dict)} 条蛋白质序列 ← {path}")
    return seq_dict


def load_ppi_actions(path: str) -> list:
    """
    加载 SHS148k PPI 注释文件。
    文件格式（制表符分隔）：
        protein1  protein2  action_type  score
    可能包含表头行（以 # 或列名开头），自动跳过。
    返回 list of (id1, id2, action, score)
    """
    records = []
    header_kw = {"protein1", "protein", "#"}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            # 跳过表头
            if parts[0].strip().lower() in header_kw:
                continue
            id1, id2, action, score = parts[0], parts[1], parts[2], parts[3]
            try:
                score = float(score)
            except ValueError:
                score = 0.0
            records.append((id1.strip(), id2.strip(), action.strip(), score))
    print(f"[load_ppi_actions] 加载 {len(records)} 条 PPI 注释 ← {path}")
    return records


# 修正上面函数签名中的笔误（保持兼容）
def load_ppi(path: str) -> list:
    return load_ppi_actions(path)


if __name__ == "__main__":
    # 快速自检
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQVVME"
    comp = amino_acid_composition(test_seq)
    oh   = one_hot_encode(test_seq, max_len=128)
    print(f"组成向量维度: {comp.shape}, 求和={comp.sum():.4f}")
    print(f"One-hot 形状: {oh.shape}")
    assert abs(comp.sum() - 1.0) < 1e-6, "组成向量未归一化！"
    print("✅ aa_utils 自检通过")
