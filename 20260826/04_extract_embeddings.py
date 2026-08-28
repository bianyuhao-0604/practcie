"""
04_extract_embeddings.py
使用 ESM2-650M 提取蛋白序列的嵌入向量，并缓存到本地。

输入: SHS27k/SHS148k 蛋白序列表（TSV）
输出: 蛋白 ID -> 嵌入向量的字典（PyTorch tensor）
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
RAW_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw')
EMBED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'embeddings')
os.makedirs(EMBED_DIR, exist_ok=True)

ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_EMBED_DIM = 1280
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sequences(filepath: str) -> dict:
    """
    加载蛋白序列表（TSV 格式）。
    
    文件格式:
      protein_id  sequence
      ENSP00000xxx  MEEPQSDPSV...
    
    Returns:
        {protein_id: sequence}
    """
    df = pd.read_csv(filepath, sep='\t')
    
    # 自动检测列名
    seq_col = None
    id_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'sequence' in col_lower or 'seq' in col_lower:
            seq_col = col
        if 'protein' in col_lower or 'id' in col_lower or 'ensg' in col_lower:
            id_col = col
    
    if seq_col is None:
        seq_col = df.columns[-1]  # 假设最后一列是序列
    if id_col is None:
        id_col = df.columns[0]    # 假设第一列是 ID
    
    sequences = {}
    for _, row in df.iterrows():
        protein_id = str(row[id_col]).strip()
        sequence = str(row[seq_col]).strip()
        if sequence and sequence != 'nan':
            sequences[protein_id] = sequence
    
    print(f"  加载了 {len(sequences)} 条蛋白序列")
    return sequences


def extract_esm2_embeddings(sequences: dict, model, tokenizer) -> dict:
    """
    使用 ESM2 提取蛋白嵌入。
    
    对每条序列，取最后一层隐藏状态的 [CLS] token 作为蛋白嵌入。
    
    Args:
        sequences: {protein_id: amino_acid_sequence}
        model: ESM2 模型
        tokenizer: ESM2 tokenizer
    
    Returns:
        {protein_id: embedding_tensor [1280]}
    """
    model.eval()
    embeddings = {}
    
    protein_ids = list(sequences.keys())
    
    # 分批处理
    for batch_start in tqdm(range(0, len(protein_ids), BATCH_SIZE), 
                            desc="提取 ESM2 嵌入"):
        batch_ids = protein_ids[batch_start:batch_start + BATCH_SIZE]
        batch_seqs = [sequences[pid] for pid in batch_ids]
        
        # 截断过长序列
        batch_seqs = [seq[:MAX_SEQ_LENGTH - 2] for seq in batch_seqs]
        
        # Tokenize
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(DEVICE)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 提取 [CLS] token 的嵌入（第一个 token）
        # last_hidden_state: [batch, seq_len, 1280]
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 1280]
        
        for i, pid in enumerate(batch_ids):
            embeddings[pid] = cls_embeddings[i].cpu()
    
    return embeddings


def main():
    print("L3-PPI ESM2 嵌入提取")
    print(f"设备: {DEVICE}")
    print(f"模型: {ESM2_MODEL_NAME}")
    print(f"嵌入维度: {ESM2_EMBED_DIM}")
    
    # 加载 ESM2 模型
    print("\n[加载] ESM2-650M 模型...")
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM2_MODEL_NAME).to(DEVICE)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 处理每个数据集
    datasets = [
        ("SHS27k", "protein.SHS27k.sequences.dictionary.pro3.tsv", "esm2_650m_SHS27k.pt"),
        ("SHS148k", "protein.SHS148k.sequences.dictionary.tsv", "esm2_650m_SHS148k.pt"),
    ]
    
    for dataset_name, seq_file, embed_file in datasets:
        seq_path = os.path.join(RAW_DIR, seq_file)
        embed_path = os.path.join(EMBED_DIR, embed_file)
        
        if not os.path.exists(seq_path):
            print(f"\n[跳过] {dataset_name} 序列文件不存在: {seq_path}")
            continue
        
        if os.path.exists(embed_path):
            print(f"\n[跳过] {dataset_name} 嵌入已存在: {embed_path}")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"处理 {dataset_name}")
        print(f"{'=' * 60}")
        
        # 加载序列
        sequences = load_sequences(seq_path)
        
        # 提取嵌入
        embeddings = extract_esm2_embeddings(sequences, model, tokenizer)
        
        # 保存
        torch.save(embeddings, embed_path)
        
        # 保存元数据
        metadata = {
            "dataset": dataset_name,
            "model": ESM2_MODEL_NAME,
            "embed_dim": ESM2_EMBED_DIM,
            "num_proteins": len(embeddings),
            "max_seq_length": MAX_SEQ_LENGTH,
        }
        meta_path = embed_path.replace('.pt', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  保存: {embed_path}")
        print(f"  嵌入数量: {len(embeddings)}")
        sample_key = list(embeddings.keys())[0]
        print(f"  嵌入形状: {embeddings[sample_key].shape}")
    
    print("\n" + "=" * 60)
    print("嵌入提取完成！")
    print(f"输出目录: {EMBED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()