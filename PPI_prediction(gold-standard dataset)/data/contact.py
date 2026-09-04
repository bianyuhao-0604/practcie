import numpy as np
import pandas as pd
import pickle
import os

# ==================== 配置区 ====================
EMBEDDING_PKL = "D:/PPI_prediction_study-master/Embeddings/embeddings_mean.pkl"  # 全量嵌入字典

SPLIT_FILES = {
    "train": {
        "pos": "D:/PPI_prediction_study-master/dataset/Intra1_pos_rr.txt",
        "neg": "D:/PPI_prediction_study-master/dataset/Intra1_neg_rr.txt",
    },
    "val": {
        "pos": "D:/PPI_prediction_study-master/dataset/Intra0_pos_rr.txt",
        "neg": "D:/PPI_prediction_study-master/dataset/Intra0_neg_rr.txt",
    },
    "test": {
        "pos": "D:/PPI_prediction_study-master/dataset/Intra2_pos_rr.txt",
        "neg": "D:/PPI_prediction_study-master/dataset/Intra2_neg_rr.txt",
    }
}

OUTPUT_DIR = "D:/PPI_prediction_study-master/data/split_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ================================================

def load_pkl_dict(path):
    """加载 pkl 字典，值转为 numpy 一维数组"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    for key, value in d.items():
        if hasattr(value, "numpy"):
            value = value.cpu().numpy()
        elif isinstance(value, list):
            value = np.array(value, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32).flatten()
        d[key] = value
    return d

def read_pairs_from_txt(path):
    """读取两列 txt 文件，返回 DataFrame 包含 Id1, Id2"""
    df = pd.read_csv(path, sep=r'\s+', header=None, names=['Id1', 'Id2'])
    df['Id1'] = df['Id1'].str.strip()
    df['Id2'] = df['Id2'].str.strip()
    return df

def build_features(df, embed_dict):
    """将蛋白质对拼接成特征向量，返回 X 和 y（这里 y 根据 label 参数指定）"""
    X_list = []
    missing = 0
    for _, row in df.iterrows():
        id1, id2 = row['Id1'], row['Id2']
        if id1 in embed_dict and id2 in embed_dict:
            feat = np.concatenate([embed_dict[id1], embed_dict[id2]])
            X_list.append(feat)
        else:
            missing += 1
    if missing:
        print(f"警告：{missing} 条蛋白质对因缺少嵌入被跳过")
    return np.array(X_list, dtype=np.float32)

# 1. 加载嵌入字典
print("加载嵌入字典...")
embed_dict = load_pkl_dict(EMBEDDING_PKL)
print(f"嵌入字典大小: {len(embed_dict)}")

# 2. 分别处理每个 split 的正负样本
for split, paths in SPLIT_FILES.items():
    print(f"\n处理 {split} 集...")
    # 正样本
    pos_df = read_pairs_from_txt(paths["pos"])
    X_pos = build_features(pos_df, embed_dict)
    y_pos = np.ones(len(X_pos), dtype=np.int64)   # 正样本标签 1
    # 负样本
    neg_df = read_pairs_from_txt(paths["neg"])
    X_neg = build_features(neg_df, embed_dict)
    y_neg = np.zeros(len(X_neg), dtype=np.int64)  # 负样本标签 0

    print(f"正样本特征形状: {X_pos.shape}, 负样本特征形状: {X_neg.shape}")

    # 保存为独立的 npz 文件
    np.savez(os.path.join(OUTPUT_DIR, f"{split}_pos_features.npz"), X=X_pos, y=y_pos)
    np.savez(os.path.join(OUTPUT_DIR, f"{split}_neg_features.npz"), X=X_neg, y=y_neg)

    # 可选：合并正负样本保存
    X_combined = np.vstack([X_pos, X_neg])
    y_combined = np.hstack([y_pos, y_neg])
    np.savez(os.path.join(OUTPUT_DIR, f"{split}_combined_features.npz"), X=X_combined, y=y_combined)
    print(f"已保存 {split} 的正负样本特征（分开和合并版本）。")

print("\n全部完成！")