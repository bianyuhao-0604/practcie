import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ==================== 配置区 ====================
# 全量蛋白质嵌入字典（pkl）路径
EMBEDDING_PKL = "D:/PPI_prediction_study-master/Embeddings/embeddings_mean.pkl"

# 正负样本 TXT 文件（每个 split 两个文件，两列，无表头，空白分隔）
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

# 输出目录
OUTPUT_DIR = "D:/PPI_prediction_study-master/data/pca_features_trainfit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PCA 目标维度
N_COMPONENTS = 40
# 随机种子
RANDOM_STATE = 42
# ================================================

def load_pkl_dict(path):
    """加载 pkl 字典并统一值为 numpy 一维数组（float32）"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    for key, value in d.items():
        if hasattr(value, "numpy"):          # torch tensor
            value = value.cpu().numpy()
        elif isinstance(value, list):
            value = np.array(value, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32).flatten()
        d[key] = value
    return d

def read_pairs_from_txt(path):
    """读取两列 txt 文件，返回 DataFrame 包含 Id1, Id2（去除首尾空格）"""
    df = pd.read_csv(path, sep=r'\s+', header=None, names=['Id1', 'Id2'])
    df['Id1'] = df['Id1'].str.strip()
    df['Id2'] = df['Id2'].str.strip()
    return df

def load_pos_neg(pos_path, neg_path):
    """合并正负样本并添加 Interact 标签，打乱顺序"""
    pos_df = read_pairs_from_txt(pos_path)
    neg_df = read_pairs_from_txt(neg_path)
    pos_df['Interact'] = 1
    neg_df['Interact'] = 0
    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return combined

def build_features(df, embed_dict):
    """将蛋白质对拼接为 2560 维特征向量，返回 X 和 y"""
    X_list, y_list = [], []
    missing = 0
    for _, row in df.iterrows():
        id1, id2 = row['Id1'], row['Id2']
        if id1 in embed_dict and id2 in embed_dict:
            feat = np.concatenate([embed_dict[id1], embed_dict[id2]])  # 1280+1280
            X_list.append(feat)
            y_list.append(row['Interact'])
        else:
            missing += 1
    if missing:
        print(f"警告：{missing} 条蛋白质对因缺少嵌入被跳过")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

# 1. 加载嵌入字典
print("加载嵌入字典...")
embed_dict = load_pkl_dict(EMBEDDING_PKL)
print(f"嵌入字典大小: {len(embed_dict)}")

# 2. 构建三个集合的原始拼接特征
features = {}
for split, paths in SPLIT_FILES.items():
    print(f"\n构建 {split} 集特征...")
    df = load_pos_neg(paths["pos"], paths["neg"])
    X, y = build_features(df, embed_dict)
    features[split] = (X, y)
    print(f"  X 形状: {X.shape}, y 分布: {np.bincount(y)}")

# 3. 只用训练集拟合 PCA
X_train, y_train = features["train"]
print(f"\n训练集原始特征维度: {X_train.shape[1]}")
print(f"拟合 PCA 到 {N_COMPONENTS} 维...")
pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
pca.fit(X_train)

# 4. 投影所有集合
pca_features = {}
for split in ["train", "val", "test"]:
    X, y = features[split]
    X_pca = pca.transform(X)
    pca_features[split] = (X_pca, y)
    print(f"{split} 降维后 X 形状: {X_pca.shape}")

# 5. 保存
for split in ["train", "val", "test"]:
    X, y = pca_features[split]
    out_file = os.path.join(OUTPUT_DIR, f"ppi_{split}_pca{N_COMPONENTS}_trainfit.npz")
    np.savez(out_file, X=X, y=y)
    print(f"已保存 {out_file}")

print("\n全部完成！")