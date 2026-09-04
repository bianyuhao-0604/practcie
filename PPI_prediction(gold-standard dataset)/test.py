import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from torch.utils.data import Dataset, DataLoader

# ==================== 请根据你的模型定义导入或复制 baseline2d 类 ====================
# 这里假设你的模型类已经定义在下方（与之前训练时保持一致）
class Attention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        B = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(B, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(B, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(B, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.dropout(torch.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0,2,1,3).contiguous().view(B, -1, self.hid_dim)
        x = self.fc(x)
        return x

class Feedforward(nn.Module):
    def __init__(self, hid_dim, ff_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        self.sa = Attention(hid_dim, n_heads, dropout)
        self.ff = Feedforward(hid_dim, ff_dim, dropout)
    def forward(self, x, mask=None):
        x = self.ln1(x + self.do1(self.sa(x, x, x, mask)))
        x = self.ln2(x + self.do2(self.ff(x)))
        return x

class baseline2d(nn.Module):
    def __init__(self, embed_dim, h3=64, kernel_size=2, pooling='avg',
                 num_heads=4, ff_dim=256, dropout=0.2):
        super(baseline2d, self).__init__()
        if embed_dim < 30:
            h = h3
            h2 = h3
        else:
            h = int(embed_dim // 4)
            h2 = int(h // 4)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.conv = nn.Conv2d(h3, 1, kernel_size=kernel_size, padding='same')
        if pooling == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)
        self.fc2 = nn.Linear(h, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.encoder = EncoderLayer(h3, num_heads, ff_dim, dropout)

    def forward(self, x1, x2, mask1=None, mask2=None):
        B = x1.size(0)
        x1 = self.layer_norm1(x1)
        x2 = self.layer_norm2(x2)
        x1 = self.ReLU(self.fc1(x1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))
        x2 = self.ReLU(self.fc1(x2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))
        if mask1 is not None:
            attn_mask1 = mask1.unsqueeze(1) & mask1.unsqueeze(2)
            attn_mask1 = attn_mask1.unsqueeze(1)
        else:
            attn_mask1 = None
        if mask2 is not None:
            attn_mask2 = mask2.unsqueeze(1) & mask2.unsqueeze(2)
            attn_mask2 = attn_mask2.unsqueeze(1)
        else:
            attn_mask2 = None
        x1 = self.encoder(x1, attn_mask1)
        x2 = self.encoder(x2, attn_mask2)
        mat = torch.einsum('bik,bjk->bijk', x1, x2)
        mat = mat.permute(0,3,1,2)
        mat = self.conv(mat)
        mat = self.pool(mat)
        m, _ = mat.view(B, -1).max(dim=1)
        return m, mat   # 注意：返回 logits，评估时需要 sigmoid

# ==================== 数据集定义（与训练时一致） ====================
class PPIDataset(Dataset):
    def __init__(self, df, h5_path, layer=33, exclude_special=True, max_len=256):
        self.df = df
        self.h5_path = h5_path
        self.layer = layer
        self.exclude_special = exclude_special
        self.max_len = max_len
        self.h5_file = None

    def _open_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id1, id2, label = row['Id1'], row['Id2'], row['label']
        h5 = self._open_h5()
        arr1 = h5[id1][:]
        arr2 = h5[id2][:]
        if self.exclude_special and arr1.shape[0] > 2:
            arr1 = arr1[1:-1]
        if self.exclude_special and arr2.shape[0] > 2:
            arr2 = arr2[1:-1]
        if self.max_len is not None:
            if arr1.shape[0] > self.max_len:
                arr1 = arr1[:self.max_len]
            if arr2.shape[0] > self.max_len:
                arr2 = arr2[:self.max_len]
        seq1 = torch.from_numpy(arr1).float()
        seq2 = torch.from_numpy(arr2).float()
        return seq1, seq2, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    seq1_list, seq2_list, labels = zip(*batch)
    B = len(seq1_list)
    L1_max = max(s.size(0) for s in seq1_list)
    L2_max = max(s.size(0) for s in seq2_list)
    D = seq1_list[0].size(1)
    padded_seq1 = torch.zeros(B, L1_max, D)
    padded_seq2 = torch.zeros(B, L2_max, D)
    mask1 = torch.zeros(B, L1_max, dtype=torch.bool)
    mask2 = torch.zeros(B, L2_max, dtype=torch.bool)
    for i, (s1, s2) in enumerate(zip(seq1_list, seq2_list)):
        L1 = s1.size(0)
        L2 = s2.size(0)
        padded_seq1[i, :L1, :] = s1
        padded_seq2[i, :L2, :] = s2
        mask1[i, :L1] = True
        mask2[i, :L2] = True
    labels = torch.stack(labels)
    return padded_seq1, padded_seq2, mask1, mask2, labels

# ==================== 配置区 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H5_PATH = "D:/pythonprojects/practice-github/PPI_prediction(gold-standard dataset)/Embeddings/embeddings_per_tok.h5"
MODEL_PATH = "D:/pythonprojects/practice-github/PPI_prediction(gold-standard dataset)/best_baseline2d_attn.pth"   # 请修改为你的模型路径
MAX_SEQ_LEN = 256
EMB_DIM = 1280
LAYER = 33

# 测试集文件
base_path = "D:/pythonprojects/practice-github/PPI_prediction(gold-standard dataset)/dataset"
test_pos = f"{base_path}/Intra2_pos_rr.txt"
test_neg = f"{base_path}/Intra2_neg_rr.txt"

# ==================== 加载数据 ====================
def load_ppi_pairs(pos_file, neg_file):
    pos_df = pd.read_csv(pos_file, sep=r'\s+', header=None, names=['Id1','Id2'])
    neg_df = pd.read_csv(neg_file, sep=r'\s+', header=None, names=['Id1','Id2'])
    pos_df['label'] = 1
    neg_df['label'] = 0
    all_pairs = pd.concat([pos_df, neg_df], ignore_index=True)
    all_pairs = all_pairs.sample(frac=1, random_state=42).reset_index(drop=True)
    return all_pairs

test_df = load_ppi_pairs(test_pos, test_neg)
print(f"测试集样本数: {len(test_df)}")
print(f"标签分布: 正样本 {test_df['label'].sum()}, 负样本 {len(test_df)-test_df['label'].sum()}")

# 构建 DataLoader
test_dataset = PPIDataset(test_df, H5_PATH, layer=LAYER, exclude_special=True, max_len=MAX_SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)

# ==================== 加载模型 ====================
model = baseline2d(embed_dim=EMB_DIM, h3=64, kernel_size=2, pooling='avg',
                   num_heads=4, ff_dim=256, dropout=0.2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==================== 收集预测概率和标签 ====================
all_probs = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        seq1, seq2, mask1, mask2, labels = batch
        seq1, seq2 = seq1.to(DEVICE), seq2.to(DEVICE)
        mask1, mask2 = mask1.to(DEVICE), mask2.to(DEVICE)
        logits, _ = model(seq1, seq2, mask1, mask2)
        probs = torch.sigmoid(logits)   # 转为概率
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

probs = np.array(all_probs)
labels = np.array(all_labels)

# ==================== 1. 概率分布统计 ====================
pos_probs = probs[labels == 1]
neg_probs = probs[labels == 0]

print("\n=== 预测概率统计 ===")
print(f"正样本概率 - mean: {pos_probs.mean():.4f}, median: {np.median(pos_probs):.4f}, "
      f"max: {pos_probs.max():.4f}, min: {pos_probs.min():.4f}")
print(f"负样本概率 - mean: {neg_probs.mean():.4f}, median: {np.median(neg_probs):.4f}, "
      f"max: {neg_probs.max():.4f}, min: {neg_probs.min():.4f}")

# ==================== 2. 概率分布直方图 ====================
plt.figure(figsize=(10,5))
plt.hist(pos_probs, bins=50, alpha=0.6, label='Positive')
plt.hist(neg_probs, bins=50, alpha=0.6, label='Negative')
plt.xlabel('Predicted probability')
plt.ylabel('Frequency')
plt.legend()
plt.title('Probability distribution by class')
plt.show()

# ==================== 3. 默认阈值指标 ====================
default_preds = (probs > 0.5).astype(int)
print("\n=== 默认阈值 0.5 指标 ===")
print(f"Accuracy: {accuracy_score(labels, default_preds):.4f}")
print(f"Precision: {precision_score(labels, default_preds):.4f}")
print(f"Recall: {recall_score(labels, default_preds):.4f}")
print(f"F1: {f1_score(labels, default_preds):.4f}")
print(f"AUROC: {roc_auc_score(labels, probs):.4f}")

# ==================== 4. 寻找最佳阈值（基于验证集？这里用测试集演示，实际应在验证集上选择） ====================
precisions, recalls, thresholds = precision_recall_curve(labels, probs)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"\n最佳阈值（基于测试集）: {best_threshold:.4f}, 对应 F1: {f1_scores[best_idx]:.4f}")

best_preds = (probs > best_threshold).astype(int)
print("=== 最佳阈值指标 ===")
print(f"Accuracy: {accuracy_score(labels, best_preds):.4f}")
print(f"Precision: {precision_score(labels, best_preds):.4f}")
print(f"Recall: {recall_score(labels, best_preds):.4f}")
print(f"F1: {f1_score(labels, best_preds):.4f}")

# ==================== 5. PR 曲线 ====================
plt.figure(figsize=(6,6))
plt.plot(recalls, precisions, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.grid(True)
plt.show()