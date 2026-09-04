import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import h5py
import os
import time
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ==================== 模型定义（加入 LayerNorm）====================
class baseline2d(nn.Module):
    def __init__(self, embed_dim, h3=64, kernel_size=2, pooling='avg'):
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, mask1=None, mask2=None):
        B = x1.size(0)
        x1 = self.layer_norm1(x1)
        x2 = self.layer_norm2(x2)

        x1 = self.ReLU(self.fc1(x1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))   # [B, L1, h3]

        x2 = self.ReLU(self.fc1(x2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))   # [B, L2, h3]

        mat = torch.einsum('bik,bjk->bijk', x1, x2)  # [B, L1, L2, h3]
        mat = mat.permute(0, 3, 1, 2)                # [B, h3, L1, L2]

        mat = self.conv(mat)                         # [B, 1, L1', L2']
        mat = self.pool(mat)                         # [B, 1, L1'', L2'']

        m, _ = mat.view(B, -1).max(dim=1)            # [B]
        pred = m                     # [B]
        return pred, mat

# ==================== 数据集（惰性加载版）====================
class PPIDataset(Dataset):
    def __init__(self, df, h5_path, layer=33, exclude_special=True, max_len=256):
        self.df = df
        self.h5_path = h5_path
        self.layer = layer
        self.exclude_special = exclude_special
        self.max_len = max_len
        self.h5_file = None  # 延迟打开

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
EMB_DIM = 1280
LAYER = 33
BATCH_SIZE = 4           # 小 batch，适应 8GB 显存
EPOCHS = 20
LR = 1e-4
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 256        # 限制序列长度

H5_PATH = "D:/PPI_prediction_study-master/Embeddings/embeddings_per_tok.h5"
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
# ================================================

def load_ppi_pairs(pos_file, neg_file):
    pos_df = pd.read_csv(pos_file, sep=r'\s+', header=None, names=['Id1', 'Id2'])
    neg_df = pd.read_csv(neg_file, sep=r'\s+', header=None, names=['Id1', 'Id2'])
    pos_df['label'] = 1
    neg_df['label'] = 0
    all_pairs = pd.concat([pos_df, neg_df], ignore_index=True)
    all_pairs = all_pairs.sample(frac=1, random_state=42).reset_index(drop=True)
    return all_pairs

def sample_balanced(df, n):
    pos = df[df['label'] == 1].sample(n=min(n//2, len(df[df['label']==1])), random_state=42)
    neg = df[df['label'] == 0].sample(n=min(n//2, len(df[df['label']==0])), random_state=42)
    return pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        seq1, seq2, mask1, mask2, labels = batch
        seq1, seq2 = seq1.to(device), seq2.to(device)
        mask1, mask2 = mask1.to(device), mask2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            preds, _ = model(seq1, seq2, mask1, mask2)
            loss = criterion(preds, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq1, seq2, mask1, mask2, labels = batch
            seq1, seq2 = seq1.to(device), seq2.to(device)
            mask1, mask2 = mask1.to(device), mask2.to(device)
            with autocast():
                preds, _ = model(seq1, seq2, mask1, mask2)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)

# ==================== 主流程 ====================
if __name__ == "__main__":
    print("加载数据...")
    train_df = load_ppi_pairs(SPLIT_FILES["train"]["pos"], SPLIT_FILES["train"]["neg"])
    val_df   = load_ppi_pairs(SPLIT_FILES["val"]["pos"], SPLIT_FILES["val"]["neg"])
    test_df  = load_ppi_pairs(SPLIT_FILES["test"]["pos"], SPLIT_FILES["test"]["neg"])

    # 小样本快速实验（可选）
    # SAMPLE_TRAIN = 1000
    # SAMPLE_VAL = 500
    # SAMPLE_TEST = 1000
    # train_df = sample_balanced(train_df, SAMPLE_TRAIN)
    # val_df = sample_balanced(val_df, SAMPLE_VAL)
    # test_df = sample_balanced(test_df, SAMPLE_TEST)

    print(f"训练样本: {len(train_df)}, 验证样本: {len(val_df)}, 测试样本: {len(test_df)}")

    train_dataset = PPIDataset(train_df, H5_PATH, layer=LAYER, exclude_special=True, max_len=MAX_SEQ_LEN)
    val_dataset   = PPIDataset(val_df, H5_PATH, layer=LAYER, exclude_special=True, max_len=MAX_SEQ_LEN)
    test_dataset  = PPIDataset(test_df, H5_PATH, layer=LAYER, exclude_special=True, max_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = baseline2d(embed_dim=EMB_DIM, h3=64, kernel_size=2, pooling='avg').to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    print("开始训练...")
    best_val_auroc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, GRAD_CLIP)
        val_preds, val_labels = evaluate(model, val_loader, DEVICE)
        val_auroc = roc_auc_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val AUROC: {val_auroc:.4f} | Time: {time.time()-start_time:.1f}s")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save(model.state_dict(), "best_baseline2d_optimized.pth")
            print(f"  保存最佳模型 (Val AUROC: {best_val_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发，训练结束")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load("best_baseline2d_optimized.pth"))

    # 测试集评估
    test_preds, test_labels = evaluate(model, test_loader, DEVICE)
    test_auroc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
    test_prec = precision_score(test_labels, (test_preds > 0.5).astype(int))
    test_rec = recall_score(test_labels, (test_preds > 0.5).astype(int))
    print(f"\n测试集结果：AUROC: {test_auroc:.4f}, Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")