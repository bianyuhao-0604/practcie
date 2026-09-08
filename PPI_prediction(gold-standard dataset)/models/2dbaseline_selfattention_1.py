"""
2d-Selfattention —— Reim et al., Bioinformatics 2025 (btaf192) 严格复现
=======================================================================
协议对齐清单（对照论文原文与 Table 1）：
  ✓ per-token 逐样本 forward + 整批 backward
  ✓ conv→pool→全局 max 读出（无 InstanceNorm、无 top-1%）
  ✓ spectral_norm 保留在 Attention w_q/k/v/fc 与 FF fc_1/fc_2
  ✓ 每epoch 训练 50% 随机子集
  ✓ MAX_LEN=1000 长序列过滤（对应论文 93719/46421/41100）
  ✓ 早停 = val ACCURACY + PATIENCE=8（不是 AUPR）
  ✓ LR 默认 1e-5（论文强调 LR 是最重要超参且强负相关）
  ✓ 评估指标：Acc / Precision / Recall / F1 / AUPR / MCC
  ✓ 不加载旧 checkpoint，每次从头训练
预期 test 结果（对标论文 Table 1）：
  Acc ≈ 0.616 | Precision ≈ 0.611 | Recall ≈ 0.553
  F1   ≈ 0.591 | AUPR ≈ 0.641 | MCC ≈ 0.22 (推算自 acc/precision/recall)
"""
import gc, random, re, math, os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score,
                             matthews_corrcoef)

# ============================================================
# 0. 配置区
# ============================================================
BASE_DIR = Path(r"D:/pythonprojects/practice-github/PPI_prediction(gold-standard dataset)")
EMB_H5   = BASE_DIR / "Embeddings" / "embeddings_per_tok.h5"

TRAIN_POS = BASE_DIR / "dataset" / "Intra1_pos_rr.txt"
TRAIN_NEG = BASE_DIR / "dataset" / "Intra1_neg_rr.txt"
VAL_POS   = BASE_DIR / "dataset" / "Intra0_pos_rr.txt"
VAL_NEG   = BASE_DIR / "dataset" / "Intra0_neg_rr.txt"
TEST_POS  = BASE_DIR / "dataset" / "Intra2_pos_rr.txt"
TEST_NEG  = BASE_DIR / "dataset" / "Intra2_neg_rr.txt"
CKPT      = BASE_DIR / "checkpoints" / "2d_selfattention_v2.pt"
CKPT.parent.mkdir(parents=True, exist_ok=True)

# ---- 模型超参数（论文协议） ----
EMBED_DIM   = 1280
H3          = 64       # encoder 隐层维度，即 fc3 输出
NUM_HEADS   = 8
FF_DIM      = 256
DROPOUT     = 0.2
POOLING     = 'max'    # 论文：2d-Selfattention 优选 max
KERNEL_SIZE = 2        # 论文未公布，默认 2

# ---- 训练超参数（论文协议） ----
LR           = 1e-5     # ★ 从 1e-4 降回 1e-5；论文强调 LR 是最强超参
BATCH_SIZE   = 16
MAX_EPOCHS   = 60
PATIENCE     = 8        # ★ 论文协议：val accuracy + 8 epoch patience
SUBSET_FRAC  = 0.5      # ★ 每 epoch 训练 50% 随机子集
MAX_LEN      = 1000     # 长序列过滤，对应论文样本数 93719/46421/41100
SEED         = 42
GRAD_CLIP    = 1.0      # 论文未明确，保留作为稳定器

# ---- 数值稳定性开关 ----
USE_LOGITS = True       # True: 移除最后 sigmoid + BCEWithLogitsLoss（更稳）
USE_AMP    = False      # AMP 加速，需 USE_LOGITS=True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# 1. 嵌入读取（惰性 h5 + 鲁棒 ID 映射）
# ============================================================
_H5_HANDLE = None
_KEYSET    = set()
_KEYMAP    = {}
_LEN_CACHE = {}

def open_h5():
    global _H5_HANDLE
    if _H5_HANDLE is None:
        print(f"打开嵌入文件: {EMB_H5}")
        _H5_HANDLE = h5py.File(EMB_H5, "r")
        keys = [k.decode() if isinstance(k, bytes) else k
                for k in _H5_HANDLE.keys()]
        _KEYSET.update(keys)
        for k in keys:
            for cand in _key_candidates(k):
                if cand and cand not in _KEYSET and cand not in _KEYMAP:
                    _KEYMAP[cand] = k
        print(f"h5 中共 {len(keys)} 个蛋白")
    return _H5_HANDLE

def _key_candidates(name):
    cands = [name]
    if "-" in name:   cands.append(name.split("-")[0])        # isoform
    if "|" in name:   cands.append(name.split("|")[0])        # 管道分隔
    if "_" in name:   cands.append(name.split("_")[0])
    if "/" in name:   cands.append(name.split("/")[-1])
    return cands

def resolve_key(name):
    if name in _KEYSET: return name
    if name in _KEYMAP: return _KEYMAP[name]
    for c in _key_candidates(name):
        if c and c in _KEYSET: return c
        if c and c in _KEYMAP: return _KEYMAP[c]
    return None

def protein_len(name):
    if name in _LEN_CACHE: return _LEN_CACHE[name]
    f = open_h5(); k = resolve_key(name)
    if k is None: _LEN_CACHE[name] = None; return None
    obj = f[k]
    if isinstance(obj, h5py.Group):
        obj = obj[list(obj.keys())[0]]
    _LEN_CACHE[name] = int(obj.shape[-2])
    return _LEN_CACHE[name]

def get_embedding_per_tok(name):
    """读取单蛋白 per-token 嵌入，返回 (L, 1280) float32 张量"""
    f = open_h5(); k = resolve_key(name)
    if k is None:
        raise KeyError(f"蛋白 {name!r} 在 h5 中找不到")
    obj = f[k]
    if isinstance(obj, h5py.Group):
        obj = obj[list(obj.keys())[0]]
    return torch.from_numpy(np.array(obj[()], dtype=np.float32))

# ============================================================
# 2. 数据加载（鲁棒 ID 解析 + 长度过滤）
# ============================================================
_ID_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?$"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$"
    r"|^ENS[A-Z]*\d+$"
)
_LABEL_KW = {"interaction", "label", "target", "y", "class", "is_interaction"}
_HEADER_KW = _LABEL_KW | {"name1","name2","protein1","protein2","protein_a",
                          "protein_b","uniprot1","uniprot2","id1","id2",
                          "seq1","seq2","sequence_a","sequence_b"}
_BINARY = {"0","1","0.0","1.0","true","false","pos","neg",
           "positive","negative","yes","no"}

def _label_to_float(v, default_label):
    s = str(v).strip().lower()
    if s in ("1","true","pos","positive","yes"): return 1.0
    if s in ("0","false","neg","negative","no"): return 0.0
    return default_label

def _find_binary_col(df_body):
    best_col, best_frac = None, 0.0
    for j in range(df_body.shape[1]):
        col = df_body.iloc[:, j].astype(str).str.strip().str.lower()
        frac = col.isin(_BINARY).mean()
        if frac > best_frac: best_col, best_frac = j, frac
    return best_col if best_frac >= 0.8 else None

def _read_one_file(path: Path, default_label: float) -> pd.DataFrame:
    df = None
    for sep in (None, "\t", ",", r"\s+", ";"):
        try:
            d = pd.read_csv(path, sep=sep, engine="python", header=None,
                            dtype=str, keep_default_na=False)
        except Exception: continue
        if d.shape[1] >= 2 and d.shape[0] >= 1:
            df = d; break
    if df is None: raise ValueError(f"无法解析 {path.name}")
    df = df[(df != "").any(axis=1)].reset_index(drop=True)
    ncols = df.shape[1]
    row0 = [str(v).strip().lower() for v in df.iloc[0].tolist()]
    row0_label_col = next((j for j, v in enumerate(row0) if v in _LABEL_KW), None)
    row0_kw_hits = sum(1 for v in row0 if v in _HEADER_KW)
    row0_id_hits = sum(1 for j, v in enumerate(row0)
                       if j != row0_label_col and _ID_RE.match(v.upper()))
    first_row_is_data = False
    if row0_label_col is not None or row0_kw_hits >= 2:
        label_col = row0_label_col
        if label_col is None:
            label_col = _find_binary_col(df.iloc[1:])
        first_row_is_data = row0_id_hits > 0
        data = df if first_row_is_data else df.iloc[1:]
    else:
        first_row_is_data = True
        data = df
        label_col = _find_binary_col(data) if ncols >= 3 else None
    name_cols = ([j for j in range(ncols) if j != label_col][:2]
                 if label_col is not None else [0, 1])
    if len(name_cols) < 2:
        raise ValueError(f"{path.name}: 至少需要 2 列蛋白 ID")
    out = pd.DataFrame({
        "name1": data.iloc[:, name_cols[0]].astype(str).str.strip(),
        "name2": data.iloc[:, name_cols[1]].astype(str).str.strip(),
    })
    if label_col is not None:
        out["interaction"] = data.iloc[:, label_col].apply(
            lambda v: _label_to_float(v, default_label))
    else:
        out["interaction"] = float(default_label)
    out = out[(out["name1"] != "") & (out["name2"] != "")].reset_index(drop=True)
    return out

def load_split(pos_path: Path, neg_path: Path, split_name: str) -> pd.DataFrame:
    df_pos = _read_one_file(pos_path, default_label=1.0)
    df_neg = _read_one_file(neg_path, default_label=0.0)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    open_h5()
    keep, missing = [], set()
    for i, (n1, n2) in enumerate(zip(df["name1"], df["name2"])):
        l1, l2 = protein_len(n1), protein_len(n2)
        if l1 is None: missing.add(n1); continue
        if l2 is None: missing.add(n2); continue
        if l1 <= MAX_LEN and l2 <= MAX_LEN: keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    n_pos = int(df["interaction"].sum()); n_neg = len(df) - n_pos
    print(f"[{split_name}] 样本={len(df)} (正={n_pos}, 负={n_neg}, "
          f"占比={n_pos/len(df):.2%}), 缺失嵌入剔除 {len(missing)} 个蛋白")
    if missing:
        print(f"  ⚠ 缺失示例: {sorted(missing)[:5]}")
    return df

class Split:
    def __init__(self, df: pd.DataFrame):
        self.name1 = df["name1"].astype(str).tolist()
        self.name2 = df["name2"].astype(str).tolist()
        self.interaction = df["interaction"].astype(float).tolist()
    def __len__(self): return len(self.interaction)
    def __getitem__(self, idxs):
        return {"name1":   [self.name1[i] for i in idxs],
                "name2":   [self.name2[i] for i in idxs],
                "interaction": [self.interaction[i] for i in idxs]}

def stratified_subset(ds: Split, frac=SUBSET_FRAC):
    pos_idx = [i for i, y in enumerate(ds.interaction) if y == 1.0]
    neg_idx = [i for i, y in enumerate(ds.interaction) if y == 0.0]
    k_pos = max(1, int(frac * len(pos_idx)))
    k_neg = max(1, int(frac * len(neg_idx)))
    idxs = random.sample(pos_idx, k_pos) + random.sample(neg_idx, k_neg)
    random.shuffle(idxs)
    return idxs

# ============================================================
# 3. 模型（spectral_norm 保留；无 InstanceNorm；全局 max 读出）
# ============================================================
class Attention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim, self.n_heads = hid_dim, n_heads
        assert hid_dim % n_heads == 0
        self.w_q = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_k = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_v = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.fc  = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.do  = nn.Dropout(dropout)
        # ★ 修复：scale 改为 buffer，避免 device 硬编码
        self.register_buffer("scale",
            torch.sqrt(torch.FloatTensor([hid_dim // n_heads])))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query); K = self.w_k(key); V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        return self.fc(x)

class Feedforward(nn.Module):
    def __init__(self, hid_dim, ff_dim, dropout, activation_fn="swish"):
        super().__init__()
        self.fc_1 = spectral_norm(nn.Linear(hid_dim, ff_dim))
        self.fc_2 = spectral_norm(nn.Linear(ff_dim, hid_dim))
        self.do   = nn.Dropout(dropout)
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "swish": nn.SiLU(),
                "leaky_relu": nn.LeakyReLU(), "mish": nn.Mish(), "elu": nn.ELU()}
        self.activation = acts[activation_fn]
    def forward(self, x):
        x = self.do(self.activation(self.fc_1(x)))
        return self.fc_2(x)

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, ff_dim, dropout, activation_fn="swish"):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(hid_dim), nn.LayerNorm(hid_dim)
        self.do1, self.do2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.sa = Attention(hid_dim, n_heads, dropout)
        self.ff = Feedforward(hid_dim, ff_dim, dropout, activation_fn)
    def forward(self, trg, mask=None):
        trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, mask)))
        trg = self.ln2(trg + self.do2(self.ff(trg)))
        return trg

class SelfAttInteraction(nn.Module):
    """论文 2d-Selfattention：conv→pool→全局 max 读出"""
    def __init__(self, embed_dim, num_heads, h3=64, dropout=0.2,
                 ff_dim=256, pooling='max', kernel_size=2, use_logits=USE_LOGITS):
        super().__init__()
        self.use_logits = use_logits
        h  = int(embed_dim // 4)   # 320
        h2 = int(h // 4)           # 80
        self.encoder = EncoderLayer(h3, num_heads, ff_dim, dropout)
        self.conv = nn.Conv2d(h3, 1, kernel_size=kernel_size, padding='same')
        if pooling == 'max':   self.pool = nn.MaxPool2d(kernel_size)
        elif pooling == 'avg': self.pool = nn.AvgPool2d(kernel_size)
        else: raise ValueError("pooling must be 'max' or 'avg'")
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)
        self.fc2 = nn.Linear(h, h2)
        self.fc3 = nn.Linear(h2, h3)
        if not use_logits:
            self.sigmoid = nn.Sigmoid()

    def forward(self, protein1, protein2, mask1=None, mask2=None):
        x1 = protein1.to(torch.float32).unsqueeze(0)  # (1, L1, 1280)
        x2 = protein2.to(torch.float32).unsqueeze(0)
        x1 = self.ReLU(self.fc3(self.ReLU(self.fc2(self.ReLU(self.fc1(x1))))))
        x2 = self.ReLU(self.fc3(self.ReLU(self.fc2(self.ReLU(self.fc1(x2))))))
        x1 = self.encoder(x1, mask1)
        x2 = self.encoder(x2, mask2)
        mat = torch.einsum('bik,bjk->bijk', x1, x2)   # (1, L1, L2, 64)
        mat = mat.permute(0, 3, 1, 2)                 # (1, 64, L1, L2)
        mat = self.conv(mat)                          # (1, 1, L1, L2)
        x = self.pool(mat)
        m = torch.max(x)                              # 全局 max 读出
        if self.use_logits:
            return m[None], mat                       # ★ 返回 (logits, 接触图)
        return self.sigmoid(m)[None], mat             # ★ 返回 (概率, 接触图)


def batch_iterate(model, batch, device):
    """逐样本 forward + 整批 backward（论文协议）"""
    preds = []
    for i in range(len(batch["interaction"])):
        s1 = get_embedding_per_tok(batch["name1"][i]).to(device)
        s2 = get_embedding_per_tok(batch["name2"][i]).to(device)
        p, _ = model(s1, s2)
        preds.append(p)
    return torch.stack(preds).view(-1)

# ============================================================
# 4. 评估与训练
# ============================================================
@torch.no_grad()
def evaluate(model, ds: Split, device=DEVICE, desc=""):
    model.eval()
    all_p, all_y = [], []
    for s in range(0, len(ds), BATCH_SIZE):
        batch = ds[list(range(s, min(s + BATCH_SIZE, len(ds))))]
        p = batch_iterate(model, batch, device).cpu()
        if USE_LOGITS:
            p = torch.sigmoid(p)   # logits → 概率，用于 AUPR 与二值化
        all_p.append(p); all_y += batch["interaction"]
    p = torch.cat(all_p).numpy(); y = np.asarray(all_y)
    pb = (p >= 0.5).astype(int)
    return {"acc":       accuracy_score(y, pb),
            "precision": precision_score(y, pb, zero_division=0),
            "recall":    recall_score(y, pb, zero_division=0),
            "f1":        f1_score(y, pb, zero_division=0),
            "aupr":      average_precision_score(y, p),
            "mcc":       matthews_corrcoef(y, pb)}

def main():
    set_seed()
    print(f"Device: {DEVICE}")
    print(f"协议: LR={LR}, BATCH={BATCH_SIZE}, PATIENCE={PATIENCE}, "
          f"SUBSET={SUBSET_FRAC}, POOL={POOLING}, KERNEL={KERNEL_SIZE}, "
          f"读出={'logits + BCEWithLogits' if USE_LOGITS else 'sigmoid + BCE'}, "
          f"AMP={USE_AMP}\n")

    train_ds = Split(load_split(TRAIN_POS, TRAIN_NEG, "train"))
    val_ds   = Split(load_split(VAL_POS,   VAL_NEG,   "val"))
    test_ds  = Split(load_split(TEST_POS,  TEST_NEG,  "test"))

    # ---- 样本数与论文对齐检查 ----
    expected = {"train": 93719, "val": 46421, "test": 41100}
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        flag = "✓" if abs(len(ds) - expected[name]) / expected[name] < 0.02 else "⚠"
        print(f"  {flag} {name}: {len(ds)} (论文: {expected[name]})")
    print()

    # ---- 嵌入读取自检 ----
    probes = list(range(0, min(100, len(train_ds)), 7))
    ok = 0
    for i in probes:
        try:
            get_embedding_per_tok(train_ds.name1[i])
            get_embedding_per_tok(train_ds.name2[i])
            ok += 1
        except KeyError:
            pass
    print(f"自检: {ok}/{len(probes)} 条样本嵌入读取成功\n")

    model = SelfAttInteraction(EMBED_DIM, NUM_HEADS, h3=H3,
                               dropout=DROPOUT, ff_dim=FF_DIM,
                               pooling=POOLING, kernel_size=KERNEL_SIZE,
                               use_logits=USE_LOGITS).to(DEVICE)
    # ★ 从头训练，不加载旧 checkpoint
    criterion = (nn.BCEWithLogitsLoss() if USE_LOGITS else nn.BCELoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_acc, patience_cnt = 0.0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        idxs = stratified_subset(train_ds)
        random.shuffle(idxs)
        ep_loss, nb = 0.0, 0
        for s in range(0, len(idxs), BATCH_SIZE):
            batch = train_ds[idxs[s:s + BATCH_SIZE]]
            preds = batch_iterate(model, batch, DEVICE)
            labels = torch.tensor(batch["interaction"],
                                  dtype=torch.float32, device=DEVICE)
            if s == 0:
                pp = torch.sigmoid(preds) if USE_LOGITS else preds
                print(f"  [诊断] preds: min={pp.min():.4f} mean={pp.mean():.4f} "
                      f"max={pp.max():.4f} | ≥0.5占比="
                      f"{(pp >= 0.5).float().mean():.2f} | labels[:8]={labels[:8].tolist()}")
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            ep_loss += loss.item(); nb += 1

        val = evaluate(model, val_ds, DEVICE, desc=f"epoch {epoch}")
        print(f"Epoch {epoch:03d} | loss={ep_loss/nb:.4f} | "
              f"val acc={val['acc']:.3f} f1={val['f1']:.3f} "
              f"aupr={val['aupr']:.3f} mcc={val['mcc']:.3f}")

        gc.collect()
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

        # ★ 论文协议：早停 = val ACCURACY + patience=8
        if val["acc"] > best_acc:
            best_acc, patience_cnt = val["acc"], 0
            torch.save(model.state_dict(), CKPT)
            print(f"  → 新最优 val acc={best_acc:.3f}, 已保存 checkpoint")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"早停：val acc {PATIENCE} 个 epoch 无改善, "
                      f"最优 acc={best_acc:.3f}")
                break

    if CKPT.exists():
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    else:
        print("⚠ 训练中未保存 checkpoint, 用最后 epoch 权重测试")

    test = evaluate(model, test_ds, DEVICE, desc="test")
    print("\n===== Test (2d-Selfattention, 论文协议) =====")
    print("论文 Table 1:  acc=0.616  precision=0.611  recall=0.553  "
          "f1=0.591  aupr=0.641")
    print("-" * 60)
    for k, v in test.items():
        print(f"{k:>10s}: {v:.3f}")
    open_h5().close()

if __name__ == "__main__":
    main()
