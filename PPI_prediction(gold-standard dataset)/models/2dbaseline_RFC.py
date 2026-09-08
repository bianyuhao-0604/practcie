# -*- coding: utf-8 -*-
"""
========================================================================
 论文复现补充基线 —— 对照 daisybio/PPI_prediction_study 原文脚本
 用法:
   python repro_baselines.py rfc          # RFC 锚点（对应 baselineRFC.py）
   python repro_baselines.py 2dbaseline   # 无注意力对照组（对应 baseline2d.py）

 与原文的三处适配（数学等价，理由见各处注释）:
   1) 原文均值嵌入存独立 .pt、PCA 存 json → 此处从 per-token h5 对 token 维
      求均值，PCA 用 sklearn 在【训练集】蛋白上拟合（无信息泄漏）
   2) 原文 RFC 用全量 train + max_len=10000 → 此处沿用与深度模型相同的
      load_split（MAX_LEN=1000、剔除缺失嵌入），保证 RFC ↔ 2d-baseline ↔
      2d-Selfattention 三者数据完全一致，对比才公平
   3) baseline2d 默认与你修复后的 SelfAtt 对齐（InstanceNorm + top-1% 读出），
      消融唯一变量 = encoder；置 STRICT_ORIGINAL=True 恢复原文写法
      （预期会重现论文 §3.6 的全正/全负振荡——本身就是可写的复现点）
========================================================================
"""
import re
import sys
import gc
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score)

try:
    import psutil
    _PS = psutil.Process()
except ImportError:
    _PS = None

# ============================================================
# 0. 配置
# ============================================================
BASE_DIR = Path(r"D:/pythonprojects/practice-github/PPI_prediction(gold-standard dataset)")
EMB_H5   = BASE_DIR / "Embeddings" / "embeddings_per_tok.h5"

TRAIN_POS = BASE_DIR / "dataset" / "Intra1_pos_rr.txt"
TRAIN_NEG = BASE_DIR / "dataset" / "Intra1_neg_rr.txt"
VAL_POS   = BASE_DIR / "dataset" / "Intra0_pos_rr.txt"
VAL_NEG   = BASE_DIR / "dataset" / "Intra0_neg_rr.txt"
TEST_POS  = BASE_DIR / "dataset" / "Intra2_pos_rr.txt"
TEST_NEG  = BASE_DIR / "dataset" / "Intra2_neg_rr.txt"

EMBED_DIM   = 1280
H3          = 64
POOLING     = 'max'       # 与你的 SelfAtt 运行一致（论文 3.3 节 2d 模型偏好 max）
KERNEL_SIZE = 2
MAX_LEN     = 1000
SEED        = 42

# ---- 2d-baseline 训练超参（与主脚本完全一致，消融只留 encoder 一个变量）----
LR          = 1e-4
BATCH_SIZE  = 16
MAX_EPOCHS  = 60
PATIENCE    = 20
SUBSET_FRAC = 0.5

# ---- RFC 超参 ----
USE_PCA = False    # ★ False = 不做 PCA, 直接用 1280 维均值嵌入; True = 原行为
COMPONENTS       = 200     # 原文脚本默认；论文 §3.2 的 RFC-40 改成 40
STRICT_ORIGINAL  = False  # True = baseline2d 恢复原文（无 InstanceNorm、max 读出）

MODE   = sys.argv[1].lower() if len(sys.argv) > 1 else "rfc"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_2D = BASE_DIR / "checkpoints" / "2d_baseline_best.pt"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. 嵌入读取（与主脚本相同的 OOM 修复版）
# ============================================================
_H5_HANDLE = None
_KEYSET    = None
_KEYMAP    = {}
_EMB_CACHE = {}
_CACHE_MAX = 0            # ★ per-token 禁缓存（上次 OOM 元凶）
_LEN_CACHE = {}


def open_h5():
    global _H5_HANDLE, _KEYSET
    if _H5_HANDLE is None:
        print(f"打开嵌入文件: {EMB_H5}")
        _H5_HANDLE = h5py.File(EMB_H5, "r")
        keys = [k.decode() if isinstance(k, bytes) else k for k in _H5_HANDLE.keys()]
        _KEYSET = set(keys)
        for k in keys:
            for cand in (k.split("-")[0], k.split("|")[0].split("/")[-1],
                         k.split("_")[0] if "_" in k else None):
                if cand and cand not in _KEYSET and cand not in _KEYMAP:
                    _KEYMAP[cand] = k
        print(f"h5 中共 {len(keys)} 个蛋白")
    return _H5_HANDLE


def resolve_key(name: str):
    if name in _KEYSET:
        return name
    if name in _KEYMAP:
        return _KEYMAP[name]
    for cand in (name.split("-")[0], name.split("|")[0].split("/")[-1],
                 name.split("_")[0] if "_" in name else None):
        if cand and cand in _KEYSET:
            return cand
        if cand and cand in _KEYMAP:
            return _KEYMAP[cand]
    return None


def protein_len(name: str):
    if name in _LEN_CACHE:
        return _LEN_CACHE[name]
    f = open_h5()
    k = resolve_key(name)
    if k is None:
        _LEN_CACHE[name] = None
        return None
    obj = f[k]
    if isinstance(obj, h5py.Group):
        obj = obj[list(obj.keys())[0]]
    _LEN_CACHE[name] = int(obj.shape[-2])
    return _LEN_CACHE[name]


def get_embedding_per_tok(name: str) -> torch.Tensor:
    """读取单蛋白 per-token 嵌入 (L, 1280)。用后即弃，不做缓存。"""
    f = open_h5()
    k = resolve_key(name)
    if k is None:
        raise KeyError(f"蛋白 {name!r} 在 h5 中找不到")
    obj = f[k]
    if isinstance(obj, h5py.Group):
        obj = obj[list(obj.keys())[-1]]
    t = torch.from_numpy(np.array(obj[()], dtype=np.float32))
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


# ============================================================
# 2. 数据加载（与主脚本一致）
# ============================================================
_ID_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?$"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$"
    r"|^ENS[A-Z]*\d+$"
)
_LABEL_KW  = {"interaction", "label", "target", "y", "class", "is_interaction"}
_HEADER_KW = _LABEL_KW | {"name1", "name2", "protein1", "protein2", "protein_a",
                          "protein_b", "uniprot1", "uniprot2", "id1", "id2"}
_BINARY = {"0", "1", "0.0", "1.0", "true", "false", "pos", "neg",
           "positive", "negative", "yes", "no"}


def _label_to_float(v, default_label):
    s = str(v).strip().lower()
    if s in ("1", "true", "pos", "positive", "yes"):  return 1.0
    if s in ("0", "false", "neg", "negative", "no"):  return 0.0
    return default_label


def _find_binary_col(df_body: pd.DataFrame):
    best_col, best_frac = None, 0.0
    for j in range(df_body.shape[1]):
        col = df_body.iloc[:, j].astype(str).str.strip().str.lower()
        frac = col.isin(_BINARY).mean()
        if frac > best_frac:
            best_col, best_frac = j, frac
    return best_col if best_frac >= 0.8 else None


def _read_one_file(path: Path, default_label: float) -> pd.DataFrame:
    df = None
    for sep in (None, "\t", ",", r"\s+", ";"):
        try:
            d = pd.read_csv(path, sep=sep, engine="python", header=None,
                            dtype=str, keep_default_na=False)
        except Exception:
            continue
        if d.shape[1] >= 2 and d.shape[0] >= 1:
            df = d
            break
    if df is None:
        raise ValueError(f"无法解析文件 {path.name}")

    df = df[(df != "").any(axis=1)].reset_index(drop=True)
    ncols = df.shape[1]

    row0 = [str(v).strip().lower() for v in df.iloc[0].tolist()]
    row0_label_col = next((j for j, v in enumerate(row0) if v in _LABEL_KW), None)
    row0_kw_hits   = sum(1 for v in row0 if v in _HEADER_KW)
    row0_id_hits   = sum(1 for j, v in enumerate(row0)
                         if j != row0_label_col and _ID_RE.match(v.upper()))

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
    print(f"[{path.name}] 解析 {len(out)} 行")
    return out


def load_split(pos_path: Path, neg_path: Path) -> pd.DataFrame:
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
        if l1 <= MAX_LEN and l2 <= MAX_LEN:
            keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)

    n_pos = int(df["interaction"].sum())
    print(f"[{pos_path.stem}+{neg_path.stem}] 样本={len(df)} "
          f"(正={n_pos}, 负={len(df)-n_pos})，缺失蛋白 {len(missing)} 个已剔除")
    return df


class Split:
    def __init__(self, df: pd.DataFrame):
        self.name1 = df["name1"].astype(str).tolist()
        self.name2 = df["name2"].astype(str).tolist()
        self.interaction = df["interaction"].astype(float).tolist()

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, idxs):
        return {"name1": [self.name1[i] for i in idxs],
                "name2": [self.name2[i] for i in idxs],
                "interaction": [self.interaction[i] for i in idxs]}


def stratified_subset(ds: Split, frac: float = SUBSET_FRAC):
    pos_idx = [i for i, y in enumerate(ds.interaction) if y == 1.0]
    neg_idx = [i for i, y in enumerate(ds.interaction) if y == 0.0]
    idxs = (random.sample(pos_idx, max(1, int(frac * len(pos_idx))))
            + random.sample(neg_idx, max(1, int(frac * len(neg_idx)))))
    random.shuffle(idxs)
    return idxs


# ============================================================
# 3. 共用前向/评估（与主脚本一致）
# ============================================================
def batch_iterate(model, batch, device):
    """逐样本前向（变长不 padding），整批一次反向"""
    preds = []
    for i in range(len(batch["interaction"])):
        s1 = get_embedding_per_tok(batch["name1"][i]).to(device)
        s2 = get_embedding_per_tok(batch["name2"][i]).to(device)
        p, _ = model(s1, s2)
        preds.append(p)
    return torch.stack(preds).view(-1)


@torch.no_grad()
def evaluate(model, ds: Split, device=DEVICE):
    model.eval()
    all_p, all_y = [], []
    for s in range(0, len(ds), BATCH_SIZE):
        batch = ds[list(range(s, min(s + BATCH_SIZE, len(ds))))]
        p = batch_iterate(model, batch, device).cpu()
        all_p.append(p)
        all_y += batch["interaction"]
    p = torch.cat(all_p).numpy()
    y = np.asarray(all_y)
    pb = (p >= 0.5).astype(int)
    return {"acc": accuracy_score(y, pb),
            "precision": precision_score(y, pb, zero_division=0),
            "recall": recall_score(y, pb, zero_division=0),
            "f1": f1_score(y, pb, zero_division=0),
            "aupr": average_precision_score(y, p)}


# ============================================================
# 4a. RFC 基线（对应原文 baselineRFC.py）
# ============================================================
_MEAN_CACHE = {}          # 均值向量每条 ~5KB，缓存安全（与 per-token 不同）


def mean_embedding(name: str) -> np.ndarray:
    """原文用独立保存的 mean representation；此处对 per-token 的 token 维求平均，等价"""
    if name in _MEAN_CACHE:
        return _MEAN_CACHE[name]
    v = get_embedding_per_tok(name).mean(dim=0).numpy().astype(np.float32)
    _MEAN_CACHE[name] = v
    return v


def run_rfc():
    set_seed()
    print(f"Device: {DEVICE}（RFC 全程 CPU）\n")

    train_df = load_split(TRAIN_POS, TRAIN_NEG)
    val_df   = load_split(VAL_POS,   VAL_NEG)
    test_df  = load_split(TEST_POS,  TEST_NEG)
    # ---- PCA 只在【训练集】蛋白上拟合（原文 PCA 是预计算的全局映射，
    #      此处改在训练集拟合，方法学上更干净且等价）----
    if USE_PCA:
        train_prots = sorted(set(train_df["name1"]) | set(train_df["name2"]))
        print(f"\n训练集唯一蛋白 {len(train_prots)} 个，计算均值嵌入…")
        M = np.stack([mean_embedding(p) for p in train_prots])
        pca = PCA(n_components=COMPONENTS, random_state=SEED).fit(M)
        print(f"PCA: {M.shape} -> {COMPONENTS} 维 "
              f"(累计解释方差 {pca.explained_variance_ratio_.sum():.2%})\n")

        proj_cache = {}
        def proj(name):
            if name not in proj_cache:
                proj_cache[name] = pca.transform(mean_embedding(name)[None])[0]
            return proj_cache[name]
        feat_dim = COMPONENTS
    else:
        proj = mean_embedding      # ★ 直接返回 (1280,) 原始均值向量, 无变换
        feat_dim = EMBED_DIM

    def build(df):
        X = np.stack([np.concatenate([proj(a), proj(b)])
                      for a, b in zip(df["name1"], df["name2"])])
        return X, df["interaction"].to_numpy()

    Xtr, ytr = build(train_df)
    Xva, yva = build(val_df)
    Xte, yte = build(test_df)
    print(f"特征维度: {Xtr.shape[1]}（= {feat_dim}×2，{'PCA' if USE_PCA else '无PCA'}）\n")


    rfc = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    rfc.fit(Xtr, ytr)

    print("===== RFC 结果（原文锚点: acc≈0.577）=====")
    for tag, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
        p01  = rfc.predict(X)
        prob = rfc.predict_proba(X)[:, 1]
        print(f"[{tag:>5}] acc={accuracy_score(y, p01):.3f} "
              f"precision={precision_score(y, p01, zero_division=0):.3f} "
              f"recall={recall_score(y, p01):.3f} "
              f"f1={f1_score(y, p01, zero_division=0):.3f} "
              f"aupr={average_precision_score(y, prob):.3f}")


# ============================================================
# 4b. 2d-baseline 无注意力对照组（对应原文 baseline2d.py）
# ============================================================
class Baseline2D(nn.Module):
    """原文 baseline2d：与 SelfAttInteraction 唯一区别 = 无 encoder 层"""
    def __init__(self, embed_dim, h3=64, kernel_size=2, pooling='max',
                 strict_original=False):
        super().__init__()
        h  = int(embed_dim // 4)     # 320
        h2 = int(h // 4)             # 80
        self.strict = strict_original

        self.conv = nn.Conv2d(h3, 1, kernel_size=kernel_size, padding='same')
        if not self.strict:
            self.map_norm = nn.InstanceNorm2d(1, affine=True)   # 对齐你修复后的 SelfAtt
        if pooling == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)
        else:
            raise ValueError("pooling must be 'max' or 'avg'")

        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)   # 两个蛋白共享（原文如此）
        self.fc2 = nn.Linear(h, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, protein1, protein2):
        x1 = protein1.to(torch.float32).unsqueeze(0)   # (1, L1, D)
        x2 = protein2.to(torch.float32).unsqueeze(0)   # (1, L2, D)
        x1 = self.ReLU(self.fc3(self.ReLU(self.fc2(self.ReLU(self.fc1(x1))))))
        x2 = self.ReLU(self.fc3(self.ReLU(self.fc2(self.ReLU(self.fc1(x2))))))

        mat = torch.einsum('bik,bjk->bijk', x1, x2)    # (1, L1, L2, h3)
        mat = mat.permute(0, 3, 1, 2)                  # (1, h3, L1, L2)
        mat = self.conv(mat)                           # (1, 1, L1, L2)
        if not self.strict:
            mat = self.map_norm(mat)
        x = self.pool(mat)

        if self.strict:                                # 原文读出：全局 max
            m = torch.max(x)
        else:                                          # 对齐读出：top-1% 均值
            flat = x.flatten()
            k = max(1, int(0.01 * flat.numel()))
            m = torch.topk(flat, k).values.mean()
        pred = self.sigmoid(m)[None]
        return pred, mat


def run_2dbaseline():
    set_seed()
    print(f"Device: {DEVICE}\n")

    train_ds = Split(load_split(TRAIN_POS, TRAIN_NEG))
    val_ds   = Split(load_split(VAL_POS,   VAL_NEG))
    test_ds  = Split(load_split(TEST_POS,  TEST_NEG))

    model = Baseline2D(EMBED_DIM, h3=H3, kernel_size=KERNEL_SIZE,
                       pooling=POOLING, strict_original=STRICT_ORIGINAL).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_aupr, patience_cnt = 0.0, 0
    CKPT_2D.parent.mkdir(parents=True, exist_ok=True)
    if CKPT_2D.exists():
        model.load_state_dict(torch.load(CKPT_2D, map_location=DEVICE))
        print("已加载 checkpoint 续训\n")

    tag = "strict原文" if STRICT_ORIGINAL else "对齐版"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        idxs = stratified_subset(train_ds)

        ep_loss, nb = 0.0, 0
        for s in range(0, len(idxs), BATCH_SIZE):
            batch = train_ds[idxs[s:s + BATCH_SIZE]]
            preds = batch_iterate(model, batch, DEVICE)
            labels = torch.tensor(batch["interaction"],
                                  dtype=torch.float32, device=DEVICE)
            if s == 0:
                print(f"  [诊断] preds: min={preds.min():.4f} mean={preds.mean():.4f} "
                      f"max={preds.max():.4f} | ≥0.5占比={(preds >= 0.5).float().mean():.2f}")
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item(); nb += 1

        val = evaluate(model, val_ds)
        print(f"Epoch {epoch:03d} | loss={ep_loss/nb:.4f} | "
              f"val acc={val['acc']:.3f} f1={val['f1']:.3f} aupr={val['aupr']:.3f}")

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        if _PS is not None:
            print(f"  [MEM] 进程内存 {_PS.memory_info().rss/1024**3:.2f} GB")

        if val["aupr"] > best_aupr:
            best_aupr, patience_cnt = val["aupr"], 0
            torch.save(model.state_dict(), CKPT_2D)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"早停：验证AUPR {PATIENCE} 个 epoch 无改善，最优 aupr={best_aupr:.3f}")
                break

    if CKPT_2D.exists():
        model.load_state_dict(torch.load(CKPT_2D, map_location=DEVICE))
    test = evaluate(model, test_ds)
    print(f"\n===== Test (2d-baseline, {tag}) =====")
    for k, v in test.items():
        print(f"{k:>10s}: {v:.3f}")

    open_h5().close()


# ============================================================
# 5. 入口
# ============================================================
if __name__ == "__main__":
    if MODE == "rfc":
        run_rfc()
    elif MODE in ("2dbaseline", "2d", "baseline2d"):
        run_2dbaseline()
    else:
        print(f"未知模式 {MODE!r}，可用: rfc | 2dbaseline")
