# -*- coding: utf-8 -*-
"""
========================================================================
 2d-Selfattention —— Reim et al., Bioinformatics 2025 (btaf192) 复现
 本版整合：
   · Fix A   梯度裁剪 + 逐 epoch 诊断 + [MEM] 内存监控
   · Fix B   conv 后 InstanceNorm（train/eval 行为一致）
   · Fix B+  top-1% 均值读出（替代全局 max，梯度通路 ×100，无新参数）
   · Fix M   禁用 per-token 嵌入缓存（修复 OOM 与合并版首次调用崩溃）
   · 早停指标 = AUPR | PATIENCE=20 | checkpoint 备份 + 续训
========================================================================
"""
import re
import gc
import random
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score)

try:                                    # ★ 内存监控（未装 psutil 则跳过打印）
    import psutil
    _PS = psutil.Process()
except ImportError:
    _PS = None

# ============================================================
# 0. 配置
# ============================================================
BASE_DIR = Path(r"D:/pythonprojects/practice-github/PPI_prediction(gold-standard dataset)")

EMB_H5  = BASE_DIR / "Embeddings" / "embeddings_per_tok.h5"

TRAIN_POS = BASE_DIR / "dataset" / "Intra1_pos_rr.txt"
TRAIN_NEG = BASE_DIR / "dataset" / "Intra1_neg_rr.txt"
VAL_POS   = BASE_DIR / "dataset" / "Intra0_pos_rr.txt"
VAL_NEG   = BASE_DIR / "dataset" / "Intra0_neg_rr.txt"
TEST_POS  = BASE_DIR / "dataset" / "Intra2_pos_rr.txt"
TEST_NEG  = BASE_DIR / "dataset" / "Intra2_neg_rr.txt"

# ---- 模型超参数（论文 2d-Selfattention 设置）----
EMBED_DIM   = 1280
H3          = 64
NUM_HEADS   = 8
FF_DIM      = 256
DROPOUT     = 0.2
POOLING     = 'max'
KERNEL_SIZE = 2

# ---- 训练超参数 ----
LR          = 1e-4    # ★ 从 1e-5 调回 1e-4：B+ 需要把 InstanceNorm 的 β 移动约 2 个单位
                      #   重新居中，Adam 每 step 位移 ≈ lr —— 1e-5 要约 80 epoch，1e-4 约 8 epoch。
                      #   摆动风险由梯度裁剪兜底。
BATCH_SIZE  = 16
MAX_EPOCHS  = 60
PATIENCE    = 20      # ★ aupr 爬升慢，8 太紧（上次 epoch 26 就是被 8 错杀的）
SUBSET_FRAC = 0.5
MAX_LEN     = 1000
SEED        = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT   = BASE_DIR / "checkpoints" / "2d_selfattention_best.pt"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. 嵌入读取
# ============================================================
_H5_HANDLE = None
_KEYSET    = None
_KEYMAP    = {}
_EMB_CACHE = {}            # per-token 嵌入缓存（默认禁用）
_CACHE_MAX = 0             # ★ 0 = 禁用。8000 条 × ~2MB ≈ 16GB，正是上次 OOM 的元凶；
                           #   热点蛋白的重复读取交给操作系统页缓存。若 epoch 明显变慢，
                           #   折中可设 500（约 1GB），绝不要回到 8000。
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
    L = obj.shape[-2]
    _LEN_CACHE[name] = int(L)
    return _LEN_CACHE[name]


def get_embedding_per_tok(name: str, layer: int = -1) -> torch.Tensor:
    """读取单蛋白 per-token 嵌入，返回 (L, 1280) 张量"""
    if _CACHE_MAX > 0 and name in _EMB_CACHE:          # ★ 缓存禁用时直接读
        return _EMB_CACHE[name]

    f = open_h5()
    k = resolve_key(name)
    if k is None:
        raise KeyError(f"蛋白 {name!r} 在 h5 中找不到（已尝试 isoform/竖线等变体）")

    obj = f[k]
    if isinstance(obj, h5py.Group):
        ks = list(obj.keys())
        pick = str(layer) if str(layer) in ks else \
            sorted(ks, key=lambda x: int(x) if str(x).isdigit() else -1)[-1]
        emb = obj[pick][()]
    else:
        emb = obj[()]

    t = torch.from_numpy(np.array(emb, dtype=np.float32))   # ★ 免二次复制的写法
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)

    if _CACHE_MAX > 0:                                       # ★ 修复：缓存写入加保护
        if len(_EMB_CACHE) >= _CACHE_MAX:                    #   （合并版此处未加保护，
            _EMB_CACHE.pop(next(iter(_EMB_CACHE)))           #    _CACHE_MAX=0 时首次调用
        _EMB_CACHE[name] = t                                 #    即抛 StopIteration）
    return t


# ============================================================
# 2. 数据加载（原样保留，已在你的数据上验证可用）
# ============================================================
_ID_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?$"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$"
    r"|^ENS[A-Z]*\d+$"
)
_LABEL_KW  = {"interaction", "label", "target", "y", "class", "is_interaction"}
_HEADER_KW = _LABEL_KW | {"name1", "name2", "protein1", "protein2", "protein_a",
                          "protein_b", "uniprot1", "uniprot2", "id1", "id2",
                          "seq1", "seq2", "sequence_a", "sequence_b"}
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
        raise ValueError(f"无法解析文件 {path.name}（分隔符未知）")

    df = df[(df != "").any(axis=1)].reset_index(drop=True)
    ncols = df.shape[1]

    row0 = [str(v).strip().lower() for v in df.iloc[0].tolist()]
    row0_label_col = next((j for j, v in enumerate(row0) if v in _LABEL_KW), None)
    row0_kw_hits   = sum(1 for v in row0 if v in _HEADER_KW)
    row0_id_hits   = sum(1 for j, v in enumerate(row0)
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
        raise ValueError(f"{path.name}: 至少需要两列蛋白 ID，实际 {ncols} 列")

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

    print(f"[{path.name}] 解析 {len(out)} 行 | name1←col{name_cols[0]}, "
          f"name2←col{name_cols[1]}, "
          f"label←{'col'+str(label_col) if label_col is not None else f'默认{default_label}'}"
          + (" | ⚠ 首行按数据保留（半表头文件）" if first_row_is_data and row0_kw_hits else ""))
    print(f"  前2行预览: {out.head(2).values.tolist()}")
    return out


def load_split(pos_path: Path, neg_path: Path) -> pd.DataFrame:
    df_pos = _read_one_file(pos_path, default_label=1.0)
    df_neg = _read_one_file(neg_path, default_label=0.0)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    open_h5()
    keep, missing_prots = [], set()
    for i, (n1, n2) in enumerate(zip(df["name1"], df["name2"])):
        l1, l2 = protein_len(n1), protein_len(n2)
        if l1 is None: missing_prots.add(n1); continue
        if l2 is None: missing_prots.add(n2); continue
        if l1 <= MAX_LEN and l2 <= MAX_LEN:
            keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)

    n_pos = int(df["interaction"].sum()); n_neg = len(df) - n_pos
    print(f"[{pos_path.stem}+{neg_path.stem}] 样本={len(df)} "
          f"(正={n_pos}, 负={n_neg}, 占比={n_pos/len(df):.2%})，"
          f"缺失嵌入蛋白 {len(missing_prots)} 个已剔除")
    if missing_prots:
        print(f"  ⚠ 缺失示例: {sorted(missing_prots)[:5]}（检查 ID 格式与 h5 key）")
    if len(df) and not 0.45 <= n_pos / len(df) <= 0.55:
        print("  ⚠ 警告：正负比例明显偏离 1:1，请检查文件！")
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
    k_pos = max(1, int(frac * len(pos_idx)))
    k_neg = max(1, int(frac * len(neg_idx)))
    idxs = random.sample(pos_idx, k_pos) + random.sample(neg_idx, k_neg)
    random.shuffle(idxs)
    return idxs


# ============================================================
# 3. 模型
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
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(DEVICE)

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
        self.do = nn.Dropout(dropout)
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
    """★ 论文中的 2d-Selfattention 模型 ★"""
    def __init__(self, embed_dim, num_heads, h3=64, dropout=0.2,
                 ff_dim=256, pooling='avg', kernel_size=2):
        super().__init__()
        h  = int(embed_dim // 4)
        h2 = int(h // 4)

        self.encoder = EncoderLayer(h3, num_heads, ff_dim, dropout)
        self.multihead = Attention(h3, num_heads, dropout)
        self.conv = nn.Conv2d(h3, 1, kernel_size=kernel_size, padding='same')
        self.map_norm = nn.InstanceNorm2d(1, affine=True)   # Fix B
        if pooling == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)
        else:
            raise ValueError("pooling must be 'max' or 'avg'")

        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)
        self.fc2 = nn.Linear(h, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, protein1, protein2, mask1=None, mask2=None):
        x1 = protein1.to(torch.float32).unsqueeze(0)
        x2 = protein2.to(torch.float32).unsqueeze(0)
        x1 = self.ReLU(self.fc3(self.ReLU(self.fc2(self.ReLU(self.fc1(x1))))))
        x2 = self.ReLU(self.fc3(self.ReLU(self.fc2(self.ReLU(self.fc1(x2))))))

        x1 = self.encoder(x1, mask1)
        x2 = self.encoder(x2, mask2)

        mat = torch.einsum('bik,bjk->bijk', x1, x2)   # (1, L1, L2, 64)
        mat = mat.permute(0, 3, 1, 2)                 # (1, 64, L1, L2)
        mat = self.conv(mat)                          # (1, 1, L1, L2)
        mat = self.map_norm(mat)                      # Fix B
        x = self.pool(mat)
        # ---- ★ Fix B+：top-1% 均值读出（无新参数，与旧 checkpoint 兼容）----
        flat = x.flatten()
        k = max(1, int(0.01 * flat.numel()))
        m = torch.topk(flat, k).values.mean()
        pred = self.sigmoid(m)[None]
        return pred, mat


def batch_iterate(model, batch, device):
    """逐样本前向（变长不 padding），整批一次反向"""
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


def main():
    set_seed()
    print(f"Device: {DEVICE}\n")

    train_ds = Split(load_split(TRAIN_POS, TRAIN_NEG))
    val_ds   = Split(load_split(VAL_POS,   VAL_NEG))
    test_ds  = Split(load_split(TEST_POS,  TEST_NEG))

    probes = list(range(0, min(100, len(train_ds)), 7))
    ok = 0
    for i in probes:
        try:
            get_embedding_per_tok(train_ds.name1[i])
            get_embedding_per_tok(train_ds.name2[i])
            ok += 1
        except KeyError:
            pass
    print(f"\n自检: {ok}/{len(probes)} 条样本嵌入读取成功\n")

    model = SelfAttInteraction(EMBED_DIM, NUM_HEADS, h3=H3, dropout=DROPOUT,
                               ff_dim=FF_DIM, pooling=POOLING,
                               kernel_size=KERNEL_SIZE).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_aupr, patience_cnt = 0.0, 0
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    # ---- ★ 续训：top-k 读出无新参数，与上一轮最优 checkpoint 完全兼容 ----
    if CKPT.exists():
        backup = CKPT.with_name(CKPT.stem + "_maxreadout_backup.pt")   # ★ 用 with_name，
        shutil.copy(CKPT, backup)                                      #   Path.replace 会真的重命名文件
        print(f"已备份 max 读出最优权重 → {backup.name}")
        try:
            model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
            print("已加载 checkpoint，以 top-1% 读出续训\n")
        except RuntimeError as e:
            print(f"checkpoint 结构不匹配，改为从头训练（{e}）\n")

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
                print(f"  [诊断] preds: min={preds.min():.4f} mean={preds.mean():.4f} "
                      f"max={preds.max():.4f} | ≥0.5占比={(preds >= 0.5).float().mean():.2f} | "
                      f"labels[:8]={labels[:8].tolist()}")
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if s == 0:
                g = sum(p.grad.abs().sum().item()
                        for p in model.parameters() if p.grad is not None)
                print(f"  [诊断] 梯度绝对值总和 = {g:.6f}")
            optimizer.step()
            ep_loss += loss.item(); nb += 1

        val = evaluate(model, val_ds)
        print(f"Epoch {epoch:03d} | loss={ep_loss/nb:.4f} | "
              f"val acc={val['acc']:.3f} f1={val['f1']:.3f} aupr={val['aupr']:.3f}")

        # ---- ★ Fix M 验证：内存应平稳在 2-4 GB，不随 epoch 单调上涨 ----
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        if _PS is not None:
            print(f"  [MEM] 进程内存 {_PS.memory_info().rss/1024**3:.2f} GB")

        if val["aupr"] > best_aupr:
            best_aupr, patience_cnt = val["aupr"], 0
            torch.save(model.state_dict(), CKPT)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"早停：验证AUPR {PATIENCE} 个 epoch 无改善，最优 aupr={best_aupr:.3f}")  # ★
                break

    if CKPT.exists():                                   # ★ 兜底：未保存过也不崩
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    else:
        print("⚠ 训练中未保存任何 checkpoint，用最后一个 epoch 的权重测试")
    test = evaluate(model, test_ds)
    print("\n===== Test (2d-Selfattention, top-1% readout) =====")
    for k, v in test.items():
        print(f"{k:>10s}: {v:.3f}")

    open_h5().close()


if __name__ == "__main__":
    main()
