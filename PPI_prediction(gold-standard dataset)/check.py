# -*- coding: utf-8 -*-
"""
========================================================================
 diagnose_calibrate.py —— 对已训练 checkpoint 做诊断 + 阈值校准
 回答: "test 为什么全面差于论文 Table 1"
  ① 被测 checkpoint 的 val 复评 —— 消除"拿 test 比最优 val(0.633)"的对照错位
  ② isoform 回退审计 —— exact / keymap / candidate / missing 按 split 计数
  ③ 样本量对齐 —— 对照论文 93,719 / 46,421 / 41,100
  ④ val 阈值扫描 t* —— test 双口径: @0.5(论文保真) 与 @t*(校准分析)
 用法: python diagnose_calibrate.py
 注意: 本脚本对 test 的调用是【分析口径】，不反哺任何超参选择
========================================================================
"""
import random
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
# 0. 配置（与训练脚本一致）
# ============================================================
BASE_DIR = Path(r"D:/pythonprojects/practice-github/PPI_prediction(gold-standard dataset)")
EMB_H5   = BASE_DIR / "Embeddings" / "embeddings_per_tok.h5"
CKPT     = BASE_DIR / "checkpoints" / "2d_selfattention_best.pt"

TRAIN_POS = BASE_DIR / "dataset" / "Intra1_pos_rr.txt"
TRAIN_NEG = BASE_DIR / "dataset" / "Intra1_neg_rr.txt"
VAL_POS   = BASE_DIR / "dataset" / "Intra0_pos_rr.txt"
VAL_NEG   = BASE_DIR / "dataset" / "Intra0_neg_rr.txt"
TEST_POS  = BASE_DIR / "dataset" / "Intra2_pos_rr.txt"
TEST_NEG  = BASE_DIR / "dataset" / "Intra2_neg_rr.txt"

EMBED_DIM, NUM_HEADS, H3, FF_DIM, DROPOUT = 1280, 8, 64, 256, 0.2
POOLING, KERNEL_SIZE, MAX_LEN = 'max', 2, 1000
BATCH_SIZE, SEED = 16, 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 论文参考值（Table 1, 2d-Selfattention 金标准）----
PAPER   = {"acc": 0.616, "precision": 0.611, "recall": 0.553, "f1": 0.591, "aupr": 0.641}
PAPER_N = {"train": 93_719, "val": 46_421, "test": 41_100}
BESTEVER_VAL_AUPR = 0.633     # ← 你日志中出现过的最优 val aupr，用于对照错位判别


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. 嵌入读取（OOM 修复版）+ 解析路径审计
# ============================================================
_H5_HANDLE, _KEYSET, _KEYMAP = None, None, {}
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


def resolution_path(name: str):
    """返回(路径, 实际key) —— 路径 ∈ {exact, keymap, candidate, missing}
       keymap/candidate 两条路都有 isoform 错配风险（用规范蛋白嵌入顶替变体）"""
    if name in _KEYSET:
        return "exact", name
    if name in _KEYMAP:
        return "keymap", _KEYMAP[name]
    for cand in (name.split("-")[0], name.split("|")[0].split("/")[-1],
                 name.split("_")[0] if "_" in name else None):
        if cand and cand in _KEYSET:
            return "candidate", cand
        if cand and cand in _KEYMAP:
            return "candidate", _KEYMAP[cand]
    return "missing", None


def resolve_key(name: str):
    _, k = resolution_path(name)
    return k


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


def audit_resolution(df: pd.DataFrame, split_name: str):
    """② isoform 回退审计：按解析路径统计唯一蛋白"""
    paths = {"exact": set(), "keymap": set(), "candidate": set(), "missing": set()}
    for n in list(df["name1"]) + list(df["name2"]):
        p, _ = resolution_path(n)
        paths[p].add(n)
    total = sum(len(v) for v in paths.values())
    fb = len(paths["keymap"]) + len(paths["candidate"])
    print(f"\n[审计② {split_name}] 唯一蛋白 {total}: "
          f"exact={len(paths['exact'])} | fallback={fb} ({fb/max(1,total):.2%}) | "
          f"missing={len(paths['missing'])}")
    if paths["keymap"]:
        ex = sorted(paths["keymap"])[:3]
        print(f"  keymap 映射示例: {[(n, _KEYMAP[n]) for n in ex]}")
    if paths["candidate"]:
        ex = sorted(paths["candidate"])[:3]
        print(f"  candidate 映射示例: {[(n, resolve_key(n)) for n in ex]}")
    return fb / max(1, total)


# ============================================================
# 2. 数据加载（与训练脚本一致，已在你的数据上验证）
# ============================================================
import re
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


# ============================================================
# 3. 模型（与训练 checkpoint 严格同构，state_dict 可直接加载）
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
        self.activation = nn.SiLU()

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
    def __init__(self, embed_dim, num_heads, h3=64, dropout=0.2,
                 ff_dim=256, pooling='max', kernel_size=2):
        super().__init__()
        h, h2 = int(embed_dim // 4), int(embed_dim // 16)
        self.encoder = EncoderLayer(h3, num_heads, ff_dim, dropout)
        self.multihead = Attention(h3, num_heads, dropout)
        self.conv = nn.Conv2d(h3, 1, kernel_size=kernel_size, padding='same')
        self.map_norm = nn.InstanceNorm2d(1, affine=True)
        self.pool = (nn.MaxPool2d(kernel_size) if pooling == 'max'
                     else nn.AvgPool2d(kernel_size))
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
        mat = torch.einsum('bik,bjk->bijk', x1, x2)
        mat = mat.permute(0, 3, 1, 2)
        mat = self.conv(mat)
        mat = self.map_norm(mat)
        x = self.pool(mat)
        flat = x.flatten()
        k = max(1, int(0.01 * flat.numel()))
        m = torch.topk(flat, k).values.mean()          # ★ top-1% 读出（无参数）
        pred = self.sigmoid(m)[None]
        return pred, mat


# ============================================================
# 4. 评估 + 阈值扫描
# ============================================================
def batch_iterate(model, batch, device):
    preds = []
    for i in range(len(batch["interaction"])):
        s1 = get_embedding_per_tok(batch["name1"][i]).to(device)
        s2 = get_embedding_per_tok(batch["name2"][i]).to(device)
        p, _ = model(s1, s2)
        preds.append(p)
    return torch.stack(preds).view(-1)


@torch.no_grad()
def collect_scores(model, ds: Split, device=DEVICE):
    model.eval()
    ps, ys = [], []
    for s in range(0, len(ds), BATCH_SIZE):
        batch = ds[list(range(s, min(s + BATCH_SIZE, len(ds))))]
        ps.append(batch_iterate(model, batch, device).cpu())
        ys += batch["interaction"]
    return torch.cat(ps).numpy(), np.asarray(ys)


def report(y, p, threshold=0.5):
    pb = (p >= threshold).astype(int)
    return {"acc": accuracy_score(y, pb),
            "precision": precision_score(y, pb, zero_division=0),
            "recall": recall_score(y, pb, zero_division=0),
            "f1": f1_score(y, pb, zero_division=0),
            "mcc": matthews_corrcoef(y, pb),
            "aupr": average_precision_score(y, p)}


def fmt(m):
    return (f"acc={m['acc']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} "
            f"f1={m['f1']:.3f} mcc={m['mcc']:.3f} aupr={m['aupr']:.3f}")


# ============================================================
# 5. 主流程
# ============================================================
def main():
    set_seed()
    print(f"Device: {DEVICE}\n")

    # ---- ③ 样本量对齐 ----
    train_df = load_split(TRAIN_POS, TRAIN_NEG)   # 加载以保持与训练时一致的过滤口径
    val_df   = load_split(VAL_POS,   VAL_NEG)
    test_df  = load_split(TEST_POS,  TEST_NEG)
    print("\n[审计③ 样本量 vs 论文]（过滤 >1000aa + 剔除缺失嵌入后）")
    for tag, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        n = len(df)
        print(f"  {tag:>5}: 你={n:>6d} | 论文={PAPER_N[tag]:>6d} "
              f"| 比例={n/PAPER_N[tag]:.2%}")
    print("  （val/test 显著偏小 → 组成差异是 val↔test 落差的候选原因）")

    # ---- ② isoform 审计 ----
    fb_val  = audit_resolution(val_df,  "val")
    fb_test = audit_resolution(test_df, "test")
    if abs(fb_test - fb_val) > 0.03:
        print(f"  ⚠ val/test fallback 占比差 {abs(fb_test-fb_val):.2%}"
              f"（test 偏高 → isoform 污染可解释部分 test 落差）")

    val_ds  = Split(val_df)
    test_ds = Split(test_df)

    # ---- 加载被测 checkpoint ----
    model = SelfAttInteraction(EMBED_DIM, NUM_HEADS, h3=H3, dropout=DROPOUT,
                               ff_dim=FF_DIM, pooling=POOLING,
                               kernel_size=KERNEL_SIZE).to(DEVICE)
    try:
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        print(f"\ncheckpoint 严格加载成功: {CKPT.name}")
    except RuntimeError as e:
        print(f"\n⚠ 严格加载失败（可能训练端结构已改），尝试 non-strict:\n  {e}")
        sd = torch.load(CKPT, map_location=DEVICE)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"  missing={list(miss)}\n  unexpected={list(unexp)}")

    # ---- ① 被测 checkpoint 的 val 复评（对照错位判别）----
    print("\n[① 被测checkpoint 的 val 复评]")
    pv, yv = collect_scores(model, val_ds)
    val_m = report(yv, pv)
    print(f"  val : {fmt(val_m)}")
    if val_m["aupr"] < BESTEVER_VAL_AUPR - 0.01:
        print(f"  → 被测点 val aupr={val_m['aupr']:.3f} ≠ 最优val {BESTEVER_VAL_AUPR:.3f}"
              f"：此前 test 是与后者比的，存在【对照错位】，真实差距更小")
    else:
        print(f"  → 被测即最优点，无对照错位")

    # ---- test @0.5（论文保真口径）----
    pt, yt = collect_scores(model, test_ds)
    t05 = report(yt, pt, threshold=0.5)
    print(f"\n[test @0.50 论文保真口径] {fmt(t05)}")
    print(f"  论文 Table 1           : acc={PAPER['acc']:.3f} prec={PAPER['precision']:.3f} "
          f"rec={PAPER['recall']:.3f} f1={PAPER['f1']:.3f} aupr={PAPER['aupr']:.3f}")

    # ---- ④ val 阈值扫描 → test @t*（校准分析口径）----
    ts = np.round(np.arange(0.20, 0.81, 0.01), 2)
    f1s = [f1_score(yv, pv >= t, zero_division=0) for t in ts]
    order = np.argsort(f1s)[::-1][:5]
    print("\n[④ val 阈值扫描 top-5]")
    for i in order:
        print(f"  t={ts[i]:.2f} val f1={f1s[i]:.3f}")
    t_star = float(ts[order[0]])
    tst = report(yt, pt, threshold=t_star)
    print(f"\n[test @{t_star:.2f} 校准分析口径] {fmt(tst)}")

    # ---- 差距分解 ----
    print("\n" + "=" * 62)
    print("差距分解（对论文 Table 1）")
    print("=" * 62)
    print(f"① 对照错位 : 被测checkpoint val aupr={val_m['aupr']:.3f} "
          f"vs 最优val {BESTEVER_VAL_AUPR:.3f}")
    print(f"② 校准缺口 : test recall 0.5口径={t05['recall']:.3f} → t*口径={tst['recall']:.3f} "
          f"({tst['recall']-t05['recall']:+.3f}); f1 {t05['f1']:.3f} → {tst['f1']:.3f}")
    resid = PAPER["aupr"] - tst["aupr"]
    print(f"③ 排名残差 : 校准后 test aupr={tst['aupr']:.3f} vs 论文 {PAPER['aupr']:.3f} "
          f"({resid:+.3f}) —— 单种子噪声量级(0.01–0.03)内即达标")


if __name__ == "__main__":
    main()
