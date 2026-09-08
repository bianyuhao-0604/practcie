"""
2d-Selfattention 批处理完整版 v3 —— Reim et al. 2025 (btaf192) 协议复现
====================================================================
v3 变化:
  ★ 粗化分桶 (50aa 一档): 解决 (L1,L2) 精确分桶退化 batch=1 的问题
  ★ BUDGET=2M (8GB 显存安全水位)
  ★ 冒烟模式 SMOKE_N=200: 统计 batch 分布后直接退出 (验证完设 0)
  ★ torch.load weights_only=True (消除 FutureWarning)
  ★ conv 偶数 kernel 手动 pad (消除慢路径)
  ★ RAM fp16 缓存 7GB + LRU + h5py Dataset 句柄缓存 (I/O 已验证 = 0s)
  ★ 预热 train 子集蛋白 / TF32 / AMP / 后台预取线程 / 每 epoch 计时
协议: LR=1e-5 | 早停=val acc, patience=8 | 50% 子集 | 全局 max 读出 | 无 InstanceNorm
"""
import gc, json, random, re, threading, queue, time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, matthews_corrcoef)

# ============================================================
# 0. 配置
# ============================================================
BASE_DIR = Path(r"D:/pythonprojects/practice-github/PPI_prediction(gold-standard dataset)")
EMB_H5   = BASE_DIR / "Embeddings" / "embeddings_per_tok.h5"
SPLITS = {"train": ("Intra1_pos_rr.txt", "Intra1_neg_rr.txt"),
          "val":   ("Intra0_pos_rr.txt", "Intra0_neg_rr.txt"),
          "test":  ("Intra2_pos_rr.txt", "Intra2_neg_rr.txt")}
CKPT   = BASE_DIR / "checkpoints" / "2d_selfattention_v2.pt"
RESUME = True

EMBED_DIM, H3, NUM_HEADS, FF_DIM, DROPOUT = 1280, 64, 8, 256, 0.2
KERNEL_SIZE, LR, GRAD_CLIP = 2, 1e-5, 1.0
MAX_EPOCHS, PATIENCE, SUBSET_FRAC, MAX_LEN, SEED = 60, 8, 0.5, 1000, 42
BUDGET   = 3_000_000                # ★ 提到 2M; OOM 则退回 1_500_000
RAM_CAP  = int(7e9)                 # RAM 缓存上限 (防 Windows 换页)
USE_AMP  = True
SMOKE_N  = 0                      # ★ 冒烟: 跑 200 batch 看统计后退出; 确认后改 0
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ============================================================
# 1. 嵌入读取: 句柄缓存 + RAM fp16 缓存 + LRU
# ============================================================
_H5, _KEYSET, _KEYMAP, _LEN = None, set(), {}, {}
_DS   = {}
_RAM, _RAM_BYTES = {}, 0

def _cands(n):
    c = [n]
    for sep in "-_|/":
        if sep in n: c.append(n.split(sep)[0])
    return c

def open_h5():
    global _H5
    if _H5 is None:
        print(f"打开嵌入文件: {EMB_H5}")
        _H5 = h5py.File(EMB_H5, "r")
        for k in _H5.keys():
            k = k.decode() if isinstance(k, bytes) else k
            _KEYSET.add(k)
            for c in _cands(k):
                if c not in _KEYSET and c not in _KEYMAP: _KEYMAP[c] = k
        print(f"h5 中共 {len(_KEYSET)} 个蛋白")
    return _H5

def resolve(n):
    if n in _KEYSET: return n
    if n in _KEYMAP: return _KEYMAP[n]
    for c in _cands(n):
        if c in _KEYSET: return c
        if c in _KEYMAP: return _KEYMAP[c]
    return None

def _dataset(k):
    d = _DS.get(k)
    if d is None:
        obj = open_h5()[k]
        if isinstance(obj, h5py.Group):
            obj = obj[list(obj.keys())[0]]
        _DS[k] = d = obj
    return d

def plen(n):
    if n not in _LEN:
        k = resolve(n)
        _LEN[n] = None if k is None else int(_dataset(k).shape[-2])
    return _LEN[n]

def emb(n):
    global _RAM_BYTES
    k = resolve(n)
    if k is None: raise KeyError(n)
    t = _RAM.get(k)
    if t is None:
        t = torch.from_numpy(np.asarray(_dataset(k)[()], dtype=np.float16))
        sz = t.numel() * 2
        if _RAM_BYTES + sz > RAM_CAP:
            while _RAM_BYTES + sz > RAM_CAP and _RAM:
                k0, t0 = next(iter(_RAM.items()))
                _RAM_BYTES -= t0.numel() * 2
                del _RAM[k0]
        if _RAM_BYTES + sz <= RAM_CAP:
            _RAM[k] = t; _RAM_BYTES += sz
    return t

# ============================================================
# 2. 数据加载 (与已验证版本一致)
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
    df = df.reset_index(drop=True)
    ncols = df.shape[1]
    row0 = [str(v).strip().lower() for v in df.iloc[0].tolist()]
    row0_label_col = next((j for j, v in enumerate(row0) if v in _LABEL_KW), None)
    row0_kw_hits = sum(1 for v in row0 if v in _HEADER_KW)
    row0_id_hits = sum(1 for j, v in enumerate(row0)
                       if j != row0_label_col and _ID_RE.match(v.upper()))
    if row0_label_col is not None or row0_kw_hits >= 2:
        label_col = row0_label_col
        if label_col is None:
            label_col = _find_binary_col(df.iloc[1:])
        first_row_is_data = row0_id_hits > 0
        data = df if first_row_is_data else df.iloc[1:]
    else:
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
        l1, l2 = plen(n1), plen(n2)
        if l1 is None: missing.add(n1); continue
        if l2 is None: missing.add(n2); continue
        if l1 <= MAX_LEN and l2 <= MAX_LEN: keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    n_pos = int(df["interaction"].sum()); n_neg = len(df) - n_pos
    print(f"[{split_name}] 样本={len(df)} (正={n_pos}, 负={n_neg}, "
          f"占比={n_pos/len(df):.2%}), 缺失嵌入剔除 {len(missing)} 个蛋白")
    return df

class Split:
    def __init__(self, df: pd.DataFrame):
        self.name1 = df["name1"].astype(str).tolist()
        self.name2 = df["name2"].astype(str).tolist()
        self.interaction = df["interaction"].astype(float).tolist()
    def __len__(self): return len(self.interaction)
    def __getitem__(self, idxs):
        return {"name1": [self.name1[i] for i in idxs],
                "name2": [self.name2[i] for i in idxs],
                "interaction": [self.interaction[i] for i in idxs]}

def stratified_subset(ds: Split, frac=SUBSET_FRAC):
    pos_idx = [i for i, y in enumerate(ds.interaction) if y == 1.0]
    neg_idx = [i for i, y in enumerate(ds.interaction) if y == 0.0]
    idxs = (random.sample(pos_idx, max(1, int(frac * len(pos_idx))))
            + random.sample(neg_idx, max(1, int(frac * len(neg_idx)))))
    random.shuffle(idxs)
    return idxs

# ============================================================
# 3. 模型 (模块名与 checkpoint 兼容)
# ============================================================
class Attention(nn.Module):
    def __init__(self, hid, nh, do):
        super().__init__()
        self.hid, self.nh = hid, nh
        self.w_q = spectral_norm(nn.Linear(hid, hid))
        self.w_k = spectral_norm(nn.Linear(hid, hid))
        self.w_v = spectral_norm(nn.Linear(hid, hid))
        self.fc  = spectral_norm(nn.Linear(hid, hid))
        self.do  = nn.Dropout(do)
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([hid // nh])))
    def forward(self, q, k, v, mask=None):
        B = q.shape[0]; d = self.hid // self.nh
        Q = self.w_q(q).view(B, -1, self.nh, d).permute(0, 2, 1, 3)
        K = self.w_k(k).view(B, -1, self.nh, d).permute(0, 2, 1, 3)
        V = self.w_v(v).view(B, -1, self.nh, d).permute(0, 2, 1, 3)
        e = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None: e = e.masked_fill(mask == 0, -1e10)
        x = torch.matmul(self.do(F.softmax(e, -1)), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, -1, self.hid)
        return self.fc(x)

class Feedforward(nn.Module):
    def __init__(self, hid, ff, do):
        super().__init__()
        self.fc_1 = spectral_norm(nn.Linear(hid, ff))
        self.fc_2 = spectral_norm(nn.Linear(ff, hid))
        self.do, self.act = nn.Dropout(do), nn.SiLU()
    def forward(self, x):
        return self.fc_2(self.do(self.act(self.fc_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, hid, nh, ff, do):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(hid), nn.LayerNorm(hid)
        self.do1, self.do2 = nn.Dropout(do), nn.Dropout(do)
        self.sa, self.ff = Attention(hid, nh, do), Feedforward(hid, ff, do)
    def forward(self, x, mask=None):
        x = self.ln1(x + self.do1(self.sa(x, x, x, mask)))
        return self.ln2(x + self.do2(self.ff(x)))

class SelfAttInteraction(nn.Module):
    def __init__(self, embed_dim, num_heads, h3=64, dropout=0.2,
                 ff_dim=256, kernel_size=2):
        super().__init__()
        h, h2 = embed_dim // 4, embed_dim // 16
        self.encoder = EncoderLayer(h3, num_heads, ff_dim, dropout)
        self.conv = nn.Conv2d(h3, 1, kernel_size, padding=0)  # 手动 pad 替代 'same'
        self.pool = nn.MaxPool2d(kernel_size)
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)
        self.fc2 = nn.Linear(h, h2)
        self.fc3 = nn.Linear(h2, h3)
    def _enc(self, x, m):
        x = self.ReLU(self.fc3(self.ReLU(self.fc2(self.ReLU(self.fc1(x))))))
        am = (m[:, :, None] * m[:, None, :]).unsqueeze(1)
        return self.encoder(x, am)
    def forward(self, x1, x2, m1, m2):
        x1 = self._enc(x1.float(), m1)
        x2 = self._enc(x2.float(), m2)
        mat = torch.einsum('bik,bjk->bijk', x1, x2)
        c = self.conv(F.pad(mat.permute(0, 3, 1, 2), (0, 1, 0, 1)))  # (B,1,L1,L2)
        c = c.masked_fill((m1[:, None, :, None] * m2[:, None, None, :]) == 0, -1e4)
        return self.pool(c).amax(dim=(1, 2, 3)), c

# ============================================================
# 4. 粗化分桶组批 + 后台预取
# ============================================================
def epoch_batches(ds, idxs, budget=BUDGET):
    """按 (L1,L2) 各自向上取整到 50 的倍数分桶:
    桶内形状一致 → 真批处理; make_batch 按桶内实际 max padding"""
    buckets = {}
    for i in idxs:
        l1, l2 = plen(ds.name1[i]), plen(ds.name2[i])
        key = ((l1 + 99) // 100, (l2 + 99) // 100)
        buckets.setdefault(key, []).append(i)
    batches = []
    for key, members in buckets.items():
        tokens = key[0] * 100 * key[1] * 100      # 桶上限的保守 token 数
        B = max(1, budget // tokens)
        for s in range(0, len(members), B):
            batches.append(members[s:s+B])
    random.shuffle(batches)
    return batches

def make_batch(ds, idxs):
    e1s = [emb(ds.name1[i]) for i in idxs]
    e2s = [emb(ds.name2[i]) for i in idxs]
    y = torch.tensor([ds.interaction[i] for i in idxs], dtype=torch.float32)
    L1, L2 = max(e.shape[0] for e in e1s), max(e.shape[0] for e in e2s)
    D, B = e1s[0].shape[1], len(idxs)
    x1 = torch.zeros(B, L1, D); x2 = torch.zeros(B, L2, D)
    m1 = torch.zeros(B, L1);    m2 = torch.zeros(B, L2)
    for j, (a, b) in enumerate(zip(e1s, e2s)):
        x1[j, :a.shape[0]] = a; x2[j, :b.shape[0]] = b
        m1[j, :a.shape[0]] = 1; m2[j, :b.shape[0]] = 1
    return x1, x2, m1, m2, y

class Prefetch:
    def __init__(self, ds, batches, depth=3):
        self.q, self.stop = queue.Queue(maxsize=depth), False
        def work():
            for b in batches:
                if self.stop: return
                self.q.put(make_batch(ds, b))
            self.q.put(None)
        threading.Thread(target=work, daemon=True).start()
    def __iter__(self): return self
    def __next__(self):
        it = self.q.get()
        if it is None: raise StopIteration
        return it
    def close(self): self.stop = True

# ============================================================
# 5. 评估 (AMP 加速; 指标与 fp32 差异 < 0.005)
# ============================================================
@torch.no_grad()
def evaluate(model, ds):
    model.eval(); P, Y = [], []
    for b in epoch_batches(ds, list(range(len(ds)))):
        x1, x2, m1, m2, y = make_batch(ds, b)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            lg, _ = model(x1.to(DEVICE), x2.to(DEVICE),
                          m1.to(DEVICE), m2.to(DEVICE))
        P.append(torch.sigmoid(lg).float().cpu()); Y += y.tolist()
    model.train()
    p, y = torch.cat(P).numpy(), np.asarray(Y)
    pb = (p >= 0.5).astype(int)
    return {"acc": accuracy_score(y, pb),
            "precision": precision_score(y, pb, zero_division=0),
            "recall": recall_score(y, pb, zero_division=0),
            "f1": f1_score(y, pb, zero_division=0),
            "aupr": average_precision_score(y, p),
            "mcc": matthews_corrcoef(y, pb)}

# ============================================================
# 6. 主流程
# ============================================================
def main():
    set_seed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"协议: LR={LR} | 早停=val acc, patience={PATIENCE} | 子集={SUBSET_FRAC} "
          f"| AMP={USE_AMP} | BUDGET={BUDGET} | RAM缓存={RAM_CAP/1e9:.0f}GB")
    print(f"SMOKE_N={SMOKE_N} (>0 为冒烟模式) | RESUME={RESUME}\n")

    train_ds = Split(load_split(BASE_DIR/"dataset"/SPLITS["train"][0],
                                BASE_DIR/"dataset"/SPLITS["train"][1], "train"))
    val_ds   = Split(load_split(BASE_DIR/"dataset"/SPLITS["val"][0],
                                BASE_DIR/"dataset"/SPLITS["val"][1], "val"))
    test_ds  = Split(load_split(BASE_DIR/"dataset"/SPLITS["test"][0],
                                BASE_DIR/"dataset"/SPLITS["test"][1], "test"))

    model = SelfAttInteraction(EMBED_DIM, NUM_HEADS, H3, DROPOUT,
                               FF_DIM, KERNEL_SIZE).to(DEVICE)
    if RESUME and CKPT.exists():
        model.load_state_dict(
            torch.load(CKPT, map_location=DEVICE, weights_only=True))  # ★ 消除警告
        print(f"已热启动: {CKPT.name}\n")
    else:
        print("从头训练\n")

    # ===== 预热 train 子集蛋白 =====
    warm = set()
    for i in stratified_subset(train_ds):
        warm.add(train_ds.name1[i]); warm.add(train_ds.name2[i])
    print(f"预热: 读取 {len(warm)} 个蛋白...")
    for i, n in enumerate(sorted(warm)):
        try: emb(n)
        except KeyError: pass
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{len(warm)}  缓存 {_RAM_BYTES/1e9:.1f}GB")
    print(f"预热完成, 缓存 {_RAM_BYTES/1e9:.1f}GB / {RAM_CAP/1e9:.0f}GB\n")

    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best, wait = 0.0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        ts = time.time()
        pf = Prefetch(train_ds, epoch_batches(train_ds,
                                              stratified_subset(train_ds)))
        tot, n = 0.0, 0
        t_data = t_gpu = 0.0
        batch_sizes, first = [], True
        for x1, x2, m1, m2, y in pf:
            t0 = time.time()
            x1, x2, m1, m2, y = (t.to(DEVICE, non_blocking=True)
                                 for t in (x1, x2, m1, m2, y))
            t1 = time.time()
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                lg, _ = model(x1, x2, m1, m2)
                loss = crit(lg, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
            t2 = time.time()
            t_data += t1 - t0; t_gpu += t2 - t1
            tot += loss.item(); n += 1
            if n % 100 == 0:
               torch.cuda.empty_cache()

            batch_sizes.append(len(y))
            if first:
                first = False
                pp = torch.sigmoid(lg)
                print(f"  [诊断] batch={len(y)} | preds: "
                      f"min={pp.min():.3f} mean={pp.mean():.3f} max={pp.max():.3f}")
            if n % 500 == 0:
                print(f"    batch {n} | loss={tot/n:.4f} | "
                      f"平均batch={np.mean(batch_sizes):.1f} | GPU={t_gpu:.0f}s")
            if SMOKE_N and n >= SMOKE_N:
                pf.close()
                print(f"\n[冒烟] {n} batch | 平均={np.mean(batch_sizes):.1f} "
                      f"p50={np.percentile(batch_sizes,50):.0f} "
                      f"p90={np.percentile(batch_sizes,90):.0f} "
                      f"最大={max(batch_sizes)}")
                print(f"[冒烟] 数据={t_data:.0f}s GPU={t_gpu:.0f}s")
                print("[冒烟] 若平均≥8: 把 SMOKE_N 改为 0 跑全量")
                return
        pf.close()

        v = evaluate(model, val_ds)
        print(f"Epoch {ep:03d} | loss={tot/n:.4f} | val acc={v['acc']:.3f} "
              f"f1={v['f1']:.3f} aupr={v['aupr']:.3f} mcc={v['mcc']:.3f} | "
              f"batch均值={np.mean(batch_sizes):.1f} | "
              f"数据={t_data:.0f}s GPU={t_gpu:.0f}s 总={time.time()-ts:.0f}s")
        gc.collect(); torch.cuda.empty_cache()

        if v["acc"] > best:
            best, wait = v["acc"], 0
            torch.save(model.state_dict(), CKPT)
            print(f"  → 新最优 val acc={best:.3f}, 已保存")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"早停: 最优 val acc={best:.3f}"); break

    model.load_state_dict(
        torch.load(CKPT, map_location=DEVICE, weights_only=True))
    t = evaluate(model, test_ds)
    print("\n===== Test (批处理版, 论文协议) =====")
    print("论文 Table 1: acc=0.616 prec=0.611 rec=0.553 f1=0.591 aupr=0.641")
    print("-" * 55)
    for k, v in t.items(): print(f"{k:>10s}: {v:.3f}")

if __name__ == "__main__":
    main()
