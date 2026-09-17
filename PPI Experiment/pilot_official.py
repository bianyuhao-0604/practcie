#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pilot_official.py v3.4 — 官方划分 Pilot：Baseline vs Level 3 (无 ESM 版)
================================================================
数据: Intra1(train) / Intra0(val) / Intra2(test), 含官方负样本
模型: Baseline (ESM-2 mean-pool + MLP) vs Level 3 (pLDDT mask + 局部attention)

历史修复:
  [FIX-1] 采样前过滤"缺序列蛋白"的对        [FIX-2] 池化只在有效残基上求均值
  [FIX-3] 剥离 ESM-2 的 BOS/EOS            [FIX-4] masked_pool 回退不含 padding
  [FIX-5] 缓存键含 max_len + 形状校验       [FIX-6] attention 块输出清零 padding
  [NEW-1] --split-mode clean (缺失回退)     [NEW-2/3] 文件名与 JSON 记 split_mode
v3.3 (依据 embed_diag: label=accession, 15% 截断@1022):
  [NEW-7]  去除 ESM 兜底; 加载失败蛋白 → FAILED → 过滤样本对
  [NEW-8]  长度规则 v3: 0<rows<Ls 视为提取截断, 安全接受
  [NEW-9]  缓存校验放宽 (兼容截断)           [NEW-10] 来源统计
v3.4 (依据 label_probe: 截断蛋白 rows 集中在 1022 = ESM 位置上限):
  [NEW-11] 提取惯例自动检测 (启动扫 ~80 个 .pt): d=n-Ls 分布 +
          mean_representations 数值校验 (判别力 4 个数量级) → bos/none/unknown
  [NEW-12] 截断蛋白在 'bos' 惯例下剥掉 t[0]=BOS (官方 extract.py 的
          token 截断保留 BOS、切掉 EOS → 否则 embedding 与 pLDDT mask
          整体错位 1 位, 影响约 14% 蛋白); 'unknown' 时原样并告警

运行:
  python pilot_official.py --n-train 500 --n-val 200 --n-test 300 --seed 42
"""
import sys, json, random, argparse, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
AF_OUTPUT  = Path("af_output")
META_FILE  = AF_OUTPUT / "metadata" / "monomer_metadata.json"
PLDDT_DIR  = AF_OUTPUT / "monomers"
CACHE_DIR  = Path("pilot_cache_v2"); CACHE_DIR.mkdir(exist_ok=True)

PT_EMB_DIR = Path("output_embeddings")     # label_probe 验证过的目录

EMB_DIM    = 1280
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLDDT_IF_THRESHOLD = 70.0

SOURCE = {"npy": 0, "pt": 0, "fail": 0}


# ============================================================
# [NEW-1] split 文件选择（official / clean + 缺失回退）
# ============================================================
def get_split_files(mode):
    suffix = "_clean" if mode == "clean" else ""
    files = {
        "train": [(f"Intra1_pos_rr{suffix}.txt", 1), (f"Intra1_neg_rr{suffix}.txt", 0)],
        "val":   [(f"Intra0_pos_rr{suffix}.txt", 1), (f"Intra0_neg_rr{suffix}.txt", 0)],
        "test":  [(f"Intra2_pos_rr{suffix}.txt", 1), (f"Intra2_neg_rr{suffix}.txt", 0)],
    }
    if mode == "clean":
        missing = [fn for fl in files.values() for fn, _ in fl
                   if not Path(fn).exists()]
        if missing:
            print(f"[WARN] clean 切分文件缺失 {len(missing)} 个: {missing[:3]}")
            print("       → 回退 official 切分 (先运行 make_clean_splits.py 生成)")
            return get_split_files("official")[0], "official(fallback)"
    return files, mode


# ============================================================
# 1. 官方数据加载
# ============================================================
def parse_pairs(path: Path):
    toks = path.read_text().split()
    if len(toks) % 2:
        toks = toks[:-1]
    return [(toks[i], toks[i+1]) for i in range(0, len(toks), 2)]


def load_official_splits(split_files, n_train, n_val, n_test, seed, valid_accs):
    """按官方 6 文件加载; [FIX-1] 采样前过滤缺序列蛋白的对"""
    rng = random.Random(seed)
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    splits = {}
    for split, files in split_files.items():
        pos, neg, dropped = [], [], 0
        for fn, label in files:
            fp = Path(fn)
            if not fp.exists():
                print(f"[WARN] {fn} 不存在, 跳过")
                continue
            for p in parse_pairs(fp):
                if p[0] in valid_accs and p[1] in valid_accs:
                    (pos if label == 1 else neg).append(p)
                else:
                    dropped += 1
        n = sizes[split]
        half = n // 2
        n_pos = min(half, len(pos))
        n_neg = min(n - n_pos, len(neg))
        sampled = [(a, b, 1) for a, b in rng.sample(pos, n_pos)] + \
                  [(a, b, 0) for a, b in rng.sample(neg, n_neg)]
        rng.shuffle(sampled)
        splits[split] = sampled
        print(f"  {split:<6}: pos {n_pos}/{len(pos)} | neg {n_neg}/{len(neg)} "
              f"| 采样 {len(sampled)} | 过滤缺序列 {dropped} 对")
    return splits


# ============================================================
# 2. 元数据与 pLDDT
# ============================================================
def load_metadata():
    with open(META_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return {acc: m["sequence"] for acc, m in raw.items()
            if m and m.get("sequence") and len(m["sequence"]) >= 30}


def load_plddt(acc):
    pf = PLDDT_DIR / f"{acc}_plddt.json"
    if pf.exists():
        try:
            vals = json.loads(pf.read_text())
            if isinstance(vals, list) and vals:
                return np.array(vals, dtype=np.float32)
        except Exception:
            pass
    return None


# ============================================================
# 3. Embedding: npy 缓存 → .pt (无 ESM)
# ============================================================
_PER_RESIDUE_KEYS = ("representations", "per_residue", "token_embeddings")


def _read_pt(acc):
    """读取 .pt → (逐残基tensor, mean, label); 任何失败返回 None"""
    pf = PT_EMB_DIR / f"{acc}.pt"
    if not pf.exists():
        return None
    try:
        obj = torch.load(pf, map_location="cpu", weights_only=True)
    except TypeError:                       # 旧版 torch 无此参数
        obj = torch.load(pf, map_location="cpu")
    except Exception:
        return None
    if not isinstance(obj, dict):
        return (obj, None, None) if isinstance(obj, torch.Tensor) else None
    t = None
    for key in _PER_RESIDUE_KEYS:           # 逐残基键优先; mean(一维)不入选
        if key in obj:
            v = obj[key]
            cand = v[max(v.keys())] if isinstance(v, dict) else v
            if isinstance(cand, torch.Tensor) and cand.dim() == 2 \
               and cand.shape[-1] == EMB_DIM:
                t = cand
                break
    if t is None:
        return None
    m, mv = None, obj.get("mean_representations")
    if isinstance(mv, dict):
        mc = mv[max(mv.keys())]
        m = mc if isinstance(mc, torch.Tensor) and mc.dim() == 1 \
            and mc.shape[0] == EMB_DIM else None
    return t, m, obj.get("label")


def detect_convention(needed, meta, n_scan=80):
    """[NEW-11] 提取惯例检测。
    信号1: d = rows-Ls 分布 (bos 惯例 → d≈+2 占绝对多数; none → d=0)
    信号2: mean_representations 与候选均值比对 (正确口径误差 ~0,
           错误口径 3e-3~3e-2, 判别 4 个数量级)
    返回 ('bos'|'none'|'unknown', 证据 dict)"""
    d_cnt, votes, scanned = Counter(), Counter(), 0
    for acc in needed:
        if scanned >= n_scan:
            break
        r = _read_pt(acc)
        if r is None:
            continue
        t, m, _ = r
        scanned += 1
        d = t.shape[0] - len(meta[acc])
        if d not in (0, 1, 2):              # 截断/异常蛋白不参与投票
            continue
        d_cnt[d] += 1
        if m is None:
            continue
        cands = {"none": t.mean(0)}
        if t.shape[0] >= 2: cands["bos"]      = t[1:].mean(0)
        if t.shape[0] >= 3: cands["bosfull"]  = t[1:-1].mean(0)
        errs = sorted(((v - m).abs().mean().item(), k) for k, v in cands.items())
        if errs[0][0] < 1e-2 and errs[1][0] > 50 * max(errs[0][0], 1e-12):
            votes["bos" if errs[0][1] != "none" else "none"] += 1
    tot = sum(d_cnt.values())
    d2 = d_cnt[2] / tot if tot else 0.0
    d0 = d_cnt[0] / tot if tot else 0.0
    if tot and d2 >= 0.8:   conv = "bos"
    elif tot and d0 >= 0.8: conv = "none"
    elif votes:             conv = votes.most_common(1)[0][0]
    else:                   conv = "unknown"
    return conv, {"scanned": scanned, "d_dist": dict(d_cnt),
                  "mean_votes": dict(votes)}


def _load_pt_embedding(acc, seq, max_len, conv):
    """[NEW-8/12] v3 适配器。
    bos 惯例: rows==Ls+2 → 剥两端; rows==Ls+1 → 剥头;
              0<rows<Ls (截断) → 剥头(去BOS, 官方token截断保留BOS切掉EOS)
    none 惯例: rows==Ls 直用; 0<rows<Ls 直用
    其他 (rows>Ls+2 / rows==0): 拒绝"""
    r = _read_pt(acc)
    if r is None:
        return None
    t, m, label = r
    n, Ls = t.shape[0], len(seq)
    # 防御分支: label 若形如提取序列 (本数据 label=accession, 不触发)
    if isinstance(label, str) and len(label) >= 30 and abs(len(label) - n) <= 2:
        Ll = len(label)
        if   n == Ll + 2: t = t[1:-1]
        elif n == Ll + 1: t = t[1:]
        elif n == Ll:     pass
        else:             return None
        if label == seq or seq.startswith(label) or label.startswith(seq):
            return t[:max_len].numpy().astype(np.float32)
        return None
    # ---- 长度规则 ----
    if   n == Ls + 2: t = t[1:-1]
    elif n == Ls:     pass
    elif n == Ls + 1:
        if conv == "bos": t = t[1:]
        else:             return None
    elif 0 < n < Ls:
        if conv == "bos":   t = t[1:]       # [NEW-12] 截断剥 BOS
        elif conv == "unknown": pass        # 保守原样 (启动时已告警)
    else:
        return None
    if t.shape[0] == 0:
        return None
    return t[:max_len].numpy().astype(np.float32)


def embed_cached(acc, seq, max_len, conv):
    """优先级: npy 缓存 → .pt; 来源统计; 失败返回 None (不崩)"""
    expected_L = min(len(seq), max_len)
    cache = CACHE_DIR / f"{acc}_L{max_len}.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.ndim == 2 and emb.shape[1] == EMB_DIM \
           and 0 < emb.shape[0] <= expected_L:      # [NEW-9]
            SOURCE["npy"] += 1
            return emb
    emb = _load_pt_embedding(acc, seq, max_len, conv)
    if emb is None:
        SOURCE["fail"] += 1
        return None
    SOURCE["pt"] += 1
    np.save(cache, emb)
    return emb


# ============================================================
# 4. 界面 mask
# ============================================================
def build_mask(acc, seq, plddt_dict, max_len):
    vals = plddt_dict.get(acc)
    if vals is None:
        return None
    m = (vals < PLDDT_IF_THRESHOLD).astype(np.float32)
    seq_len = min(len(seq), max_len)
    if len(m) >= seq_len:
        return m[:seq_len]
    return np.concatenate([m, np.zeros(seq_len - len(m), np.float32)])


# ============================================================
# 5. Dataset
# ============================================================
class PPIDataset(Dataset):
    def __init__(self, samples, acc2emb, acc2mask, max_len):
        self.samples, self.max_len = samples, max_len
        self.emb, self.mask = acc2emb, acc2mask

    def __len__(self):
        return len(self.samples)

    def _pad_emb(self, arr):
        L, D = arr.shape
        if L >= self.max_len:
            return arr[:self.max_len]
        return np.concatenate([arr, np.zeros((self.max_len - L, D), np.float32)])

    def _pad_mask(self, m):
        if m is None:
            return np.zeros(self.max_len, np.float32)
        m = m[:self.max_len]
        return np.concatenate([m, np.zeros(self.max_len - len(m), np.float32)])

    def __getitem__(self, i):
        a, b, y = self.samples[i]
        ea, eb = self._pad_emb(self.emb[a]), self._pad_emb(self.emb[b])
        ma, mb = self._pad_mask(self.mask.get(a)), self._pad_mask(self.mask.get(b))
        return (torch.from_numpy(ea), torch.from_numpy(eb),
                torch.from_numpy(ma), torch.from_numpy(mb),
                torch.tensor(y, dtype=torch.long))


# ============================================================
# 6. 池化与模型
# ============================================================
def valid_positions(x):
    return (x.abs().sum(-1) > 0).float()


def seq_mean(x, eps=1e-8):
    v = valid_positions(x)
    return (x * v.unsqueeze(-1)).sum(1) / (v.sum(1, keepdim=True) + eps)


def masked_pool(x, m, eps=1e-8):
    v = valid_positions(x)
    m_eff = m * v
    m_sum = m_eff.sum(dim=1, keepdim=True)
    m_safe = torch.where(m_sum > 0, m_eff, v)
    return (x * m_safe.unsqueeze(-1)).sum(1) / (m_safe.sum(1, keepdim=True) + eps)

# ── 补丁 1: BaselineModel (整体替换) ──
class BaselineModel(nn.Module):
    def __init__(self, d=EMB_DIM, h=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(d*4),
            nn.Linear(d*4,h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h//2, 2))

    def forward(self, ea, eb, ma, mb):
        pa, pb = seq_mean(ea), seq_mean(eb)
        z = torch.cat([pa, pb, pa*pb, (pa-pb).abs()], -1)   # [NEW-13]
        return self.mlp(z)


class LocalAttentionBlock(nn.Module):
    def __init__(self, d=EMB_DIM, heads=8, radius=5):
        super().__init__()
        self.radius = radius
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        dilated = F.max_pool1d(
            mask.unsqueeze(1), kernel_size=2*self.radius+1,
            stride=1, padding=self.radius).squeeze(1)
        empty = dilated.sum(-1) == 0
        if empty.any():
            dilated[empty] = 1.0
        valid = valid_positions(x)
        attn_mask = dilated * valid
        zero_rows = (attn_mask.sum(-1) == 0)
        if zero_rows.any():
            attn_mask[zero_rows] = valid[zero_rows]
        still = (attn_mask.sum(-1) == 0)
        if still.any():
            attn_mask[still, 0] = 1.0
        out, _ = self.attn(x, x, x, key_padding_mask=(attn_mask == 0))
        return self.norm(x + out) * valid.unsqueeze(-1)   # [FIX-6]


# ── 补丁 2: Level3Model (整体替换) ──
class Level3Model(nn.Module):
    def __init__(self, d=EMB_DIM, h=256):
        super().__init__()
        self.attn_a = LocalAttentionBlock(d)
        self.attn_b = LocalAttentionBlock(d)
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(d*4),
            nn.Linear(d*4, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h//2, 2))

    def forward(self, ea, eb, ma, mb):
        ea, eb = self.attn_a(ea, ma), self.attn_b(eb, mb)
        pa, pb = masked_pool(ea, ma), masked_pool(eb, mb)
        z = torch.cat([pa, pb, pa*pb, (pa-pb).abs()], -1)   # [NEW-13]
        return self.mlp(z)

# ============================================================
# 7. 训练与评估
# ============================================================
def train_model(model, tr_loader, va_loader, epochs, lr, name):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for ea, eb, ma, mb, y in tr_loader:
            ea, eb, ma, mb, y = (t.to(DEVICE) for t in (ea, eb, ma, mb, y))
            opt.zero_grad()
            loss = crit(model(ea, eb, ma, mb), y)
            loss.backward(); opt.step()
        sch.step()
        mcc, auc, _ = evaluate(model, va_loader)
        if mcc > best:
            best, best_state = mcc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep+1) % 5 == 0:
            print(f"  [{name}] Ep{ep+1:2d} val MCC={mcc:.4f} AUROC={auc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for ea, eb, ma, mb, y in loader:
        ea, eb, ma, mb = (t.to(DEVICE) for t in (ea, eb, ma, mb))
        p = F.softmax(model(ea, eb, ma, mb), -1)[:, 1]
        ys.extend(y.tolist()); ps.extend(p.cpu().tolist())
    ys, ps = np.array(ys), np.array(ps)
    if len(set(ys)) < 2:
        return 0., 0., 0.
    pred = (ps > 0.5).astype(int)
    return (matthews_corrcoef(ys, pred),
            roc_auc_score(ys, ps),
            average_precision_score(ys, ps))


# ============================================================
# 8. 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["official", "clean"],
                    default="official")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- 启动自检 ----
    print(f"[INFO] Device: {DEVICE} | seed: {args.seed} | ESM: 已移除(纯 .pt)")
    if not PT_EMB_DIR.exists():
        print(f"[ERROR] PT_EMB_DIR 不存在: {PT_EMB_DIR.resolve()}")
        sys.exit(1)
    n_pt = len(list(PT_EMB_DIR.glob("*.pt")))
    if n_pt == 0:
        print(f"[ERROR] {PT_EMB_DIR.resolve()} 下没有 .pt 文件")
        sys.exit(1)
    print(f"[INFO] .pt 目录 OK: {n_pt} 个文件")

    # ---- split 文件选择 ----
    split_files, actual_mode = get_split_files(args.split_mode)
    print(f"[INFO] split-mode: {args.split_mode} → 实际: {actual_mode}")

    # ---- 数据 ----
    print(f"\n[STEP] 加载切分 ({actual_mode}) ...")
    meta = load_metadata()
    print(f"[INFO] 序列可用蛋白: {len(meta)}")
    splits = load_official_splits(split_files, args.n_train, args.n_val,
                                  args.n_test, args.seed, set(meta))

    # ---- embedding + mask ----
    all_pairs = splits["train"] + splits["val"] + splits["test"]
    needed = sorted({a for s in all_pairs for a in s[:2]})
    needed = [a for a in needed if a in meta]

    # ---- [NEW-11] 惯例检测 (在 embedding 之前) ----
    print(f"\n[STEP] .pt 提取惯例检测 (扫描 {min(80, len(needed))} 个) ...")
    CONV, ev = detect_convention(needed, meta)
    print(f"[INFO] 惯例: {CONV} | {ev}")
    if CONV == "unknown":
        print("[WARN] 惯例未知 → 截断蛋白按'无特殊token'处理, 可能整体偏移 1 位")
        print("       → 请把上一行 [INFO] 反馈, 由 d_dist/mean_votes 进一步判定")

    print(f"\n[STEP] embedding 加载: {len(needed)} 个蛋白 (max_len={args.max_len})")
    acc2emb, acc2mask, plddt_dict, FAILED = {}, {}, {}, set()
    for i, acc in enumerate(needed, 1):
        emb = embed_cached(acc, meta[acc], args.max_len, CONV)
        if emb is None:
            FAILED.add(acc)
            continue
        acc2emb[acc] = emb
        plddt_dict[acc] = load_plddt(acc)
        acc2mask[acc] = build_mask(acc, meta[acc], plddt_dict, args.max_len)
        if i % 200 == 0:
            print(f"  {i}/{len(needed)}")

    # ---- 来源统计 + 失败处理 ----
    print(f"[INFO] embedding 来源: .pt={SOURCE['pt']} | npy缓存={SOURCE['npy']} "
          f"| 失败={SOURCE['fail']} (共 {sum(SOURCE.values())})")
    if FAILED:
        print(f"[WARN] {len(FAILED)} 个蛋白无法对齐 → 过滤相关样本对 "
              f"(约 {100*len(FAILED)/max(1,len(needed)):.1f}%)")
        print(f"       示例: {sorted(FAILED)[:5]}")
        for split in ("train", "val", "test"):
            n0 = len(splits[split])
            splits[split] = [s for s in splits[split]
                             if s[0] not in FAILED and s[1] not in FAILED]
            if len(splits[split]) < n0:
                print(f"       {split}: {n0} → {len(splits[split])}")

    fracs = [m.mean() for m in acc2mask.values() if m is not None]
    n_with_mask = sum(1 for m in acc2mask.values() if m is not None)
    if fracs:
        print(f"[INFO] 界面候选占比: {np.mean(fracs):.2%} (中位 {np.median(fracs):.2%})")
        print(f"[INFO] 有 pLDDT 数据: {n_with_mask}/{len(acc2emb)} "
              f"({100*n_with_mask/max(1,len(acc2emb)):.0f}%)")

    # ---- DataLoader ----
    mk = lambda data, sh: DataLoader(
        PPIDataset(data, acc2emb, acc2mask, args.max_len),
        batch_size=args.batch_size, shuffle=sh)
    tr_loader = mk(splits["train"], True)
    va_loader = mk(splits["val"], False)
    te_loader = mk(splits["test"], False)

    # ---- 训练 ----
    print(f"\n{'='*60}\n Baseline (ESM-2 mean-pool + MLP)\n{'='*60}")
    base, _ = train_model(BaselineModel().to(DEVICE),
                          tr_loader, va_loader, args.epochs, args.lr, "Base")

    print(f"\n{'='*60}\n Level 3 (pLDDT mask + local attention)\n{'='*60}")
    l3, _ = train_model(Level3Model().to(DEVICE),
                        tr_loader, va_loader, args.epochs, args.lr, "L3")

    # ---- test 评估 ----
    bm, ba, bp = evaluate(base, te_loader)
    lm, la, lp = evaluate(l3, te_loader)

    print(f"\n{'='*60}")
    print(f" Test 集 ({actual_mode}, n={len(splits['test'])})")
    print(f"{'='*60}")
    print(f"{'Model':<12}{'MCC':>10}{'AUROC':>10}{'AUPRC':>10}")
    print(f"{'-'*42}")
    print(f"{'Baseline':<12}{bm:>10.4f}{ba:>10.4f}{bp:>10.4f}")
    print(f"{'Level 3':<12}{lm:>10.4f}{la:>10.4f}{lp:>10.4f}")
    print(f"{'-'*42}")
    d = lm - bm
    print(f"Δ MCC: {d:+.4f}")

    # ---- 保存 ----
    tag = args.split_mode if actual_mode == args.split_mode \
          else f"{args.split_mode}_FALLBACK"
    result = {
        "seed": args.seed, "n_train": args.n_train,
        "n_val": args.n_val, "n_test": args.n_test,
        "split_mode": args.split_mode, "actual_mode": actual_mode,
        "pt_convention": CONV, "convention_evidence": ev,   # [NEW-11]
        "baseline": {"mcc": bm, "auroc": ba, "auprc": bp},
        "level3": {"mcc": lm, "auroc": la, "auprc": lp},
        "emb_source": SOURCE, "n_failed": len(FAILED),
        "mask_frac_mean": float(np.mean(fracs)) if fracs else None,
        "mask_frac_median": float(np.median(fracs)) if fracs else None,
    }
    out = f"pilot_official_{tag}_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] 结果已保存: {out}")

    if d > 0.05:
        print("✅ Level 3 显著有效, 建议多种子复验 + 扩大规模")
    elif d > 0:
        print("⚠️ 小幅提升, 需 ≥3 种子平均后判断")
    else:
        print("❌ 未提升, 尝试调整 PLDDT_IF_THRESHOLD (60/80)")


if __name__ == "__main__":
    main()
