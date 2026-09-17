#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sanity_probe.py — pilot 结果归因探针
1) 配对健全性  2) pLDDT 长度对齐  3) 截断蛋白 BOS 指纹  4) LR 信号探针
只读数据文件; 第4节会写 npy 缓存(为正式重跑预热, 无害)"""
import json, random
from pathlib import Path
from collections import Counter
import numpy as np
import torch

PT_DIR    = Path("output_embeddings")
META_F    = Path("af_output/metadata/monomer_metadata.json")
PLDDT_DIR = Path("af_output/monomers")
CACHE     = Path("pilot_cache_v2"); CACHE.mkdir(exist_ok=True)
EMB_DIM, MAX_LEN = 1280, 400
SPLITS = {"train": [("Intra1_pos_rr.txt",1),("Intra1_neg_rr.txt",0)],
          "val":   [("Intra0_pos_rr.txt",1),("Intra0_neg_rr.txt",0)],
          "test":  [("Intra2_pos_rr.txt",1),("Intra2_neg_rr.txt",0)]}

meta = {a: m["sequence"] for a, m in json.loads(META_F.read_text(encoding="utf-8")).items()
        if m and m.get("sequence") and len(m["sequence"]) >= 30}

def parse_pairs(p):
    t = p.read_text().split()
    if len(t) % 2: t = t[:-1]
    return [(t[i], t[i+1]) for i in range(0, len(t), 2)]

def read_t(acc):
    pf = PT_DIR / f"{acc}.pt"
    if not pf.exists(): return None
    try: obj = torch.load(pf, map_location="cpu", weights_only=True)
    except TypeError: obj = torch.load(pf, map_location="cpu")
    if isinstance(obj, dict) and "representations" in obj:
        v = obj["representations"]
        return v[max(v.keys())] if isinstance(v, dict) else v
    return obj if isinstance(obj, torch.Tensor) else None

# ============ 1) 配对健全性 ============
print("=== 1) 配对健全性 ===")
allpairs, prots = {}, {}
for sp, fl in SPLITS.items():
    pos, neg = set(), set()
    for fn, lab in fl:
        for a, b in parse_pairs(Path(fn)):
            (pos if lab == 1 else neg).add((a, b))
    allpairs[sp] = (pos, neg)
    U = pos | neg
    prots[sp] = {a for p in U for a in p}
    n_self = sum(1 for a, b in U if a == b)
    print(f"  {sp}: {len(U)} 唯一对 | 自配 {n_self} | 蛋白 {len(prots[sp])}")
tU, vU, eU = (allpairs[s][0] | allpairs[s][1] for s in ("train","val","test"))
print(f"  对级泄漏: train∩val={len(tU&vU)} train∩test={len(tU&eU)} val∩test={len(vU&eU)}")
print(f"  蛋白级重叠: train∩test={len(prots['train']&prots['test'])} "
      f"train∩val={len(prots['train']&prots['val'])}")

# ============ 2) pLDDT 长度对齐 ============
print("\n=== 2) pLDDT 长度 vs metadata 序列长度 ===")
need_all = sorted({a for sp in allpairs.values() for P in sp for p in P for a in p if a in meta})
ok = bad = miss = 0
bad_ex = []
for a in need_all:
    pf = PLDDT_DIR / f"{a}_plddt.json"
    if not pf.exists():
        miss += 1; continue
    try: v = json.loads(pf.read_text())
    except Exception: miss += 1; continue
    if len(v) == len(meta[a]): ok += 1
    else:
        bad += 1
        if len(bad_ex) < 3: bad_ex.append(f"{a}(plddt{len(v)}/seq{len(meta[a])})")
print(f"  相等 {ok} | 不等 {bad} {bad_ex} | 缺失 {miss} / {len(need_all)}")

# ============ 3) 截断蛋白 BOS 指纹 ============
print("\n=== 3) 截断蛋白 BOS 指纹 (裁决 t[0] 是否特殊token) ===")
ex0, tr0, tr1 = [], [], []
for acc in need_all[:2500]:
    t = read_t(acc)
    if t is None or t.dim() != 2 or t.shape[-1] != EMB_DIM: continue
    n, Ls = t.shape[0], len(meta[acc])
    if n > 60:
        if 0 < n < Ls:
            tr0.append(t[0].clone()); tr1.append(t[1].clone())
        elif n == Ls:
            ex0.append(t[0].clone())
    if len(tr0) >= 150 and len(ex0) >= 150: break
cap = min(len(tr0), len(ex0), 120)
print(f"  样本: 精确 {min(len(ex0),cap)} | 截断 {min(len(tr0),cap)} "
      f"(扫描 need_all 前 2500, 找到截断 {len(tr0)})")
def cc(V):
    V = torch.stack(V[:cap]); V = V / V.norm(dim=1, keepdim=True)
    S = (V @ V.T).numpy(); i, j = np.triu_indices(len(V), 1)
    return float(S[i, j].mean())
def cosv(a, b): return torch.dot(a/a.norm(), b/b.norm()).item()
A, B, C = cc(ex0), cc(tr0), cc(tr1)
cA, cB, cC = (torch.stack(x[:cap]).mean(0) for x in (ex0, tr0, tr1))
nB = torch.stack(tr0[:cap]).norm(dim=1).mean().item()
nC = torch.stack(tr1[:cap]).norm(dim=1).mean().item()
print(f"  跨蛋白两两余弦: exact_t0(A)={A:.3f} | trunc_t0(B)={B:.3f} | trunc_t1(C)={C:.3f}")
print(f"  B−C = {B-C:+.3f}   质心cos: A−B={cosv(cA,cB):.3f} A−C={cosv(cA,cC):.3f}   "
      f"行范数: B={nB:.1f} C={nC:.1f}")
verd_bos = (B - C > 0.05) and (cosv(cA, cC) > cosv(cA, cB) + 0.05)
verd_txt = "疑似BOS(截断需剥第0行)" if verd_bos else "none安全(t[0]=残基)"
if 0.02 < B - C <= 0.05: verd_txt += " [边缘, 请把上面数值发回人工判断]"
print(f"  → 判定: {verd_txt}")

# ============ 4) LR 信号探针 ============
print("\n=== 4) LR 信号探针 (1500/300/300, seed 0) ===")
def load_emb(acc, seq):
    c = CACHE / f"{acc}_L{MAX_LEN}.npy"
    if c.exists():
        e = np.load(c)
        if e.ndim == 2 and e.shape[1] == EMB_DIM and 0 < e.shape[0] <= min(len(seq), MAX_LEN):
            return e
    t = read_t(acc)
    if t is None or t.dim() != 2 or t.shape[-1] != EMB_DIM: return None
    n, Ls = t.shape[0], len(seq)
    if n == Ls + 2: t = t[1:-1]
    elif n == Ls or (0 < n < Ls): pass
    else: return None
    e = t[:MAX_LEN].numpy().astype(np.float32)
    np.save(c, e)
    return e

rng = random.Random(0)
n_map = {"train": 1500, "val": 300, "test": 300}
data = {}
for sp in ("train", "val", "test"):
    pos, neg = allpairs[sp]
    pos = [p for p in pos if p[0] in meta and p[1] in meta]
    neg = [p for p in neg if p[0] in meta and p[1] in meta]
    n = n_map[sp]; half = n // 2
    s = ([(a, b, 1) for a, b in rng.sample(sorted(pos), min(half, len(pos)))] +
         [(a, b, 0) for a, b in rng.sample(sorted(neg), min(n - half, len(neg)))])
    rng.shuffle(s)
    data[sp] = s

needed = sorted({a for s in data.values() for x in s for a in x[:2]})
emb = {}
for i, a in enumerate(needed, 1):
    e = load_emb(a, meta[a])
    if e is not None: emb[a] = e
    if i % 500 == 0: print(f"  加载 {i}/{len(needed)}")
for sp in data:
    n0 = len(data[sp])
    data[sp] = [x for x in data[sp] if x[0] in emb and x[1] in emb]
    print(f"  {sp}: {n0} → {len(data[sp])}")

def feat(a, b):
    ea, eb = emb[a].mean(0), emb[b].mean(0)
    return np.concatenate([ea, eb, ea * eb, np.abs(ea - eb)]).astype(np.float32)

X = {sp: np.stack([feat(a, b) for a, b, _ in data[sp]]) for sp in data}
Y = {sp: np.array([y for _, _, y in data[sp]]) for sp in data}

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, matthews_corrcoef

best = None
for C in (1e-3, 1e-2, 1e-1):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
    clf.fit(X["train"], Y["train"])
    va = roc_auc_score(Y["val"], clf.predict_proba(X["val"])[:, 1])
    print(f"  LR C={C:g}: val AUROC={va:.3f}")
    if best is None or va > best[0]: best = (va, C, clf)
va, C, clf = best
print(f"  选 C={C:g} (按 val)")
for sp in ("train", "val", "test"):
    p = clf.predict_proba(X[sp])[:, 1]
    print(f"  {sp:<5}: AUROC={roc_auc_score(Y[sp], p):.3f} "
          f"MCC={matthews_corrcoef(Y[sp], (p > .5)):.3f}")
clf2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
clf2.fit(np.vstack([X["train"], X["val"]]), np.concatenate([Y["train"], Y["val"]]))
p = clf2.predict_proba(X["test"])[:, 1]
print(f"  train+val→test: AUROC={roc_auc_score(Y['test'], p):.3f}")

print(f"\n{'='*50}")
print("汇总: BOS指纹={} | pLDDT不等={} | LR(val={:.3f})".format(verd_txt, bad, va))
print(f"判读: LR test≥0.62→特征有信号,重跑加样本; 双~0.5→特征侧需重设计")
