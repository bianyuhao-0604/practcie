#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""label_probe.py — 确认 .pt 的 label 语义 + 长度不匹配分布 (只读)"""
import json
from pathlib import Path
from collections import Counter
import torch

PT_DIR = Path("output_embeddings")
META   = Path("af_output/metadata/monomer_metadata.json")
SPLITS = ["Intra1_pos_rr.txt","Intra1_neg_rr.txt","Intra0_pos_rr.txt",
          "Intra0_neg_rr.txt","Intra2_pos_rr.txt","Intra2_neg_rr.txt"]

raw = json.loads(META.read_text(encoding="utf-8"))
meta = {a: m["sequence"] for a, m in raw.items()
        if m and m.get("sequence") and len(m["sequence"]) >= 30}
needed = set()
for fn in SPLITS:
    needed.update(t for t in Path(fn).read_text().split() if t in meta)
needed = sorted(needed)

# 1) 三个文件的 label 直接打印
print("=== label 内容抽查 ===")
for acc in ["A0A024RBG1", "A0A075B6H7", "A0A075B6H8"]:
    o = torch.load(PT_DIR / f"{acc}.pt", map_location="cpu", weights_only=False)
    lab, n = o["label"], o["representations"][33].shape[0]
    print(f"{acc}: rows={n} | label({type(lab).__name__})={repr(lab)[:60]}")

# 2) 全量长度分布
print(f"\n=== {len(needed)} 个蛋白长度关系扫描 ===")
ok, trunc, weird = 0, 0, 0
n_vals, diff_vals, label_lens = Counter(), Counter(), Counter()
for acc in needed:
    seq = meta[acc]
    o = torch.load(PT_DIR / f"{acc}.pt", map_location="cpu", weights_only=False)
    t = o["representations"][33]
    n, Ls = t.shape[0], len(seq)
    if isinstance(o.get("label"), str):
        label_lens[len(o["label"])] += 1
    d = n - Ls
    if d in (0, 1, 2):   ok += 1
    elif 0 < n < Ls:     trunc += 1; n_vals[n] += 1; diff_vals[Ls - n] += 1
    else:                weird += 1; n_vals[n] += 1; diff_vals[d] += 1
    if len(needed) > 3000 and acc == needed[2999]:
        print("  ... 1/4")
print(f"长度吻合: {ok} | 截断(0<n<Ls): {trunc} | 无法解释(n>Ls+2): {weird}")
print(f"截断蛋白的 rows 分布(top5): {n_vals.most_common(5)}")
print(f"截断差 Ls-n 分布(top5): {diff_vals.most_common(5)}")
print(f"label 长度分布(top3): {label_lens.most_common(3)}")
