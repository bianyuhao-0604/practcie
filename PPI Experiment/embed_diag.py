#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""embed_diag.py — 定位 .pt 加载失败原因 (只读诊断, 不写任何文件)"""
import sys, json, statistics
from pathlib import Path
from collections import Counter
import torch

PT_DIR  = Path("output_embeddings")
META    = Path("af_output/metadata/monomer_metadata.json")
EMB_DIM = 1280
SPLITS  = ["Intra1_pos_rr.txt", "Intra1_neg_rr.txt", "Intra0_pos_rr.txt",
           "Intra0_neg_rr.txt", "Intra2_pos_rr.txt", "Intra2_neg_rr.txt"]
_PER_RESIDUE_KEYS = ("representations", "per_residue", "token_embeddings")

def main():
    raw = json.loads(META.read_text(encoding="utf-8"))
    meta = {a: m["sequence"] for a, m in raw.items()
            if m and m.get("sequence") and len(m["sequence"]) >= 30}

    needed = set()
    for fn in SPLITS:
        p = Path(fn)
        if p.exists():
            toks = p.read_text().split()
            needed.update(t for t in toks if t in meta)
    needed = sorted(needed)
    lens = [len(meta[a]) for a in needed]
    n_long = sum(1 for L in lens if L > 1022)
    print(f"needed 蛋白: {len(needed)} | PT_DIR: {PT_DIR.resolve()}")
    print(f"序列长度: 中位 {int(statistics.median(lens))} | >1022(截断嫌疑): {n_long} 个\n")

    stats, examples = Counter(), {}
    def rec(key, detail):
        stats[key] += 1
        if len(examples.setdefault(key, [])) < 3:
            examples[key].append(detail)

    v1_ok = v2_ok = 0
    for i, acc in enumerate(needed, 1):
        seq = meta[acc]
        pf = PT_DIR / f"{acc}.pt"
        if not pf.exists():
            rec("A_无文件", acc); continue
        try:
            obj = torch.load(pf, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(pf, map_location="cpu")
        except Exception as e:
            rec("B_加载失败", f"{acc}: {type(e).__name__}"); continue
        if isinstance(obj, dict):
            t, label = None, obj.get("label")
            for k in _PER_RESIDUE_KEYS:
                if k in obj:
                    v = obj[k]
                    t = v[max(v.keys())] if isinstance(v, dict) else v
                    break
        else:
            t, label = obj, None
        if not isinstance(t, torch.Tensor) or t.dim() != 2 or t.shape[-1] != EMB_DIM:
            rec("C_维度异常", f"{acc}: {tuple(t.shape) if hasattr(t, 'shape') else type(t)}")
            continue
        n, Ls = t.shape[0], len(seq)
        if (n - Ls) in (0, 1, 2):          # 旧适配器能接受(注意: 含H_错位风险!)
            v1_ok += 1
        if isinstance(label, str) and label:
            Ll, dev = len(label), n - len(label)
            if dev not in (0, 1, 2):
                rec(f"rows≠label(偏差{dev:+d})", f"{acc}(rows{n} label{Ll})"); continue
            if   label == seq:          rel = "E_相同"
            elif seq.startswith(label): rel = "F_提取截断"
            elif label.startswith(seq): rel = "G_label更长"
            else:                       rel = "H_序列不同"
            rec(f"{rel}|rows=OK", f"{acc}(seq{Ls} rows{n} label{Ll})")
            if rel != "H_序列不同":
                v2_ok += 1
        else:
            ok = (n - Ls) in (0, 1, 2)
            rec("D_无label" + ("" if ok else "(v1拒)"), f"{acc}(seq{Ls} rows{n})")
            if ok:
                v2_ok += 1
        if i % 500 == 0:
            print(f"  ... {i}/{len(needed)}")

    print("=== 分类统计 ===")
    for k, v in stats.most_common():
        print(f"{v:>6}  {k}")
    print("\n=== 每类示例(前3) ===")
    for k, lst in examples.items():
        print(f"{k}: {lst}")
    N = len(needed)
    print(f"\n=== 预测 ===")
    print(f"v1(当前代码)能直接加载: {v1_ok}/{N}")
    print(f"v2(label感知补丁)能加载: {v2_ok}/{N} ({100*v2_ok/max(1,N):.1f}%)")
    print(f"打 v2 补丁后 ESM 兜底预计: {N - v2_ok} 个")
    n_h = sum(v for k, v in stats.items() if k.startswith("H_"))
    if n_h:
        print(f"\n[注意] H_序列不同 {n_h} 个: v1 长度恰好一致会静默错位; v2 拒绝(正确)")

if __name__ == "__main__":
    main()
