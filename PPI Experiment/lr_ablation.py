#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lr_ablation.py — 特征级消融: pLDDT 界面先验是否携带增量信号
==============================================================
与 pilot_official.py v3.4 同 seed / 同采样函数 / 同样本 (seed 42, 4000/600/600)
embedding 直接读 pilot_cache_v2 (上次跑已全部缓存, 秒级加载)

三个特征集 (LR, C 按 val 选):
  A  全局 mean-pool            [ga,gb,ga*gb,|ga-gb|]            5120 维
  B  全局 + 界面(低pLDDT区)    A ⊕ [ma,mb,ma*mb,|ma-mb|]       10240 维
  C  仅界面池化                [ma,mb,ma*mb,|ma-mb|]            5120 维

回答两个问题:
  1) B−A: 界面先验的增量 (核心研究问题, 无深度模型混淆)
  2) LR-A vs MLP-Base(0.585): 同样本上 LR 是否显著高于 MLP → 训练协议 vs 特征瓶颈
"""
import json, random, argparse, warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, matthews_corrcoef

warnings.filterwarnings("ignore")

META_FILE = Path("af_output/metadata/monomer_metadata.json")
PLDDT_DIR = Path("af_output/monomers")
CACHE     = Path("pilot_cache_v2")
EMB_DIM, MAX_LEN, THR = 1280, 400, 70.0
SPLIT_FILES = {
    "train": [("Intra1_pos_rr.txt", 1), ("Intra1_neg_rr.txt", 0)],
    "val":   [("Intra0_pos_rr.txt", 1), ("Intra0_neg_rr.txt", 0)],
    "test":  [("Intra2_pos_rr.txt", 1), ("Intra2_neg_rr.txt", 0)],
}

# ---- 以下三个函数与 pilot_official.py 逐字一致 (保证同样本) ----
def parse_pairs(path):
    toks = path.read_text().split()
    if len(toks) % 2:
        toks = toks[:-1]
    return [(toks[i], toks[i+1]) for i in range(0, len(toks), 2)]


def load_official_splits(n_train, n_val, n_test, seed, valid_accs):
    rng = random.Random(seed)
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    splits = {}
    for split, files in SPLIT_FILES.items():
        pos, neg = [], []
        for fn, label in files:
            for p in parse_pairs(Path(fn)):
                if p[0] in valid_accs and p[1] in valid_accs:
                    (pos if label == 1 else neg).append(p)
        n = sizes[split]; half = n // 2
        n_pos = min(half, len(pos)); n_neg = min(n - n_pos, len(neg))
        sampled = [(a, b, 1) for a, b in rng.sample(pos, n_pos)] + \
                  [(a, b, 0) for a, b in rng.sample(neg, n_neg)]
        rng.shuffle(sampled)
        splits[split] = sampled
    return splits


def load_metadata():
    raw = json.loads(META_FILE.read_text(encoding="utf-8"))
    return {a: m["sequence"] for a, m in raw.items()
            if m and m.get("sequence") and len(m["sequence"]) >= 30}


def load_plddt(acc):
    pf = PLDDT_DIR / f"{acc}_plddt.json"
    if pf.exists():
        try:
            vals = json.loads(pf.read_text())
            if isinstance(vals, list) and vals:
                return np.array(vals, np.float32)
        except Exception:
            pass
    return None


def build_mask(vals, seq, max_len):
    m = (vals < THR).astype(np.float32)
    sl = min(len(seq), max_len)
    return m[:sl] if len(m) >= sl else np.concatenate([m, np.zeros(sl-len(m), np.float32)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = load_metadata()
    splits = load_official_splits(args.n_train, args.n_val, args.n_test,
                                  args.seed, set(meta))

    needed = sorted({a for s in splits.values() for x in s for a in x[:2]})
    print(f"[INFO] 同样本消融 | 蛋白 {len(needed)} | 直接读缓存 {CACHE}")

    acc2pg, acc2pm, FALLBACK = {}, {}, set()
    for acc in needed:
        cf = CACHE / f"{acc}_L{MAX_LEN}.npy"
        if not cf.exists():
            continue
        e = np.load(cf)
        if e.ndim != 2 or e.shape[1] != EMB_DIM or e.shape[0] == 0:
            continue
        L = e.shape[0]
        valid = (np.abs(e).sum(-1) > 0).astype(np.float32)
        pg = (e * valid[:, None]).sum(0) / (valid.sum() + 1e-8)
        vals = load_plddt(acc)
        if vals is None:
            pm, fb = pg, True
        else:
            m = build_mask(vals, meta[acc], MAX_LEN)[:L] * valid
            if m.sum() > 0:
                pm = (e * m[:, None]).sum(0) / (m.sum() + 1e-8)
                fb = False
            else:
                pm, fb = pg, True
        if fb:
            FALLBACK.add(acc)
        acc2pg[acc], acc2pm[acc] = pg.astype(np.float32), pm.astype(np.float32)

    print(f"[INFO] 缓存命中 {len(acc2pg)}/{len(needed)} | "
          f"界面池化回退(全高/缺pLDDT) {len(FALLBACK)} 个")

    for sp in splits:
        n0 = len(splits[sp])
        splits[sp] = [x for x in splits[sp]
                      if x[0] in acc2pg and x[1] in acc2pg]
        print(f"  {sp}: {n0} → {len(splits[sp])}")

    def feats(name):
        out = {}
        for sp, data in splits.items():
            rows = []
            for a, b, _ in data:
                ga, gb = acc2pg[a], acc2pg[b]
                ma, mb = acc2pm[a], acc2pm[b]
                if name == "A":
                    rows.append(np.concatenate([ga, gb, ga*gb, np.abs(ga-gb)]))
                elif name == "B":
                    rows.append(np.concatenate(
                        [ga, gb, ga*gb, np.abs(ga-gb),
                         ma, mb, ma*mb, np.abs(ma-mb)]))
                else:
                    rows.append(np.concatenate([ma, mb, ma*mb, np.abs(ma-mb)]))
            out[sp] = np.stack(rows).astype(np.float32)
        return out

    def run_lr(X, y):
        best = None
        for C in (1e-4, 1e-3, 1e-2):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, C=C))
            clf.fit(X["train"], y["train"])
            va = roc_auc_score(y["val"], clf.predict_proba(X["val"])[:, 1])
            if best is None or va > best[0]:
                best = (va, C, clf)
        va, C, clf = best
        res = {}
        for sp in ("train", "val", "test"):
            p = clf.predict_proba(X[sp])[:, 1]
            res[sp] = (roc_auc_score(y[sp], p), matthews_corrcoef(y[sp], p > 0.5))
        return va, C, res

    Y = {sp: np.array([y for _, _, y in data]) for sp, data in splits.items()}
    results = {}
    for name, label in [("A", "全局"), ("B", "全局+界面"), ("C", "仅界面")]:
        X = feats(name)
        va, C, res = run_lr(X, Y)
        results[name] = (va, C, res)
        print(f"\n[{name}] {label} ({X['train'].shape[1]} 维, C={C:g})")
        for sp in ("train", "val", "test"):
            a, m = res[sp]
            print(f"  {sp:<5}: AUROC={a:.4f} MCC={m:.4f}")

    d_test_BA = results["B"][2]["test"][0] - results["A"][2]["test"][0]
    d_test_CA = results["C"][2]["test"][0] - results["A"][2]["test"][0]
    mlp_base_auc = 0.5850          # pilot seed42 的 MLP-Base test AUROC
    lr_a_test = results["A"][2]["test"][0]

    print(f"\n{'='*60}")
    print(f" 消融结论 (test AUROC)")
    print(f"{'='*60}")
    print(f"  B−A (界面先验增量)      : {d_test_BA:+.4f}")
    print(f"  C−A (仅界面 vs 仅全局)  : {d_test_CA:+.4f}")
    print(f"  LR-A({lr_a_test:.4f}) vs MLP-Base({mlp_base_auc:.4f}): "
          f"{'LR 高 → MLP训练是瓶颈' if lr_a_test - mlp_base_auc > 0.03 else '相当 → 特征本身是瓶颈'}")
    print(f"\n判据: B−A ≥ +0.02 → 界面先验有信号, 值得做轻量L2.5;"
          f"\n      B−A < +0.02 → 先验无增量, 诚实负结论 + 多种子固化")

    out = {"seed": args.seed,
           "fallback_pm": len(FALLBACK),
           "B_minus_A_test_auc": float(d_test_BA),
           "C_minus_A_test_auc": float(d_test_CA),
           "lrA": {sp: [float(v) for v in results["A"][2][sp]]
                   for sp in ("train", "val", "test")},
           "lrB": {sp: [float(v) for v in results["B"][2][sp]]
                   for sp in ("train", "val", "test")}}
    with open(f"lr_ablation_seed{args.seed}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[INFO] 已保存: lr_ablation_seed{args.seed}.json")


if __name__ == "__main__":
    main()
