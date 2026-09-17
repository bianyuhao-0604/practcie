#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structure_baseline.py — 仅结构信息 (AF pLDDT) 的 PPI 基线 (M0)
==============================================================
与 pilot_official.py 使用: 同官方划分文件 / 同采样函数(逐字复制) / 同 seed
→ 样本对与 pilot 完全一致 (各自通道的缺失过滤除外, 见运行输出)。

通道: 每蛋白 pLDDT 曲线 → 21 维特征 (无任何序列 embedding):
  log长度 | mean/std/min/max | frac(<70,<50,>90) |
  最长低置信段/低置信段数/低置信总量 | 8 分位均值 | N端/C端50均值
配对特征: [fa, fb, |fa-fb|, fa*fb] = 84 维
模型: Struct-LR (C 按 val 选) + Struct-MLP (同 pilot 训练协议)

三模型阶梯 (同批样本):
  M0 本脚本         仅结构
  M1 pilot Baseline 仅序列 (不用 mask)
  M2 pilot Level 3  序列 + 结构

运行 (参数与 pilot 重跑一致):
  python structure_baseline.py --n-train 4000 --n-val 600 --n-test 600 --seed 42
"""
import sys, json, random, argparse, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

AF_OUTPUT = Path("af_output")
META_FILE = AF_OUTPUT / "metadata" / "monomer_metadata.json"
PLDDT_DIR = AF_OUTPUT / "monomers"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 以下三个函数与 pilot_official.py 逐字一致 (保证同样本)
# ============================================================
def parse_pairs(path: Path):
    toks = path.read_text().split()
    if len(toks) % 2:
        toks = toks[:-1]
    return [(toks[i], toks[i+1]) for i in range(0, len(toks), 2)]


def load_official_splits(split_files, n_train, n_val, n_test, seed, valid_accs):
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


def load_metadata():
    with open(META_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return {acc: m["sequence"] for acc, m in raw.items()
            if m and m.get("sequence") and len(m["sequence"]) >= 30}


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
            print(f"[WARN] clean 切分缺失 {len(missing)} 个 → 回退 official")
            return get_split_files("official")[0], "official(fallback)"
    return files, mode


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
# 结构特征 (纯 pLDDT, 无序列)
# ============================================================
def struct_features(vals):
    v = np.asarray(vals, dtype=np.float32)
    L = v.shape[0]
    f = [float(np.log1p(L)), float(v.mean()), float(v.std()),
         float(v.min()), float(v.max()),
         float((v < 70).mean()), float((v < 50).mean()), float((v > 90).mean())]
    below = v < 70
    runs, cur = [], 0
    for b in below:
        if b:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    f += [float(max(runs)) if runs else 0.0,
          float(len(runs)), float(sum(runs))]
    f += [float(b.mean()) for b in np.array_split(v, 8)]
    k = min(50, L)
    f += [float(v[:k].mean()), float(v[-k:].mean())]
    return np.array(f, dtype=np.float32)


def pair_feat(fa, fb):
    return np.concatenate([fa, fb, np.abs(fa - fb), fa * fb]).astype(np.float32)


def clf_metrics(p, y):
    if len(set(y.tolist())) < 2:
        return 0., 0., 0.
    pred = (p > 0.5).astype(int)
    return (matthews_corrcoef(y, pred),
            roc_auc_score(y, p), average_precision_score(y, p))


# ============================================================
# Struct-MLP (同 pilot 训练协议)
# ============================================================
class StructMLP(nn.Module):
    def __init__(self, din=84, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h//2, 2))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def eval_mlp(model, X, y):
    model.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                    batch_size=256, shuffle=False)
    ps = []
    for xb, _ in dl:
        xb = xb.to(DEVICE)
        ps.append(F.softmax(model(xb), -1)[:, 1].cpu().numpy())
    return clf_metrics(np.concatenate(ps), y)


def train_mlp(Xtr, ytr, Xva, yva, epochs, lr, bs):
    model = StructMLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                  torch.from_numpy(ytr.astype(np.int64))),
                    batch_size=bs, shuffle=True)
    best, best_state = -1., None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        sch.step()
        mcc, auc, _ = eval_mlp(model, Xva, yva)
        if mcc > best:
            best, best_state = mcc, {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}
        if (ep+1) % 5 == 0:
            print(f"  [SMLP] Ep{ep+1:2d} val MCC={mcc:.4f} AUROC={auc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["official", "clean"],
                    default="official")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"[INFO] M0 仅结构基线 | Device: {DEVICE} | seed: {args.seed}")
    split_files, actual_mode = get_split_files(args.split_mode)
    meta = load_metadata()
    print(f"[INFO] 序列可用蛋白: {len(meta)} | split: {actual_mode}")
    splits = load_official_splits(split_files, args.n_train, args.n_val,
                                  args.n_test, args.seed, set(meta))

    needed = sorted({a for s in splits.values() for x in s for a in x[:2]})
    needed = [a for a in needed if a in meta]

    print(f"\n[STEP] 结构特征: {len(needed)} 个蛋白 ...")
    acc2feat, MISSING = {}, set()
    for i, acc in enumerate(needed, 1):
        pv = load_plddt(acc)
        if pv is None:
            MISSING.add(acc)
        else:
            acc2feat[acc] = struct_features(pv)
        if i % 1000 == 0:
            print(f"  {i}/{len(needed)}")
    print(f"[INFO] pLDDT 缺失 {len(MISSING)}/{len(needed)} "
          f"({100*len(MISSING)/max(1,len(needed)):.1f}%)")

    for sp in ("train", "val", "test"):
        n0 = len(splits[sp])
        splits[sp] = [x for x in splits[sp]
                     if x[0] in acc2feat and x[1] in acc2feat]
        print(f"  {sp}: {n0} → {len(splits[sp])}")

    X = {sp: np.stack([pair_feat(acc2feat[a], acc2feat[b])
                       for a, b, _ in splits[sp]]) for sp in splits}
    Y = {sp: np.array([y for _, _, y in splits[sp]]) for sp in splits}

    # ---- Struct-LR ----
    print("\n[STEP] Struct-LR (C 按 val 选) ...")
    best = None
    for C in (1e-3, 1e-2, 1e-1):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=C))
        clf.fit(X["train"], Y["train"])
        va = roc_auc_score(Y["val"], clf.predict_proba(X["val"])[:, 1])
        print(f"  C={C:g}: val AUROC={va:.3f}")
        if best is None or va > best[0]:
            best = (va, C, clf)
    va, C, clf = best
    lr_m = {sp: clf_metrics(clf.predict_proba(X[sp])[:, 1], Y[sp])
            for sp in ("train", "val", "test")}
    clf2 = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=C))
    clf2.fit(np.vstack([X["train"], X["val"]]),
             np.concatenate([Y["train"], Y["val"]]))
    tv_t = clf_metrics(clf2.predict_proba(X["test"])[:, 1], Y["test"])

    # ---- Struct-MLP ----
    print(f"\n[STEP] Struct-MLP (Ep{args.epochs}, lr={args.lr:g}) ...")
    mu = X["train"].mean(0); sd = X["train"].std(0); sd[sd < 1e-8] = 1.0
    Z = {sp: ((X[sp] - mu) / sd).astype(np.float32) for sp in X}
    mlp = train_mlp(Z["train"], Y["train"], Z["val"], Y["val"],
                    args.epochs, args.lr, args.batch_size)
    mlp_m = {sp: eval_mlp(mlp, Z[sp], Y[sp]) for sp in ("train", "val", "test")}

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f" M0 仅结构 Test ({actual_mode}, n={len(splits['test'])})")
    print(f"{'='*60}")
    print(f"{'Model':<12}{'MCC':>10}{'AUROC':>10}{'AUPRC':>10}")
    print("-" * 42)
    for name, m in [("Struct-LR", lr_m), ("Struct-MLP", mlp_m)]:
        mcc, auc, ap = m["test"]
        print(f"{name:<12}{mcc:>10.4f}{auc:>10.4f}{ap:>10.4f}")
    tv_auc = tv_t[1]
    print("-" * 42)
    print(f"LR(train+val→test): AUROC={tv_auc:.4f}")

    out = f"structure_baseline_{args.split_mode}_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump({
            "seed": args.seed, "actual_mode": actual_mode,
            "n_train": len(splits["train"]), "n_val": len(splits["val"]),
            "n_test": len(splits["test"]), "n_missing_plddt": len(MISSING),
            "lr": {sp: [float(x) for x in lr_m[sp]] for sp in lr_m},
            "lr_trainval_test": [float(x) for x in tv_t],
            "mlp": {sp: [float(x) for x in mlp_m[sp]] for sp in mlp_m},
        }, f, indent=2)
    print(f"\n[INFO] 已保存: {out}")
    print("[提示] 与 pilot 的 Baseline(M1)/Level3(M2) 同 seed 同参数对照")


if __name__ == "__main__":
    main()
