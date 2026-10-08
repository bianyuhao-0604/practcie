"""Phase 2 验收 v2：fp16 存储合同（row == src.astype(fp16)）+ 三 split + collate 冒烟。"""
import os, sys
import numpy as np
sys.path.insert(0, ".")
from config import MAX_LEN
from data import FeatureStore, PairDataset, make_collate, load_pairs

FEAT = "workdir/features"
store = FeatureStore(FEAT)
assert store.n == 11018, store.n
assert store.seq is not None and store.seq.shape == (11018, MAX_LEN, 1280), getattr(store.seq, "shape", None)
print(f"store n={store.n} seq={store.seq.shape} {store.seq.dtype}")

primaries = [l.split("\t")[0] for l in open(f"{FEAT}/features_node.tsv").readlines()[1:]]
rng = np.random.default_rng(0)
rows = [0, 1, 3000, 5555, 11016, 11017] + sorted(rng.choice(11018, 10, replace=False).tolist())
worst = 0.0
for i in rows:
    p = primaries[i]
    row = np.asarray(store.seq[i]); src = np.load(f"{FEAT}/seq_{p}.npy")
    L, Lc = int(store.lengths[i]), min(len(src), MAX_LEN)
    assert L == Lc, f"LEN mismatch {p}: store={L} src={Lc}"
    ref = np.asarray(src[:Lc], np.float16)                  # collect 的写入合同
    if not np.array_equal(row[:Lc], ref):
        d = np.abs(row[:Lc].astype(np.float32) - src[:Lc].astype(np.float32))
        raise SystemExit(f"REAL CORRUPTION row {i} {p}: max|diff|={d.max():.3f} (== 嵌入量级? 行没写进去)")
    d = np.abs(row[:Lc].astype(np.float32) - src[:Lc].astype(np.float32))
    worst = max(worst, float(d.max()))
    assert not np.isnan(row[:Lc]).any() and float(np.abs(row[:Lc]).max()) > 0
    if Lc < MAX_LEN: assert row[Lc:].max() == 0, f"padding not zero {p}"
src0 = np.load(f"{FEAT}/seq_{primaries[0]}.npy")
print(f"row-level OK: {len(rows)} rows bitwise==fp16(src)  worst|quant|={worst:.2e} (良性)  src_dtype={src0.dtype}")

ls = np.asarray(store.lengths)
print(f"lengths: min={ls.min()} med={int(np.median(ls))} max={ls.max()} truncated(>={MAX_LEN})={(ls>=MAX_LEN).sum()}")

ds = PairDataset(load_pairs("workdir/data/val.csv"), store)
for s in ["train", "test"]:
    PairDataset(load_pairs(f"workdir/data/{s}.csv"), store)
print(f"PairDataset ok val={len(ds)}")

batch = make_collate(store)([ds[i] for i in [0, 1, 2, 3]])
A, yb = batch["A"], batch["y"]
print("collate A:", {k: tuple(v.shape) for k, v in A.items()})
print("y:", tuple(yb.shape) if hasattr(yb, "shape") else type(yb).__name__,
      yb.tolist() if hasattr(yb, "tolist") else yb)
print("STORE VERIFIED (fp16 contract) — R0 fuel loaded")
