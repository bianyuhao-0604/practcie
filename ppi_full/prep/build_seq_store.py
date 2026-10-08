"""散文件 -> seq.npy（仅 seq 通道，增量建库）+ 全量验收，单文件两阶段。"""
import os, sys, json
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "features")
import store as st
from config import MAX_LEN
from data import FeatureStore, PairDataset, make_collate, load_pairs

FEAT = "workdir/features"

# ---------- Phase 1: collect ----------
print("=== PHASE 1: collect seq channel ===", flush=True)
primaries = [l.split("\t")[0] for l in open(f"{FEAT}/features_node.tsv").readlines()[1:]]
n = len(primaries)
assert n == 11018 and len(set(primaries)) == n

plddt_full = np.load(f"{FEAT}/plddt.npy")
if plddt_full.ndim != 2 or plddt_full.shape[0] != n:
    raise SystemExit(f"FATAL plddt.npy shape {plddt_full.shape}, expect ({n}, {MAX_LEN})")
if plddt_full.shape[1] != MAX_LEN:
    print(f"WARN plddt width {plddt_full.shape[1]} != {MAX_LEN}, realigning (pad=100.0)", flush=True)
    fixed = np.full((n, MAX_LEN), 100.0, np.float16)
    Wm = min(plddt_full.shape[1], MAX_LEN)
    fixed[:, :Wm] = plddt_full[:, :Wm]
    plddt_full = fixed
n_fill = 0
for i in range(n):                        # 无结构行（全零）-> 100.0，防 L2 质量门误杀
    if not plddt_full[i].any():
        plddt_full[i] = 100.0
        n_fill += 1
if n_fill:
    print(f"plddt: {n_fill} all-zero rows -> 100.0 (no-structure, quality-gate off)", flush=True)

old_map = json.load(open(f"{FEAT}/meta.json")).get("pid2idx", {})

st.RES_LAYOUT = {"seq": (1280,)}           # 只开 seq 通道；str/surf/iface 各自提取完再增量建
w = st.StoreWriter(FEAT, primaries)
for i, pid in enumerate(primaries):
    seq = np.load(os.path.join(FEAT, f"seq_{pid}.npy"))
    L = min(len(seq), MAX_LEN)
    w.write(pid, length=L, plddt=plddt_full[i], seq=seq)
    if i % 1000 == 0:
        print(f"collect {i}/{n}", flush=True)
w.flush_meta(extra=dict(channels_present=["seq"],
                        note="seq channel only; str/surf/iface/text/genome pending"))

meta = json.load(open(f"{FEAT}/meta.json"))
new_map = meta["pid2idx"]
bad = [p for p in primaries if p in old_map and old_map[p] != new_map[p]]
if bad:
    raise SystemExit(f"FATAL skeleton meta misaligned on {len(bad)} primaries e.g. {bad[:3]}")
kept = 0
for k, v in old_map.items():              # 骨架 meta 里探针已验证可用的键，原样保留
    if k not in new_map and 0 <= v < n:
        new_map[k] = v
        kept += 1
n_alias = 0
for line in open(f"{FEAT}/uniprot_primary_map.txt"):
    t = line.rstrip("\n").split("\t")
    if len(t) >= 2 and t[1] and t[0] not in new_map and t[1] in new_map:
        new_map[t[0]] = new_map[t[1]]
        n_alias += 1
json.dump(meta, open(f"{FEAT}/meta.json", "w"))
print(f"COLLECT_SEQ_DONE n={n} skeleton_keys_kept={kept} aliases_from_mapfile={n_alias}", flush=True)

# ---------- Phase 2: verify（项目自己的 FeatureStore/PairDataset/collate 验收） ----------
print("=== PHASE 2: verify ===", flush=True)
store = FeatureStore(FEAT)
assert store.n == 11018
assert store.seq is not None and store.seq.shape == (11018, MAX_LEN, 1280), \
    f"seq={getattr(store.seq, 'shape', None)}"
print(f"store n={store.n} seq={store.seq.shape} {store.seq.dtype}")

pl = np.load(f"{FEAT}/plddt.npy", mmap_mode="r")
for i in [0, 3000, 11017]:
    p, L = primaries[i], int(store.lengths[i])
    row = np.asarray(store.seq[i])
    src = np.load(f"{FEAT}/seq_{p}.npy")
    Lc = min(len(src), MAX_LEN)
    assert L == Lc and np.array_equal(row[:Lc], src[:Lc]), f"row mismatch {p}"
    if Lc < MAX_LEN:
        assert row[Lc:].max() == 0, f"padding not zero {p}"
    print(f"  row {i} {p} L={L} plddt_mean={np.asarray(pl[i, :L], np.float32).mean():.1f}", flush=True)

ds = PairDataset(load_pairs("workdir/data/val.csv"), store)
for s in ["train", "test"]:
    PairDataset(load_pairs(f"workdir/data/{s}.csv"), store)
print(f"PairDataset ok val={len(ds)}")

batch = make_collate(store)([ds[i] for i in [0, 1, 2, 3]])
A = batch["A"]
print("collate A:", {k: tuple(v.shape) for k, v in A.items()})
yb = batch["y"]
print("y:", tuple(yb.shape) if hasattr(yb, "shape") else type(yb).__name__,
      yb.tolist() if hasattr(yb, "tolist") else yb)
print("STORE VERIFIED — R0 fuel loaded")
