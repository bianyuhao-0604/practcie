"""NaN 取证（只读不写）：row0 深挖 + 源文件全扫 + store 全扫 + 坏蛋白序列速览。"""
import os, sys
import numpy as np
sys.path.insert(0, ".")

FEAT = "workdir/features"
primaries = [l.split("\t")[0] for l in open(f"{FEAT}/features_node.tsv").readlines()[1:]]
assert len(primaries) == 11018, len(primaries)

print("=== A: row 0 forensics (A0A1B0GX68) ===", flush=True)
p0 = primaries[0]
src = np.load(os.path.join(FEAT, f"seq_{p0}.npy"))
row = np.asarray(np.load(os.path.join(FEAT, "seq.npy"), mmap_mode="r")[0])
for name, arr in [("src_file", src), ("store_row", row)]:
    fin = np.isfinite(arr)
    mx = float(np.abs(arr[fin]).max()) if fin.any() else 0.0
    print(f"  {name}: dtype={arr.dtype} shape={arr.shape} nan={int(np.isnan(arr).sum())} "
          f"inf={int(np.isinf(arr).sum())} max|fin|={mx:.3f}")
m = ~np.isfinite(src)
if m.any():
    res = np.where(m.any(-1))[0]; per = m.sum(-1)[res]
    print(f"  坏残基: {len(res)}/{src.shape[0]} 位置[:12]={res[:12].tolist()} 尾3={res[-3:].tolist()}")
    print(f"  每坏残基维度数: min={int(per.min())} max={int(per.max())}/1280 "
          f"({'整残基' if per.max()==1280 else '部分维度'})")
L = min(len(src), 1022)
fin = np.isfinite(src[:L])
print(f"  NaN位置 src==store: {np.array_equal(np.isnan(row[:L]), np.isnan(src[:L]))}  "
      f"有限部分 bitwise: {np.array_equal(row[:L][fin], np.asarray(src[:L], np.float16)[fin])}")

print("=== B: 源文件全扫 11018（~1-3 分钟）===", flush=True)
bad = []
for i, pid in enumerate(primaries):
    try:
        a = np.load(os.path.join(FEAT, f"seq_{pid}.npy"), mmap_mode="r")
        if not np.isfinite(a).all():
            bad.append((pid, int(np.isnan(a).sum()), int(np.isinf(a).sum()), a.shape))
    except Exception as e:
        bad.append((pid, -1, -1, str(e)[:50]))
    if i % 2000 == 0: print(f"  scan {i}/11018", flush=True)
print(f"SCAN_SRC_DONE bad={len(bad)}/11018", flush=True)
for pid, nn, ni, shp in bad: print(f"  BAD {pid}: nan={nn} inf={ni} shape={shp}")

print("=== C: store(seq.npy) 全扫（29 GB，~1-2 分钟）===", flush=True)
mm = np.load(os.path.join(FEAT, "seq.npy"), mmap_mode="r")
bad_rows = []
for s in range(0, 11018, 64):
    chunk = np.asarray(mm[s:s+64])
    ok = np.isfinite(chunk).all(axis=(1, 2))
    bad_rows += [int(s+j) for j in np.where(~ok)[0]]
print(f"SCAN_STORE_DONE bad_rows={len(bad_rows)} pids={[primaries[i] for i in bad_rows][:20]}", flush=True)

print("=== D: 坏蛋白序列速览（低复杂度/异常字符线索）===", flush=True)
want = {p for p, *_ in bad} | {primaries[i] for i in bad_rows} | {p0}
fa = next((f for f in [os.path.join(FEAT, "..", "data", "sequences_primary.fasta"),
                       "workdir/data/sequences_primary.fasta"] if os.path.exists(f)), None)
if fa:
    cur, buf = None, []
    def show(pid, s):
        if pid in want:
            odd = {c: s.count(c) for c in "BXZUO*" if s.count(c)}
            print(f"  {pid}: len={len(s)} head={s[:50]!r} odd={odd}")
    for line in open(fa):
        if line.startswith(">"):
            if cur: show(cur, "".join(buf))
            parts = line[1:].split(); cur, buf = (parts[0] if parts else None), []
        else: buf.append(line.strip())
    if cur: show(cur, "".join(buf))
else:
    print("  fasta 未找到（可跳过）")
print("FORENSICS_DONE", flush=True)
