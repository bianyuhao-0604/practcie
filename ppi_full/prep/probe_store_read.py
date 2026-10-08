"""读取层定罪 probe：FeatureStore vs np.load 裸读 vs 散文件，同进程三读。只读不写。"""
import os, sys, time, warnings
import numpy as np
sys.path.insert(0, ".")
from data import FeatureStore

FEAT = "workdir/features"
P0 = "A0A1B0GX68"

def _w(fn):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = fn()
    return r, [str(w.message)[:60] for w in ws]

print("=== 0) 本机 numpy 警言语义实测（你的 2.x vs 沙箱 1.26.1）===", flush=True)
print("numpy:", np.__version__)
for name, a, b in [("nan - finite", [np.nan], [1.0]), ("inf - inf", [np.inf], [np.inf])]:
    A, B = np.array(a, np.float32), np.array(b, np.float32)
    r, w = _w(lambda A=A, B=B: A - B)
    print(f"  {name}: result={r[0]}  warn={w or '无'}")

print("=== 1) FeatureStore 内省（定罪点①）===", flush=True)
store = FeatureStore(FEAT)
mm = store.seq
print(f"  type={type(mm).__name__}  dtype={mm.dtype}  shape={mm.shape}")
print(f"  offset={getattr(mm, 'offset', None)}   mode={getattr(mm, 'mode', None)}")
print(f"  >>> offset=128 → 清白；offset=0 → 定罪（头被当数据）")

print("=== 2) 同进程三读 row 0 ===", flush=True)
row_s = np.asarray(mm[0])
row_r = np.asarray(np.load(os.path.join(FEAT, "seq.npy"), mmap_mode="r")[0])
src = np.load(os.path.join(FEAT, f"seq_{P0}.npy"))
for name, a in [("store(FeatureStore)", row_s), ("raw(np.load)    ", row_r), ("src(散文件)     ", src)]:
    print(f"  {name}: dtype={a.dtype} shape={a.shape} nan={int(np.isnan(a).sum())} "
          f"inf={int(np.isinf(a).sum())} max|fin|={float(np.abs(a[np.isfinite(a)]).max()):.3f}")
print(f"  store==raw 逐位: {np.array_equal(row_s, row_r)}")
print(f"  store[:115]==fp16(src[:115]): {np.array_equal(row_s[:115], np.asarray(src[:115], np.float16))}")

print("=== 3) store 行内非有限定位 ===", flush=True)
bad = ~np.isfinite(row_s)
if bad.any():
    flat = np.where(~np.isfinite(row_s.ravel()))[0]
    print(f"  非有限展平位置: {flat[:10].tolist()}  （集中在 <64 → 文件头被当数据）")
else:
    print("  全部有限 → 指纹A 排除，等 data.py 走指纹B（变换型）")

print("=== 4) 头字节检验（定罪点②）===", flush=True)
with open(os.path.join(FEAT, "seq.npy"), "rb") as f:
    hdr16 = np.frombuffer(f.read(128), np.float16)
print(f"  seq.npy 头128字节 as fp16: inf={int(np.isinf(hdr16).sum())} nan={int(np.isnan(hdr16).sum())}")
print(f"  store.seq[0] 前64元素 == 头 as fp16: {np.array_equal(row_s.ravel()[:64], hdr16)}")

print("=== 5) 系统性：多行 store vs raw ===", flush=True)
rawmm = np.load(os.path.join(FEAT, "seq.npy"), mmap_mode="r")
for i in [1, 2, 3000, 11017]:
    print(f"  row {i}: 逐位相等={np.array_equal(np.asarray(mm[i]), np.asarray(rawmm[i]))}")
print("PROBE_DONE", flush=True)
