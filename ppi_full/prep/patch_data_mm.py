p = "data.py"
b = open(p, "rb").read()
old = b'            return np.memmap(p, dtype=np.float16, mode="r", shape=shape) if os.path.exists(p) else None'
new = (b'            if not os.path.exists(p):\n'
       b'                return None\n'
       b'            m = np.load(p, mmap_mode="r")\n'
       b'            assert tuple(m.shape) == tuple(shape), f"{name}.npy {tuple(m.shape)} != {tuple(shape)}"\n'
       b'            return m')
n = b.count(old)
if n == 0:
    assert b.count(b'm = np.load(p, mmap_mode="r")') == 1, "patch pattern missing - stop"
    print("ALREADY PATCHED - skip")
elif n == 1:
    open(p, "wb").write(b.replace(old, new))
    print("PATCHED: _mm np.memmap(offset=0) -> np.load(mmap_mode=r) + shape assert")
else:
    raise SystemExit(f"AMBIGUOUS: {n} matches - stop")
compile(open(p, "rb").read(), p, "exec")
print("COMPILE OK")

import glob
hits = 0
for f in sorted(glob.glob("*.py") + glob.glob("prep/*.py") + glob.glob("features/*.py")):
    try:
        lines = open(f, "rb").readlines()
    except OSError:
        continue
    for i, ln in enumerate(lines, 1):
        if b"memmap" in ln and b"open_memmap" not in ln:
            hits += 1
            print(f"  RISK {f}:{i}: {ln.decode('utf-8','replace').strip()[:100]}")
print(f"MEMMAP_SCAN hits={hits} (0 = no other bare np.memmap)")
