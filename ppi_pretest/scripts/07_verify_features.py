"""07_verify_features.py — 逐残基定长格式校验"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import DATA, FEATURES, MAX_LEN
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPECTED = {"esm2": (MAX_LEN, 480), "prostt5": (MAX_LEN, 1024),
            "text": (384,), "genome": (64,)}

def main():
    ids = (DATA / "proteins.txt").read_text().split()
    all_ok = True
    for name, shape in EXPECTED.items():
        d = FEATURES / name
        lens = None
        if name in ("esm2", "prostt5"):
            lj = d / "lengths.json"
            if not lj.exists():
                print(f"{name:8s}: !! 缺 lengths.json (先跑 fix_pad_residue.py)")
                all_ok = False; continue
            lens = json.loads(lj.read_text())
        missing, bad_shape, bad_nan, bad_pad = [], [], [], []
        for p in ids:
            f = d / f"{p}.npy"
            if not f.exists():
                missing.append(p); continue
            a = np.load(f)
            if a.shape != shape:
                bad_shape.append((p, tuple(a.shape))); continue
            if not np.isfinite(a).all():
                bad_nan.append(p)
        if lens is not None:                      # 抽查填充区全零
            for p in ids[:50]:
                if lens.get(p, MAX_LEN) < MAX_LEN:
                    if np.abs(np.load(d / f"{p}.npy")[lens[p]:]).max() != 0:
                        bad_pad.append(p)
        ok = not (missing or bad_shape or bad_nan or bad_pad)
        all_ok = all_ok and ok
        print(f"{name:8s}: {'OK' if ok else '!!'}  文件 {len(ids)-len(missing)}/{len(ids)}  期望 {shape}")
        if missing:   print(f"    缺失 {len(missing)}, 前5: {missing[:5]}")
        if bad_shape: print(f"    形状错 {len(bad_shape)}, 前5: {bad_shape[:5]}")
        if bad_nan:   print(f"    NaN/Inf {len(bad_nan)}, 前5: {bad_nan[:5]}")
        if bad_pad:   print(f"    填充区非零 {len(bad_pad)}, 前5: {bad_pad[:5]}")
    print("\n校验通过" if all_ok else "\n!! 存在问题，见上")

if __name__ == "__main__":
    main()
