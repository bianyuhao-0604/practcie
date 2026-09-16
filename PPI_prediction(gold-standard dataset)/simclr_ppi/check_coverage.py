# check_coverage.py — 检查三个 block 的蛋白是否都在 npz 里
import numpy as np
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

for arm in ["prep", "simclr"]:
    z = np.load(BASE / "simclr_out" / f"embeddings_{arm}.npz")
    ids = set(z["ids"].astype(str))
    print(f"\n[{arm}] N={len(ids)}")
    for role, blk in [("train", "Intra1"), ("val", "Intra0"), ("test", "Intra2")]:
        need = set()
        for suf in ["pos", "neg"]:
            for line in open(BASE / "dataset" / f"{blk}_{suf}_rr.txt", encoding="utf-8"):
                p = line.split()
                if len(p) >= 2:
                    need |= {p[0], p[1]}
        miss = need - ids
        print(f"  {role}<-{blk}: 需要 {len(need)} 蛋白, 缺 {len(miss)}"
              + (f"  示例: {sorted(miss)[:5]}" if miss else "  ✓"))
