"""变长 (L,D) → 定长 (256,D) 零填充 + lengths.json。仅在重提取后运行一次。"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import FEATURES, MAX_LEN

for name, D in [("esm2", 480), ("prostt5", 1024)]:
    d = FEATURES / name
    if (d / "lengths.json").exists():
        print(f"{name}: lengths.json 已存在, 跳过 (重做请先删它)"); continue
    fs = sorted(d.glob("*.npy"))
    if not fs:
        sys.exit(f"!! {d} 无文件: 先完成重提取 (Step 1)")
    lengths = {}
    for i, f in enumerate(fs):
        a = np.load(f)
        if a.ndim != 2:
            sys.exit(f"!! {name}/{f.name} shape={a.shape} 不是 (L,{D}) — "
                     f"仍是池化残留, 回到 Step 0/1")
        assert a.shape[1] == D, f"{f.name} D={a.shape[1]} != {D}"
        L = min(a.shape[0], MAX_LEN)
        out = np.zeros((MAX_LEN, D), dtype=np.float16)
        out[:L] = a[:L]
        np.save(f, out)                     # 覆盖为定长
        lengths[f.stem] = int(L)
        if i % 1000 == 0:
            print(f"  {name}: {i}/{len(fs)}")
    (d / "lengths.json").write_text(json.dumps(lengths))
    print(f"{name}: {len(lengths)} 个 -> ({MAX_LEN},{D}) + lengths.json")
