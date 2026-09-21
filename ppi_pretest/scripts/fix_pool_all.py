"""esm2/prostt5 逐残基 (L, D) → 均值池化定长 (D,)，与 text/genome 对齐。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import FEATURES

for name in ["esm2", "prostt5"]:
    d = FEATURES / name
    n = nfix = 0
    dim = None
    for f in d.glob("*.npy"):
        a = np.load(f)
        if a.ndim == 2:                       # fp32 池化避免 fp16 累加误差
            v = a.astype(np.float32).mean(axis=0)
            np.save(f, v.astype(np.float16))
            nfix += 1
            dim = v.shape[0]
        elif a.ndim == 1:
            dim = a.shape[0]                  # 已是池化格式，跳过
        n += 1
    print(f"{name}: 共 {n} 个, 本次池化 {nfix} 个, 定长维度 D = {dim}")
