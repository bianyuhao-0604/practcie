"""特征库冒烟校验（N6：seq 硬断言，str/surf 阶段性警告）。collect 后必跑。"""
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Paths, MAX_LEN
from data import FeatureStore, PairDataset, make_collate, load_pairs

paths = Paths()
store = FeatureStore(paths.feat_dir)
print(f"N={store.n}  has_text={int(store.has_text.sum())}  has_genome={int(store.has_genome.sum())}")
assert store.seq.shape == (store.n, MAX_LEN, 1280), store.seq.shape
assert store.str.shape == (store.n, MAX_LEN, 1024), store.str.shape
assert store.surf.shape == (store.n, MAX_LEN, 80), store.surf.shape
assert store.iface.shape == (store.n, MAX_LEN)
L = np.asarray(store.lengths)
assert (L > 0).all() and (L <= MAX_LEN).all(), f"lengths 异常: min={L.min()} max={L.max()}"
P = np.asarray(store.plddt)
assert (P >= 0).all() and (P <= 100).all(), "pLDDT 越界"
S = np.abs(np.asarray(store.seq[: min(store.n, 256), :64])).max()
print(f"非平凡性: |seq|max={S:.4f}")
assert S > 1e-6, "seq 疑似全零——检查 ESM-2 提取是否成功"
for name, arr in (("str", store.str), ("surf", store.surf)):
    v = np.abs(np.asarray(arr[: min(store.n, 256), :64])).max()
    if v <= 1e-6:
        print(f"WARNING: {name} 全零——若为分阶段提取属预期，否则检查 {name} 通道")
tr_csv = os.path.join(paths.data_dir, "train.csv")
if os.path.exists(tr_csv):
    tr = load_pairs(tr_csv)
    batch = make_collate(store)([PairDataset(tr, store)[i] for i in range(8)])
    for side in ("A", "B"):
        for k, v in batch[side].items():
            print(f"{side}.{k}: {tuple(v.shape)}")
    print(f"y: {tuple(batch['y'].shape)}")
else:
    print("train.csv 不存在，跳过 collate 演示")
print("VERIFY OK")
