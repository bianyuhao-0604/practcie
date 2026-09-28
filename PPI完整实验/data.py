"""L0 数据层 + L2 质量过滤 + FeatureStore 读取端 + figshare→csv 转换 CLI。
补丁：P7(蛋白不相交断言)、B3(时间截断)、P0-4(读取端布局)。
自查修正：蛋白级特征按侧独立收集（此前版本误共享 A 侧）。"""
import argparse, json, os, random
import numpy as np
import pandas as pd
import torch
from config import MAX_LEN

COL_ALIASES = {
    "pid_a": ["pid_a", "protein1", "uniprot_a", "id1", "p1", "a"],
    "pid_b": ["pid_b", "protein2", "uniprot_b", "id2", "p2", "b"],
    "label": ["label", "y", "interaction", "is_interaction"],
    "date":  ["date", "evidence_date", "updated", "first_seen"],
}

def load_pairs(csv_path, date_col=None, cutoff=None):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    ren = {}
    for std, aliases in COL_ALIASES.items():
        for c in df.columns:
            if str(c).lower() in aliases:
                ren[c] = std
                break
    df = df.rename(columns=ren)
    assert {"pid_a", "pid_b", "label"} <= set(df.columns), f"columns unrecognised: {list(df.columns)}"
    keep = ["pid_a", "pid_b"] + (["date"] if "date" in df.columns else []) + ["label"]
    df = df[keep].copy()
    df["pid_a"] = df["pid_a"].astype(str)
    df["pid_b"] = df["pid_b"].astype(str)
    df["label"] = df["label"].astype(np.float32)
    if date_col and cutoff:
        if date_col not in df.columns:
            raise ValueError(f"date_col '{date_col}' not in {list(df.columns)}")
        n0 = len(df)
        df = df[pd.to_datetime(df[date_col]) <= pd.Timestamp(cutoff)].reset_index(drop=True)
        print(f"[time cutoff {cutoff}] {n0} -> {len(df)} rows")
    return df

def validate_splits(splits):
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pa = set(splits[names[i]].pid_a) | set(splits[names[i]].pid_b)
            pb = set(splits[names[j]].pid_a) | set(splits[names[j]].pid_b)
            inter = sorted(pa & pb)
            assert not inter, (f"LEAKAGE: proteins shared between '{names[i]}' and "
                               f"'{names[j]}': {len(inter)}, e.g. {inter[:5]}")
    return True

class FeatureStore:
    def __init__(self, feat_dir):
        self.dir = feat_dir
        meta = json.load(open(os.path.join(feat_dir, "meta.json")))
        self.pid2idx = meta["pid2idx"]
        self.n = int(meta["n"])
        self.has_text = np.array(meta["has_text"], bool)
        self.has_genome = np.array(meta["has_genome"], bool)
        self.lengths = np.load(os.path.join(feat_dir, "lengths.npy"), mmap_mode="r")
        self.plddt = np.load(os.path.join(feat_dir, "plddt.npy"), mmap_mode="r")

        def _mm(name, shape):
            p = os.path.join(feat_dir, name + ".npy")
            return np.memmap(p, dtype=np.float16, mode="r", shape=shape) if os.path.exists(p) else None

        self.seq = _mm("seq", (self.n, MAX_LEN, 1280))
        self.str = _mm("str", (self.n, MAX_LEN, 1024))
        self.surf = _mm("surf", (self.n, MAX_LEN, 80))
        self.iface = _mm("iface", (self.n, MAX_LEN))
        self.text = _mm("text", (self.n, 128))
        self.genome = _mm("genome", (self.n, 256))

    def idx(self, pid):
        return self.pid2idx[pid]

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, df, store):
        missing = sorted({p for p in list(df.pid_a) + list(df.pid_b) if p not in store.pid2idx})
        assert not missing, (f"{len(missing)} pids missing from feature store (e.g. {missing[:5]}); "
                             f"re-run feature extraction / build_store --collect")
        self.df = df.reset_index(drop=True)
        self.ia = np.array([store.pid2idx[p] for p in self.df.pid_a], np.int64)
        self.ib = np.array([store.pid2idx[p] for p in self.df.pid_b], np.int64)
        self.y = self.df.label.values.astype(np.float32)

    def __len__(self):
        return len(self.ia)

    def __getitem__(self, i):
        return int(self.ia[i]), int(self.ib[i]), float(self.y[i])

def make_collate(store):
    def collate(items):
        ias = [it[0] for it in items]
        ibs = [it[1] for it in items]
        ys = [it[2] for it in items]
        La = np.asarray(store.lengths)[ias].astype(int)
        Lb = np.asarray(store.lengths)[ibs].astype(int)
        Lmax = min(MAX_LEN, max(int(La.max()), int(Lb.max())))
        Lmax = ((Lmax + 7) // 8) * 8
        B = len(items)

        def side(idxs, Ls):
            d = {}
            for k in ("seq", "str", "surf"):
                src = getattr(store, k)
                D = src.shape[-1]
                arr = np.zeros((B, Lmax, D), np.float32)
                for jj, (ix, L) in enumerate(zip(idxs, Ls)):
                    Lc = min(int(L), Lmax)
                    if Lc > 0:
                        arr[jj, :Lc] = np.asarray(src[ix, :Lc], np.float32)
                d[k] = arr
            iface = np.zeros((B, Lmax), np.float32)
            plddt = np.zeros((B, Lmax), np.float32)
            for jj, (ix, L) in enumerate(zip(idxs, Ls)):
                Lc = min(int(L), Lmax)
                if Lc > 0:
                    iface[jj, :Lc] = np.asarray(store.iface[ix, :Lc], np.float32)
                    plddt[jj, :Lc] = np.asarray(store.plddt[ix, :Lc], np.float32)
            d["len"] = np.minimum(np.asarray(Ls), Lmax).astype(int)
            d["iface_raw"] = iface
            d["plddt"] = plddt
            # 蛋白级特征：per-side 独立收集（自查修正点）
            tx = np.zeros((B, 128), np.float32)
            ge = np.zeros((B, 256), np.float32)
            tm = np.zeros(B, np.float32)
            gm = np.zeros(B, np.float32)
            for jj, ix in enumerate(idxs):
                if store.text is not None and store.has_text[ix]:
                    tx[jj] = np.asarray(store.text[ix], np.float32)
                    tm[jj] = 1.0
                if store.genome is not None and store.has_genome[ix]:
                    ge[jj] = np.asarray(store.genome[ix], np.float32)
                    gm[jj] = 1.0
            d["text"] = tx
            d["genome"] = ge
            d["text_mask"] = tm
            d["genome_mask"] = gm
            return d

        A, Bd = side(ias, La), side(ibs, Lb)
        for d in (A, Bd):                                # L2：pLDDT<70 → surf/iface 置零（无参数）
            keep70 = (d["plddt"] >= 70.0).astype(np.float32)
            d["surf"] = d["surf"] * keep70[..., None]
            d["iface"] = d["iface_raw"] * keep70

        def pack(d):
            pad = np.arange(Lmax)[None, :] >= d["len"][:, None]
            return {
                "seq": torch.from_numpy(np.ascontiguousarray(d["seq"])),
                "str": torch.from_numpy(np.ascontiguousarray(d["str"])),
                "surf": torch.from_numpy(np.ascontiguousarray(d["surf"])),
                "iface": torch.from_numpy(np.ascontiguousarray(d["iface"])),
                "pad": torch.from_numpy(pad),
                "text": torch.from_numpy(np.ascontiguousarray(d["text"])),
                "genome": torch.from_numpy(np.ascontiguousarray(d["genome"])),
                "text_mask": torch.from_numpy(d["text_mask"]),
                "genome_mask": torch.from_numpy(d["genome_mask"]),
            }

        return {"A": pack(A), "B": pack(Bd), "y": torch.tensor(ys, dtype=torch.float32)}
    return collate

class BucketSampler(torch.utils.data.Sampler):
    """按 max(La,Lb) 排序分桶 → 桶内成批 → 每 epoch 打乱批序（省 padding 算力；确定性可复现）。"""
    def __init__(self, ds, store, bs, seed, frac=1.0):
        n = len(ds)
        take = int(n * frac) if frac < 1.0 else n
        rng = random.Random(seed)
        idxs = np.array(rng.sample(range(n), take))
        Ls = np.asarray(store.lengths)
        L = np.maximum(Ls[ds.ia[idxs]], Ls[ds.ib[idxs]])
        order = idxs[np.argsort(L, kind="stable")]
        self.batches = [order[k:k + bs].tolist() for k in range(0, len(order), bs)]
        self._rng = random.Random(seed)

    def __iter__(self):
        bs = list(self.batches)
        self._rng.shuffle(bs)
        for b in bs:
            yield b

    def __len__(self):
        return len(self.batches)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="原始互作文件 → 标准化 csv (pid_a,pid_b[,date],label)")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pid-col-a", default=None)
    ap.add_argument("--pid-col-b", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--date-col", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.inp, sep=None, engine="python")
    ren = {}
    for std, col in (("pid_a", a.pid_col_a), ("pid_b", a.pid_col_b),
                     ("label", a.label_col), ("date", a.date_col)):
        if col:
            assert col in df.columns, f"{col} not in {list(df.columns)}"
            ren[col] = std
    if "pid_a" not in ren:
        for std, aliases in COL_ALIASES.items():
            for c in df.columns:
                if str(c).lower() in aliases and std not in ren.values():
                    ren[c] = std
                    break
    df = df.rename(columns=ren)
    need = [c for c in ("pid_a", "pid_b", "label") if c not in df.columns]
    assert not need, f"missing standard columns {need}; 请用 --pid-col-a/--pid-col-b/--label-col 显式指定"
    cols = ["pid_a", "pid_b"] + (["date"] if "date" in df.columns else []) + ["label"]
    df[cols].to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows, cols={cols}  ← 请人工抽检 5 行核对 pid/label 语义")
