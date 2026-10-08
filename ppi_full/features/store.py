"""特征库写入端（P0-4 布局分离）+ collect CLI（零填充对齐）。"""
import argparse, json, os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LEN, Paths

RES_LAYOUT = {"seq": (1280,), "str": (1024,), "surf": (80,), "iface": ()}
PROT_LAYOUT = {"text": (128,), "genome": (256,)}

class StoreWriter:
    def __init__(self, feat_dir, pids):
        os.makedirs(feat_dir, exist_ok=True)
        self.dir = feat_dir
        self.n = len(pids)
        self.pid2idx = {p: i for i, p in enumerate(pids)}
        self.lengths = np.zeros(self.n, np.int32)
        self.plddt = np.zeros((self.n, MAX_LEN), np.float16)
        self.has_text = np.zeros(self.n, bool)
        self.has_genome = np.zeros(self.n, bool)
        self.arrs = {}
        for k, tail in RES_LAYOUT.items():
            shape = (self.n, MAX_LEN, *tail) if tail else (self.n, MAX_LEN)
            self.arrs[k] = np.lib.format.open_memmap(
                os.path.join(feat_dir, k + ".npy"), mode="w+", dtype=np.float16, shape=shape)
        for k, dim in PROT_LAYOUT.items():
            self.arrs[k] = np.lib.format.open_memmap(
                os.path.join(feat_dir, k + ".npy"), mode="w+", dtype=np.float16, shape=(self.n, *dim))

    def write(self, pid, length, plddt=None, **feat):
        i = self.pid2idx[pid]
        L = min(int(length), MAX_LEN)
        self.lengths[i] = L
        if plddt is not None:
            self.plddt[i, :L] = np.asarray(plddt[:L], np.float16)
        else:
            self.plddt[i, :L] = np.float16(100.0)          # 缺失 → 掩码不生效
        for k, v in feat.items():
            if v is None:
                continue
            v = np.asarray(v, np.float16)
            if k in PROT_LAYOUT:
                self.arrs[k][i] = v                        # P0-4：蛋白级独立布局
                if k == "text":
                    self.has_text[i] = True                # P0-4：置位可达
                else:
                    self.has_genome[i] = True
            else:
                self.arrs[k][i, :L] = v[:L]

    def flush_meta(self, extra=None):
        for k in self.arrs:
            self.arrs[k].flush()
        np.save(os.path.join(self.dir, "lengths.npy"), self.lengths)
        np.save(os.path.join(self.dir, "plddt.npy"), self.plddt)
        meta = dict(pid2idx=self.pid2idx, n=self.n,
                    has_text=self.has_text.tolist(), has_genome=self.has_genome.tolist())
        if extra:
            meta.update(extra)
        json.dump(meta, open(os.path.join(self.dir, "meta.json"), "w"))

def _fit(v, L, D):
    out = np.zeros((L, D), np.float16)
    if v is not None:
        c = min(len(v), L)
        out[:c] = v[:c]
    return out

def _fit1(v, L):
    out = np.zeros(L, np.float16)
    if v is not None:
        c = min(len(v), L)
        out[:c] = v[:c]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--feat-dir", default=None)
    a = ap.parse_args()
    feat_dir = a.feat_dir or Paths().feat_dir
    pids = [l.strip() for l in open(a.pids) if l.strip()]
    if a.init:
        w = StoreWriter(feat_dir, pids)
        w.flush_meta(extra=dict(note="skeleton; run extractors then --collect"))
        print(f"init ok: N={len(pids)} -> {feat_dir}")
        return
    if a.collect:
        w = StoreWriter(feat_dir, pids)
        warn = dict(no_str=0, no_surf=0, no_iface=0, no_text=0, no_genome=0, len_mismatch=0)
        for i, pid in enumerate(pids):
            seq = np.load(os.path.join(feat_dir, f"seq_{pid}.npy"))
            L = min(len(seq), MAX_LEN)
            def _t(kind):
                p = os.path.join(feat_dir, f"{kind}_{pid}.npy")
                return np.load(p) if os.path.exists(p) else None
            st, su, ic, pl = _t("str"), _t("surf"), _t("iface"), _t("plddt")
            tx, ge = _t("text"), _t("genome")
            if st is None: warn["no_str"] += 1
            if su is None: warn["no_surf"] += 1
            if ic is None: warn["no_iface"] += 1
            if tx is None: warn["no_text"] += 1
            if ge is None: warn["no_genome"] += 1
            if pl is not None and len(pl) != len(seq):
                warn["len_mismatch"] += 1
            w.write(pid, length=L,
                    plddt=np.asarray(pl, np.float16) if pl is not None else None,
                    seq=_fit(seq, L, 1280), str=_fit(st, L, 1024), surf=_fit(su, L, 80),
                    iface=_fit1(ic, L), text=tx, genome=ge)
            if i % 500 == 0:
                print(f"collect {i}/{len(pids)}")
        w.flush_meta(extra=dict(warn=warn))
        print("collect done; warns:", warn)

if __name__ == "__main__":
    main()
