"""Intra1 蛋白池（严格不接触 Intra0/2）+ 无向 PPI 边表 + SupCon 批内正样本掩码"""
import numpy as np, pandas as pd, torch
from prep_embeddings import read_pairs

class SimCLRData:
    def __init__(self, prep_npz, split_dir):
        d = np.load(prep_npz)
        all2row = {k: i for i, k in enumerate(d["ids"].tolist())}
        tr = pd.concat([read_pairs(split_dir / "Intra1_pos_rr.txt"),
                        read_pairs(split_dir / "Intra1_neg_rr.txt")], ignore_index=True)
        self.ids = pd.unique(pd.concat([tr["a"], tr["b"]])).tolist()
        self.X = torch.from_numpy(d["X"][np.array([all2row[p] for p in self.ids])])
        self.N = len(self.ids)
        self.id2idx = {p: i for i, p in enumerate(self.ids)}
        e = set()
        for a, b in zip(tr["a"], tr["b"]):
            if a in self.id2idx and b in self.id2idx:
                i, j = self.id2idx[a], self.id2idx[b]
                if i != j: e.add((min(i, j), max(i, j)))
        e = torch.tensor(sorted(e), dtype=torch.long)
        self.n_edges = len(e)
        if len(e):
            lo, hi = torch.minimum(e[:, 0], e[:, 1]), torch.maximum(e[:, 0], e[:, 1])
            self.codes = torch.unique(lo * self.N + hi)      # 已排序 int64
        else:
            self.codes = torch.empty(0, dtype=torch.long)
        print(f"SimCLR 蛋白池 {self.N}, PPI 正样本边 {self.n_edges}")

    def pos_mask(self, idx, device):
        """(2n, 2n) 块对角布尔掩码：同视图内为 PPI 伙伴的位置为 True"""
        b, n = idx.to(device), idx.numel()
        pairs = torch.cartesian_prod(b, b)
        lo, hi = torch.minimum(pairs[:, 0], pairs[:, 1]), torch.maximum(pairs[:, 0], pairs[:, 1])
        codes = self.codes.to(device)
        if codes.numel() == 0:
            m = torch.zeros(n, n, dtype=torch.bool, device=device)
        else:
            code = lo * self.N + hi
            pos = torch.searchsorted(codes, code).clamp(max=codes.numel() - 1)
            m = (codes[pos] == code).view(n, n); m.fill_diagonal_(False)
        M = torch.zeros(2 * n, 2 * n, dtype=torch.bool, device=device)
        M[:n, :n], M[n:, n:] = m, m
        return M
