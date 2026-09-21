import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class PPIPairs(Dataset):
    def __init__(self, csv_path, feats, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.feats = feats          # dict: pid -> {seq, str, text, genome}
        self.L = max_len
    def __len__(self): return len(self.df)
    def _pad(self, x, L):
        out = np.zeros((L, x.shape[-1]), dtype=np.float32)
        n = min(L, x.shape[0]); out[:n] = x[:n]
        return out
    def __getitem__(self, i):
        r = self.df.iloc[i]
        pa, pb = r.pidA, r.pidB
        sa = self.feats[pa]["seq"]; sb = self.feats[pb]["seq"]
        item = {
            "seqA": self._pad(sa, self.L), "seqB": self._pad(sb, self.L),
            "lenA": min(len(sa), self.L), "lenB": min(len(sb), self.L),
            "glbA": np.concatenate([self.feats[pa]["text"], self.feats[pa]["genome"]]).astype(np.float32),
            "glbB": np.concatenate([self.feats[pb]["text"], self.feats[pb]["genome"]]).astype(np.float32),
            "y": np.float32(r.label),
        }
        if self.feats[pa].get("str") is not None:
            item["strA"] = self._pad(self.feats[pa]["str"], self.L)
            item["strB"] = self._pad(self.feats[pb]["str"], self.L)
        return item

def collate(batch):
    out = {}
    for k in ["lenA", "lenB"]: out[k] = torch.tensor([b[k] for b in batch])
    out["y"] = torch.tensor([b["y"] for b in batch])
    out["glbA"] = torch.tensor(np.stack([b["glbA"] for b in batch]))
    out["glbB"] = torch.tensor(np.stack([b["glbB"] for b in batch]))
    for k in ["seqA", "seqB", "strA", "strB"]:
        if k in batch[0]:
            out[k] = torch.tensor(np.stack([b[k] for b in batch]))
    L = out["seqA"].shape[1]
    out["padA"] = torch.arange(L).unsqueeze(0) >= out["lenA"].unsqueeze(1)
    out["padB"] = torch.arange(L).unsqueeze(0) >= out["lenB"].unsqueeze(1)
    return out
