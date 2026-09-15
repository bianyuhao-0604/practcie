"""Stage 3: fc2_20_2_dense 单变量对比。--npz 指定嵌入库（prep=对照, simclr=实验），
多 seed 训练并保存测试集预测；--compare 做配对置换检验。"""
import argparse, json, sys, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from config import BASE, SPLIT_DIR, OUT_DIR, FC2_EPOCHS, FC2_BATCH, FC2_LR, \
                   FC2_PATIENCE, N_SEEDS, N_PERM, SEED
sys.path.insert(0, str(BASE))
from models.fc2_20_2_dense import FC2_20_2Dense
from prep_embeddings import read_pairs

def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_split(split):
    return pd.concat([read_pairs(SPLIT_DIR / f"{split}_pos_rr.txt").assign(y=1),
                      read_pairs(SPLIT_DIR / f"{split}_neg_rr.txt").assign(y=0)],
                     ignore_index=True)

class PairDS(Dataset):
    def __init__(self, df, E, id2row):
        ok = df.a.isin(id2row) & df.b.isin(id2row)
        self.dropped = int((~ok).sum())
        df = df[ok].reset_index(drop=True)
        self.a = np.array([id2row[p] for p in df.a])
        self.b = np.array([id2row[p] for p in df.b])
        self.y = df.y.values.astype(np.float32)
        self.E = torch.from_numpy(E)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.stack([self.E[self.a[i]], self.E[self.b[i]]]), self.y[i]

@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); ps, ys = [], []
    for x, y in dl:
        ps.append(model(x.to(dev)).cpu()); ys.append(y)
    return torch.cat(ps).numpy().ravel(), torch.cat(ys).numpy()

def metrics(y, p):
    pred = (p > 0.5).astype(int)
    return {"acc": float((pred == y).mean()),
            "precision": float(precision_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "aupr": float(average_precision_score(y, p))}

def run_tag(npz_path, tag, n_seeds, spec_norm=False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(npz_path)
    E, id2row = d["X"], {k: i for i, k in enumerate(d["ids"].tolist())}
    splits = {s: load_split(s) for s in ["Intra1", "Intra0", "Intra2"]}
    tr_ds = PairDS(splits["Intra1"], E, id2row)
    va_ds = PairDS(splits["Intra0"], E, id2row)
    te_ds = PairDS(splits["Intra2"], E, id2row)
    print(f"[{tag}] 缺嵌入对: train {tr_ds.dropped}, val {va_ds.dropped}, test {te_ds.dropped}")
    tr_dl = DataLoader(tr_ds, FC2_BATCH, shuffle=True, num_workers=0, drop_last=False)
    va_dl = DataLoader(va_ds, FC2_BATCH, num_workers=0)
    te_dl = DataLoader(te_ds, FC2_BATCH, num_workers=0)

    all_results, test_preds = [], []
    for s_i in range(n_seeds):
        seed = SEED + s_i; seed_all(seed)
        model = FC2_20_2Dense(embed_dim=1280, spec_norm=spec_norm).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=FC2_LR)
        best_acc, best_state, wait = -1, None, 0
        for ep in range(FC2_EPOCHS):
            model.train()
            for x, y in tr_dl:
                x, y = x.to(dev), y.to(dev).view(-1, 1)
                p = model(x).clamp(1e-6, 1 - 1e-6)
                loss = F.binary_cross_entropy(p, y)
                opt.zero_grad(); loss.backward(); opt.step()
            va_p, va_y = predict(model, va_dl, dev)
            acc = (va_p > 0.5).astype(int)
            acc = float((acc == va_y).mean())
            if acc > best_acc: best_acc, wait = acc, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= FC2_PATIENCE: break
        model.load_state_dict(best_state)
        te_p, te_y = predict(model, te_dl, dev)
        m = metrics(te_y, te_p)
        all_results.append(m); test_preds.append(te_p)
        np.savez(OUT_DIR / f"fc2_{tag}_seed{seed}.npz", y=te_y, p=te_p)
        print(f"[{tag}] seed {seed}: {m} (best val acc {best_acc:.4f})")

    agg = {k: (float(np.mean([r[k] for r in all_results])),
               float(np.std([r[k] for r in all_results]))) for k in all_results[0]}
    print(f"[{tag}] mean±std: " + ", ".join(f"{k}={v[0]:.4f}±{v[1]:.4f}" for k, v in agg.items()))
    json.dump({"tag": tag, "per_seed": all_results, "agg": agg},
              open(OUT_DIR / f"fc2_{tag}_summary.json", "w"), indent=2)

def compare(tag_a, tag_b, n_seeds, n_perm):
    """A 好于 B 的置换检验：两模型各 seed 预测取概率平均后，对 y 置换"""
    pa = np.mean([np.load(OUT_DIR / f"fc2_{tag_a}_seed{SEED+i}.npz")["p"] for i in range(n_seeds)], 0)
    pb = np.mean([np.load(OUT_DIR / f"fc2_{tag_b}_seed{SEED+i}.npz")["p"] for i in range(n_seeds)], 0)
    y = np.load(OUT_DIR / f"fc2_{tag_a}_seed{SEED}.npz")["y"]
    aA, aB = (pa > .5).astype(int), (pb > .5).astype(int)
    obs = (aA == y).mean() - (aB == y).mean()
    rng = np.random.default_rng(SEED); cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if abs((aA == yp).mean() - (aB == yp).mean()) >= abs(obs): cnt += 1
    print(f"Δacc({tag_a} - {tag_b}) = {obs*100:+.2f}pp, 置换检验 p = {(cnt+1)/(n_perm+1):.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--spec_norm", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1], args.seeds, N_PERM)
    else:
        assert args.npz and args.tag
        run_tag(args.npz, args.tag, args.seeds, args.spec_norm)
