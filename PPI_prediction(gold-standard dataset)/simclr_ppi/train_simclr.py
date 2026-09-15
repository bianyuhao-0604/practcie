"""Stage 1: 对比预训练。每 PROBE_EVERY 轮在 Intra0 上跑线性探针，
以探针准确率选最优 Adapter（Intra0 即官方 val split，选模型协议合规）。"""
import os, csv, argparse, random, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from config import *
from simclr_data import SimCLRData
from simclr_augment import EmbedAugment
from simclr_model import Adapter, ProjectionHead
from simclr_loss import hybrid_loss
from prep_embeddings import read_pairs

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

class IdxDS(Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return i

@torch.no_grad()
def probe_features(adapter, X_all, all2row, df, device, n_pairs):
    """正负各采 n_pairs//2 条对，打乱后五五分训测；保证两半两类齐全"""
    pos, neg = df[df.y == 1], df[df.y == 0]
    k = min(n_pairs // 2, len(pos), len(neg))
    sub = pd.concat([pos.sample(k, random_state=0),
                     neg.sample(k, random_state=1)], ignore_index=True)
    sub = sub[sub.a.isin(all2row) & sub.b.isin(all2row)]
    sub = sub.sample(frac=1.0, random_state=2).reset_index(drop=True)
    prots = pd.unique(pd.concat([sub.a, sub.b]))
    rows = np.array([all2row[p] for p in prots])
    H = torch.from_numpy(X_all[rows]).to(device)
    if adapter is not None:
        was = adapter.training; adapter.eval()
        H = adapter(H)
        if was: adapter.train()
    Hp = H.cpu().numpy()
    rm = {p: i for i, p in enumerate(prots)}
    Xf = np.hstack([Hp[[rm[a] for a in sub.a]], Hp[[rm[b] for b in sub.b]]])
    return Xf, sub.y.values

def probe_acc(adapter, X_all, all2row, df, device, n_pairs):
    Xf, y = probe_features(adapter, X_all, all2row, df, device, n_pairs)
    half = len(y) // 2
    clf = LogisticRegression(max_iter=1000).fit(Xf[:half], y[:half])
    return clf.score(Xf[half:], y[half:])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--tag", type=str, default="main")
    args = ap.parse_args()
    seed_all(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb = os.environ.get("USE_WANDB") == "1"
    if use_wandb:
        import wandb; wandb.init(project="hypertune", config=vars(args))

    prep = np.load(PREP_NPZ)
    X_all, all2row = prep["X"], {k: i for i, k in enumerate(prep["ids"].tolist())}
    data = SimCLRData(PREP_NPZ, SPLIT_DIR)
    data.X = data.X.to(dev)   # 22MB 整卡驻留, 顺带免去每步 CPU→GPU 拷贝

    aug = EmbedAugment(DIM_MASK_P, NOISE_STD, SCALE_RANGE)
    dl = DataLoader(IdxDS(data.N), batch_size=args.batch, shuffle=True,
                    drop_last=True, num_workers=0)

    va = pd.concat([read_pairs(SPLIT_DIR / "Intra0_pos_rr.txt").assign(y=1),
                    read_pairs(SPLIT_DIR / "Intra0_neg_rr.txt").assign(y=0)], ignore_index=True)
    assert (va.y == 1).sum() > 0 and (va.y == 0).sum() > 0, "Intra0 缺类, 数据异常"
    acc_raw = probe_acc(None, X_all, all2row, va, dev, PROBE_NPAIRS)
    print(f"[probe] 原始嵌入基线 acc = {acc_raw:.4f}")

    emb_dim = int(X_all.shape[1])            # = 1280, 直接从数据读
    adapter = Adapter(dim=emb_dim, hidden=emb_dim, spec_norm=SPEC_NORM).to(dev)
    proj    = ProjectionHead(dim=emb_dim, out=PROJ_DIM, spec_norm=SPEC_NORM).to(dev)

    params = list(adapter.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, hist = -1.0, []
    for ep in range(args.epochs):
        tot = ti = ts = nb = 0
        for idx in dl:
            idx = idx.to(dev)
            h = data.X[idx].to(dev)
            z1 = proj(adapter(aug(h)))
            z2 = proj(adapter(aug(h)))
            M = data.pos_mask(idx, dev)
            loss, li, ls = hybrid_loss(z1, z2, M, args.temp, args.alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); ti += li; ts += ls; nb += 1
        sched.step()
        msg = f"ep {ep+1:3d} loss {tot/nb:.4f} (inst {ti/nb:.4f} sup {ts/nb:.4f})"
        rec = {"epoch": ep + 1, "loss": tot/nb, "inst": ti/nb, "sup": ts/nb}
        if (ep + 1) % PROBE_EVERY == 0 or ep == args.epochs - 1:
            acc = probe_acc(adapter, X_all, all2row, va, dev, PROBE_NPAIRS)
            rec["probe"] = acc; msg += f" | probe {acc:.4f} (Δ {(acc-acc_raw)*100:+.2f}pp)"
            if acc > best:
                best = acc
                torch.save(adapter.state_dict(), ADAPTER_CKPT)
                msg += " ← best, saved"
        hist.append(rec); print(msg, flush=True)
        if use_wandb: wandb.log(rec)

    with open(SIMCLR_LOG, "w", newline="", encoding="utf-8") as f:
            fieldnames = ['ep', 'loss', 'inst', 'sup', 'probe']
            w = csv.DictWriter(f, fieldnames=fieldnames, restval=''); w.writeheader(); w.writerows(hist)
    print(f"完成。最优探针 Δ = {(best-acc_raw)*100:+.2f}pp, Adapter 已存 {ADAPTER_CKPT}")
    print("止损判据: Δ < +0.5pp → 嵌入级重塑饱和, 停止该路线")

if __name__ == "__main__":
    main()
