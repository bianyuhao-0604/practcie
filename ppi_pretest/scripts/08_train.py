# scripts/08_train.py — 逐残基双塔版: esm2/prostt5 各自 (256,D)+独立掩码; text/genome 蛋白级向量
import sys, json, time, random, argparse, gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import *
from model import PPIModel

FEAT_DEV = "cuda"   # 特征常驻设备; 显存紧张(<6GB)改 "cpu", 自动逐批上卡


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_modality(ids, name):
    X = np.stack([np.load(FEATURES / name / f"{p}.npy") for p in ids])  # [N,256,D]
    lens = json.loads((FEATURES / name / "lengths.json").read_text())
    L = np.array([int(lens[p]) for p in ids], dtype=np.int64)
    return X, L


def build_tensors(flags):
    ids = (DATA / "proteins.txt").read_text().split()
    Xe, Le = load_modality(ids, "esm2")
    print(f"esm2: {Xe.shape}  长度范围 [{Le.min()}, {Le.max()}]")
    Xp, Lp = None, None
    if flags["use_struct"]:
        Xp, Lp = load_modality(ids, "prostt5")
        uniq = sorted(set((Lp - Le).tolist()))
        print(f"prostt5: {Xp.shape}  长度范围 [{Lp.min()}, {Lp.max()}]  "
              f"长度差(Lp-Le) 唯一值={uniq}")
        if len(uniq) == 1:
            print(f"  常数差 {uniq[0]} = prostt5 分词器附加 token 数; "
                  f"双塔各自掩码, 无需 token 对齐")
    blocks = []
    if flags["use_text"]:
        blocks.append(np.stack([np.load(FEATURES / "text" / f"{p}.npy")
                                for p in ids]).astype(np.float32))
    if flags["use_genome"]:
        blocks.append(np.stack([np.load(FEATURES / "genome" / f"{p}.npy")
                                for p in ids]).astype(np.float32))
    Xg = np.concatenate(blocks, axis=1) if blocks else None
    if Xg is not None:
        print(f"全局矩阵: {Xg.shape}")
    return ids, Xe, Le, Xp, Lp, Xg


def load_pairs(csv_path, id2row):
    df = pd.read_csv(csv_path)
    ca, cb, cy = "pidA", "pidB", "label"
    assert {ca, cb, cy}.issubset(df.columns), f"{csv_path.name}: 列={list(df.columns)}"
    ok = df[ca].isin(id2row) & df[cb].isin(id2row)
    if (~ok).any():
        print(f"  {csv_path.name}: 跳过 {int((~ok).sum())} 行未知ID")
    df = df[ok]
    ia = df[ca].map(id2row).to_numpy(np.int64)
    ib = df[cb].map(id2row).to_numpy(np.int64)
    y = df[cy].astype(np.float32).to_numpy()
    print(f"  {csv_path.name}: {len(y)} 对  正例率={y.mean():.3f}")
    return ia, ib, y


def fetch(i, Xe, Me, Xp, Mp, Xg, dev):
    """按蛋白索引批量取特征并搬到 dev (未启用模态返回 None)"""
    xs = Xe[i].to(dev); ms = Me[i].to(dev)
    xp = Xp[i].to(dev) if Xp is not None else None
    mp = Mp[i].to(dev) if Mp is not None else None
    g = Xg[i].to(dev) if Xg is not None else None
    return xs, ms, xp, mp, g


@torch.no_grad()
def evaluate(model, Xe, Me, Xp, Mp, Xg, ia, ib, y, dev, bs=512):
    model.eval(); ps = []
    for s in range(0, len(y), bs):
        a, b = ia[s:s + bs], ib[s:s + bs]
        xsA, msA, xpA, mpA, ga = fetch(a, Xe, Me, Xp, Mp, Xg, dev)
        xsB, msB, xpB, mpB, gb = fetch(b, Xe, Me, Xp, Mp, Xg, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_bf16_supported()):
            logits = model(xsA, xsB, msA, msB, xpA, xpB, mpA, mpB, ga, gb)
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(ps), y


def train_one(r_stage, ablation, seed):
    set_seed(seed)
    dev = "cuda"
    flags = {"use_struct": r_stage in ("R3", "R4", "R5", "R6", "R7"),
             "use_text":   r_stage in ("R6", "R7") and ablation != "no_global",
             "use_genome": r_stage in ("R7",) and ablation != "no_global"}

    ids, Xe, Le, Xp, Lp, Xg = build_tensors(flags)
    id2row = {p: i for i, p in enumerate(ids)}
    Xe = torch.from_numpy(Xe).to(FEAT_DEV)
    Me = (torch.arange(MAX_LEN)[None, :] < torch.from_numpy(Le)[:, None]).to(FEAT_DEV)
    if Xp is not None:
        Xp = torch.from_numpy(Xp).to(FEAT_DEV)
        Mp = (torch.arange(MAX_LEN)[None, :] < torch.from_numpy(Lp)[:, None]).to(FEAT_DEV)
    else:
        Mp = None
    Xg = torch.from_numpy(Xg).to(FEAT_DEV) if Xg is not None else None

    ia_tr, ib_tr, y_tr = load_pairs(DATA / "pairs_tr.csv", id2row)
    ia_va, ib_va, y_va = load_pairs(DATA / "pairs_va.csv", id2row)
    ia_te, ib_te, y_te = load_pairs(DATA / "pairs_te.csv", id2row)
    ia_tr = torch.from_numpy(ia_tr).to(FEAT_DEV)
    ib_tr = torch.from_numpy(ib_tr).to(FEAT_DEV)
    ia_va = torch.from_numpy(ia_va).to(FEAT_DEV); ib_va = torch.from_numpy(ib_va).to(FEAT_DEV)
    ia_te = torch.from_numpy(ia_te).to(FEAT_DEV); ib_te = torch.from_numpy(ib_te).to(FEAT_DEV)
    y_tr_t = torch.from_numpy(y_tr).to(dev)

    model = PPIModel(flags, d_seq=ESM2_DIM, d_str=PROSTT5_DIM,
                      d_txt=TEXT_DIM, d_gen=GENOME_DIM,
                      d_fuse=D_FUSE, p=DROPOUT).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_par:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n_tr = len(y_tr)
    steps = (n_tr + BATCH - 1) // BATCH
    total = EPOCHS * steps
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))

    best, bad, hist = 0.0, 0, []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); losses = []
        perm = torch.randperm(n_tr, device=FEAT_DEV)
        for s in range(0, n_tr, BATCH):
            idx = perm[s:s + BATCH]
            xsA, msA, xpA, mpA, ga = fetch(ia_tr[idx], Xe, Me, Xp, Mp, Xg, dev)
            xsB, msB, xpB, mpB, gb = fetch(ib_tr[idx], Xe, Me, Xp, Mp, Xg, dev)
            yv = y_tr_t[idx.to(dev)]
            y_s = yv * (1 - LS_EPS) + (1 - yv) * LS_EPS
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_bf16_supported()):
                logits = model(xsA, xsB, msA, msB, xpA, xpB, mpA, mpB, ga, gb)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y_s)
                if RDROP_ALPHA > 0 and ablation != "no_rdrop":
                    logits2 = model(xsA, xsB, msA, msB, xpA, xpB, mpA, mpB, ga, gb)
                    p1 = torch.sigmoid(logits).float()
                    p2 = torch.sigmoid(logits2).float()
                    kl = 0.5 * (F.kl_div((p2 + 1e-7).log(), p1, reduction="batchmean")
                                + F.kl_div((p1 + 1e-7).log(), p2, reduction="batchmean"))
                    loss = loss + RDROP_ALPHA * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())

        pv, yv = evaluate(model, Xe, Me, Xp, Mp, Xg, ia_va, ib_va, y_va, dev)
        auprc = average_precision_score(yv, pv)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_auprc": auprc})
        print(f"[{r_stage}|{ablation}|s{seed}] ep{ep} loss={np.mean(losses):.4f} "
              f"valAUPRC={auprc:.4f} ({time.time() - t0:.0f}s) "
              f"VRAM={torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
        if auprc > best:
            best = auprc; bad = 0
            torch.save(model.state_dict(), CKPT / f"{r_stage}_{ablation}_s{seed}.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stop"); break

    model.load_state_dict(torch.load(CKPT / f"{r_stage}_{ablation}_s{seed}.pt",
                                     weights_only=True))
    pt, yt = evaluate(model, Xe, Me, Xp, Mp, Xg, ia_te, ib_te, y_te, dev)
    m = {"r_stage": r_stage, "ablation": ablation, "seed": seed, "params": n_par,
         "best_val_auprc": best,
         "test_auprc": average_precision_score(yt, pt),
         "test_acc": accuracy_score(yt, (pt > 0.5).astype(int)),
         "test_auroc": roc_auc_score(yt, pt)}
    (RESULTS / f"metrics_{r_stage}_{ablation}_{seed}.json").write_text(
        json.dumps(m, indent=2), encoding="utf-8")
    (RESULTS / f"hist_{r_stage}_{ablation}_{seed}.json").write_text(
        json.dumps(hist, indent=2), encoding="utf-8")
    print("TEST:", m)
    del Xe, Me, Xp, Mp, Xg, ia_tr, ib_tr, y_tr_t
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-stage", default="R7")
    ap.add_argument("--ablation", default="none")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train_one(a.r_stage, a.ablation, a.seed)
