# -*- coding: utf-8 -*-
"""
cross_dscript.py v2 — 交叉格实验: {base, simclr} 嵌入 × D-SCRIPT 风格接触图架构
================================================================================
v1 → v2 的唯一差异 (消融变量): 头部交互容量
  v1: 残基对 concat→线性→接触图→全图均值。均值算子把跨蛋白交互项精确消灭,
      score = g(x_A) + g(x_B) 加性打分器 → Bernett 基准实测 0.5368 (加性天花板)
  v2: 交互头 mean-token concat + Hadamard 乘积 → MLP, 恢复跨蛋白乘性交互
      (沙盒数值验证: v1 头满足加性恒等式 f(a,b)-f(a,c)-f(d,b)+f(d,c)≈0)
目的:
  a) 诊断性消融: v2 base 若从 0.537 跳回 0.58-0.63, 塌缩诊断当场证实
  b) 交叉格判决: Δ(simclr−base) 的 aupr ≥ +0.010 且 ≥2/3 种子同向 → 跨架构成立
数据: Bernett gold-standard 六文件 dataset/Intra{1,0,2}_{pos,neg}_rr.txt
      split: Intra1=train, Intra0=val, Intra2=test (与 train_fc2.py L53-56 一致)
嵌入: simclr_out/embeddings_{prep,simclr}.npz, key = ids / X (mu/sd/W 已烘焙进 X)
协议: 对齐 train_fc2.py — 五指标(阈值0.5), best-val-acc epoch 选点, test 汇报,
      种子 42/43/44, 广播 token 化 L=16 (结论需带此限定)
输出: v2 → simclr_out/cross_results_v2.csv; --v1-head → cross_results_v1head.csv

用法:
  python cross_dscript.py --probe     # 只侦察数据加载, 不训练
  python cross_dscript.py --quick     # 单格校准: base + 首种子
  python cross_dscript.py             # 全表: 2 嵌入 × 3 种子
  python cross_dscript.py --v1-head   # 消融回放: v1 加性头, 应复现 ~0.537
"""
import argparse, copy, csv, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score)

# ---------- 0. 路径与常量 (全部实证, 无猜测) ----------
ROOT     = os.path.dirname(os.path.abspath(__file__))
PPROJ    = os.path.dirname(ROOT)
SIMOUT   = os.path.join(PPROJ, "simclr_out")
DATA_DIR = os.path.join(PPROJ, "dataset")

EMB_PATH = {"base":   os.path.join(SIMOUT, "embeddings_prep.npz"),
            "simclr": os.path.join(SIMOUT, "embeddings_simclr.npz")}
OUT_CSV  = os.path.join(SIMOUT, "cross_results.csv")

SPLITS = {"train": "Intra1", "val": "Intra0", "test": "Intra2"}

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.backends.cudnn.benchmark = True

L_TOKENS, PROJ_DIM, FILTERS, FINAL_C = 16, 100, 64, 25
EPOCHS_DEF, BATCH, LR, DROPOUT = 30, 512, 1e-3, 0.2
SEEDS_DEF = [42, 43, 44]
METRICS = ["acc", "precision", "recall", "f1", "aupr"]
EMB = {}

# ---------- 1. 蛋白池 ----------
def load_pool():
    npzs = {t: np.load(p, allow_pickle=True) for t, p in EMB_PATH.items()}
    ids = [str(x) for x in npzs["base"]["ids"]]
    pid2i = {p: i for i, p in enumerate(ids)}
    for t in EMB_PATH:
        if "ids" in npzs[t].files:
            assert [str(x) for x in npzs[t]["ids"]] == ids, f"{t} 的 ids 与 base 不一致!"
        else:
            assert npzs[t]["X"].shape == npzs["base"]["X"].shape, f"{t} 的 X 形状不一致!"
        EMB[t] = torch.tensor(npzs[t]["X"], dtype=torch.float32, device=DEV)
    return pid2i

# ---------- 2. pairs: Bernett 六文件 ----------
def load_intra(split, pid2i):
    rows = []
    for suffix, y in (("_pos_rr.txt", 1), ("_neg_rr.txt", 0)):
        p = os.path.join(DATA_DIR, split + suffix)
        n_ok = n_miss = n_bad = 0
        with open(p, encoding="utf-8-sig") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 2:
                    if ln.strip():
                        n_bad += 1
                    continue
                a, b = parts[0], parts[1]
                if a in pid2i and b in pid2i:
                    rows.append((pid2i[a], pid2i[b], y))
                    n_ok += 1
                else:
                    n_miss += 1
        print(f"  [{split} {'pos' if y else 'neg'}] {os.path.basename(p)}: "
              f"{n_ok} 对入库, miss={n_miss}, bad={n_bad}", flush=True)
    arr = np.asarray(rows, dtype=np.int64)
    assert len(arr) > 0, f"{split} 加载为空: {DATA_DIR}"
    return arr

# ---------- 3. 模型 ----------
class Projection(nn.Module):
    """1280 → 100 线性投影 + dropout (v1 同款, 冻结不动)"""
    def __init__(self, din, dout, p):
        super().__init__()
        self.fc, self.drop = nn.Linear(din, dout), nn.Dropout(p)
    def forward(self, x):                       # (B, L, din) -> (B, L, dout)
        return self.drop(self.fc(x))

class Tower(nn.Module):
    """6 层 conv1d: 100 → 64×5 → 25 (v1 同款, 冻结不动)"""
    def __init__(self, cin=PROJ_DIM, f=FILTERS, out=FINAL_C, k=3, layers=6):
        super().__init__()
        chans = [cin] + [f] * (layers - 1) + [out]
        self.convs = nn.ModuleList(
            [nn.Conv1d(chans[i], chans[i + 1], k, padding=k // 2) for i in range(layers)])
    def forward(self, z):                       # (B, L, C) -> (B, L, out)
        h = z.transpose(1, 2)
        for c in self.convs:
            h = F.relu(c(h))
        return h.transpose(1, 2)

class InteractionHead(nn.Module):
    """v2 头: mean-token concat + Hadamard 乘积 → MLP — 恢复跨蛋白乘性交互"""
    def __init__(self, c=FINAL_C, h=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * c, h), nn.ReLU(),
                                 nn.Dropout(DROPOUT), nn.Linear(h, 1))
    def forward(self, za, zb):                  # (B, L, C) × 2 → (B,)
        ha, hb = za.mean(1), zb.mean(1)         # (B, C)
        return self.mlp(torch.cat([ha, hb, ha * hb], dim=-1)).squeeze(-1)

class ContactMapHead(nn.Module):
    """v1 头 (仅 --v1-head 消融回放用): 接触图 + 全图均值 = 加性打分器"""
    def __init__(self, c=FINAL_C):
        super().__init__()
        self.proj = nn.Linear(2 * c, 1)
    def forward(self, za, zb):                  # (B, L, C) × 2 → (B,)
        B, L, C = za.shape
        pa = za.unsqueeze(2).expand(B, L, L, C)   # [b,i,j,:] = za[b,i,:]
        pb = zb.unsqueeze(1).expand(B, L, L, C)   # [b,i,j,:] = zb[b,j,:]
        cmap = self.proj(torch.cat([pa, pb], dim=-1)).squeeze(-1)   # (B, L, L)
        return cmap.mean(dim=(1, 2))

class DScriptStyle(nn.Module):
    def __init__(self, din=1280, use_v1_head=False):
        super().__init__()
        self.proj  = Projection(din, PROJ_DIM, DROPOUT)
        self.tower = Tower()
        self.head  = ContactMapHead() if use_v1_head else InteractionHead()
    def forward(self, xa, xb):                  # (B, L, din) × 2 → (B,) logit
        za, zb = self.tower(self.proj(xa)), self.tower(self.proj(xb))
        return self.head(za, zb)

# ---------- 4. 训练与评估 ----------
class PairData:
    def __init__(self, rows):
        self.ia = torch.tensor(rows[:, 0], device=DEV)
        self.ib = torch.tensor(rows[:, 1], device=DEV)
        self.y = torch.tensor(rows[:, 2], dtype=torch.float32, device=DEV)
        self.n_pos = int(rows[:, 2].sum())

def to_tokens(emb, idx):                        # (N,d) + LongTensor(B,) → (B, L, d)
    return emb[idx].unsqueeze(1).expand(-1, L_TOKENS, -1).contiguous()

@torch.no_grad()
def evaluate(model, emb, d, bs=4096):
    model.eval()
    scores = []
    for i in range(0, len(d.y), bs):
        sl = slice(i, i + bs)
        scores.append(model(to_tokens(emb, d.ia[sl]), to_tokens(emb, d.ib[sl])))
    p = torch.sigmoid(torch.cat(scores)).cpu().numpy()
    y = d.y.cpu().numpy().astype(int)
    pred = (p > 0.5).astype(int)
    return {"acc": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "aupr": average_precision_score(y, p)}

def train_one(tag, seed, data, epochs, use_v1_head):
    torch.manual_seed(seed)
    np.random.seed(seed)
    emb = EMB[tag]
    model = DScriptStyle(EMB[tag].shape[1], use_v1_head=use_v1_head).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tr, n = data["train"], len(data["train"].y)
    best_val, best_state, best_ep = -1.0, None, -1
    for ep in range(1, epochs + 1):
        model.train()
        perm, tot, t0 = torch.randperm(n, device=DEV), 0.0, time.time()
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(to_tokens(emb, tr.ia[b]), to_tokens(emb, tr.ib[b]))
            loss = F.binary_cross_entropy_with_logits(out, tr.y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        vm = evaluate(model, emb, data["val"])
        mark = ""
        if vm["acc"] > best_val:
            best_val, best_ep, mark = vm["acc"], ep, "  <-- best"
            best_state = copy.deepcopy(model.state_dict())
        print(f"  [{tag} s{seed}] ep{ep:02d}/{epochs} loss {tot / n:.4f} "
              f"val_acc {vm['acc']:.4f} val_aupr {vm['aupr']:.4f}{mark}  "
              f"[{time.time() - t0:.1f}s]", flush=True)
    model.load_state_dict(best_state)
    return evaluate(model, emb, data["test"]), best_val, best_ep

def summarize(rows, out_csv=OUT_CSV):
    print("\n" + "=" * 78)
    head_name = "v1 加性头(塌缩消融)" if rows and rows[0].get("head") == "v1" else "v2 交互头"
    print(f"交叉格终局: D-SCRIPT 风格架构({head_name}) x {{base, simclr}} 嵌入 (均值±标准差)")
    agg = {}
    for tag in ("base", "simclr"):
        rs = [r for r in rows if r["tag"] == tag]
        if not rs:
            continue
        agg[tag] = {m: float(np.mean([r[m] for r in rs])) for m in METRICS}
        line = f"  [{tag:6s}] "
        for m in METRICS:
            vals = [r[m] for r in rs]
            line += f"{m}={np.mean(vals):.4f}±{np.std(vals):.4f}  "
        print(line)
    if "base" in agg and "simclr" in agg:
        print("  Δ(增益)  " + "  ".join(f"{m}={agg['simclr'][m] - agg['base'][m]:+.4f}"
                                        for m in METRICS))
        print()
        print("  [参照 FC2 内部对照] acc +0.0171 / aupr +0.0226")
        print("  [判读] Δaupr ≥ +0.010 且多数种子同向 → 重塑增益跨架构成立(广播token化限定)")
    print(f"CSV → {out_csv}", flush=True)

# ---------- 5. 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只侦察数据加载, 不训练")
    ap.add_argument("--quick", action="store_true", help="只跑 base + 首种子")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS_DEF)))
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEF)
    ap.add_argument("--v1-head", action="store_true",
                    help="消融回放: v1 加性头, 应复现 ~0.537 (cudnn 非确定性, 非逐位)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    out_csv = os.path.join(
        SIMOUT, "cross_results_v1head.csv" if args.v1_head else "cross_results_v2.csv")

    pid2i = load_pool()
    print(f"[pool] {len(pid2i)} proteins, dim {EMB['base'].shape[1]}, dev={DEV}", flush=True)

    raw = {}
    for sp in ("train", "val", "test"):
        print(f"[{sp}] <- {SPLITS[sp]} (= train_fc2.py 同款分配)")
        raw[sp] = load_intra(SPLITS[sp], pid2i)
    data = {sp: PairData(raw[sp]) for sp in raw}
    for sp in ("train", "val", "test"):
        d = data[sp]
        print(f"[{sp}] {len(d.y)} 对 (pos {d.n_pos} / neg {len(d.y) - d.n_pos})")
    print("[对照] train ≈163k / val ≈59k / test ≈52k, miss 应≈0", flush=True)
    if args.probe:
        print("[probe] 完成, 未训练")
        return

    tags = ["base"] if args.quick else ["base", "simclr"]
    if args.quick:
        seeds = seeds[:1]
    head_tag = "v1" if args.v1_head else "v2"
    print(f"[mode] head={head_tag} | seeds={seeds} | epochs={args.epochs} | CSV → {out_csv}",
          flush=True)

    t0, rows_out = time.time(), []
    for tag in tags:
        for seed in seeds:
            print(f"\n=== [{tag} / seed {seed} / {head_tag} 头] 开始 ===", flush=True)
            te, bv, be = train_one(tag, seed, data, args.epochs, args.v1_head)
            row = {"tag": tag, "seed": seed, "head": head_tag,
                   **{m: te[m] for m in METRICS},
                   "best_val_acc": round(bv, 4), "best_ep": be}
            rows_out.append(row)
            print(f"=== [{tag}/{seed}/{head_tag}] test: "
                  + " ".join(f"{m}={te[m]:.4f}" for m in METRICS)
                  + f" (best val {bv:.4f} @ep{be}) | 累计 {(time.time() - t0) / 60:.1f} min",
                  flush=True)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
                w.writeheader()
                w.writerows(rows_out)
    summarize(rows_out, out_csv)

if __name__ == "__main__":
    main()
