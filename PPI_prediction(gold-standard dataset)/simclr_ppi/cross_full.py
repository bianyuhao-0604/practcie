# cross_full.py — SimCLR plug-in × 全基线 × 双臂 × 多种子统一 runner(按实际目录布局)
import os, sys, time, random, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score)
from scipy import stats

# ---- 路径(基于 __file__ 定位,与 cd 无关)----
HERE   = Path(__file__).resolve().parent   # ...\simclr_ppi
BASE   = HERE.parent                       # ...\PPI_prediction(gold-standard dataset)
DATA   = BASE / "dataset"
SIMOUT = BASE / "simclr_out"
sys.path.insert(0, str(BASE))              # 让 models.* / data.* 可导入(attention.py 内部依赖)

from models.fc2_20_2_dense import FC2_20_2Dense
from attention_compat import AttB, RichouxPP, build_tuna, fix_scale

EPS = 1280
DEFAULT_SEEDS = list(range(42, 62))
# block 即 train/val/test 本体;可由 --blocks 覆盖。以 cross_dscript.py 实际映射为准!
DEFAULT_BLOCKS = {"train": "Intra1", "val": "Intra0", "test": "Intra2"}

# ---- 数据:pos/neg txt 成对文件 ----
def read_pair_txt(path):
    """读 Intra*_pos_rr.txt / Intra*_neg_rr.txt:每行 'ID1<ws>ID2'(tab 或空格),
    兼容可能的首行表头。返回 [(id1, id2), ...]"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:                 # 空行/表头/异常行跳过
                continue
            out.append((parts[0], parts[1]))
    return out

def load_pairs(role, blocks):
    blk = blocks[role]
    pos = read_pair_txt(DATA / f"{blk}_pos_rr.txt")
    neg = read_pair_txt(DATA / f"{blk}_neg_rr.txt")
    pairs = [(a, b, 1.0) for a, b in pos] + [(a, b, 0.0) for a, b in neg]
    print(f"[data] {role} <- {blk}: pos={len(pos)} neg={len(neg)} total={len(pairs)}")
    return pairs

def load_emb(path):
    """实测结构: ids=<U10 (N,), X=(N,1280), 另有 mu/sd/W 元数据键(忽略)。
    定位规则: uids=字符串键; 矩阵=行数==len(uids) 的唯一 2-D 数值键。"""
    z = np.load(path, allow_pickle=False)
    EPS_DIM = 1280

    str_keys = [k for k in z.files if z[k].dtype.kind in "US"]
    mat_keys = [k for k in z.files if z[k].dtype.kind in "fiu" and z[k].ndim == 2]
    assert str_keys and mat_keys, \
        f"非 ids+matrix 结构; 键: {[(k, str(z[k].dtype), z[k].shape) for k in z.files]}"

    uids = np.asarray(z[str_keys[0]]).astype(str)
    assert len(set(uids)) == len(uids), f"ids 有重复 {len(uids)-len(set(uids))} 个"
    cands = [k for k in mat_keys if z[k].shape[0] == len(uids)]
    assert len(cands) == 1, \
        f"无法唯一定位嵌入矩阵: 候选 {[(k, z[k].shape) for k in mat_keys]}"
    mat = z[cands[0]].astype(np.float32)
    assert mat.shape[1] == EPS_DIM, f"维度 {mat.shape[1]} != {EPS_DIM}"

    ignored = [k for k in z.files if k not in (cands[0], str_keys[0])]
    print(f"[emb ] {path.name}: N={len(uids)}, dim={mat.shape[1]}, "
          f"ids键='{str_keys[0]}', X键='{cands[0]}', 忽略={ignored}")
    return {u: mat[i] for i, u in enumerate(uids)}


class Stack:
    def __init__(self, pairs, emb, tag=""):
        miss = ({p[0] for p in pairs} | {p[1] for p in pairs}) - set(emb)
        assert not miss, f"[{tag}] {len(miss)} 个 uid 缺嵌入!检查 txt 的 ID 命名空间与 npz 键。示例: {sorted(miss)[:5]}"
        self.i1 = np.stack([emb[p[0]] for p in pairs])
        self.i2 = np.stack([emb[p[1]] for p in pairs])
        self.y  = np.array([p[2] for p in pairs], dtype=np.float32)
    def __len__(self): return len(self.y)
    def batch(self, idx, dev):
        return (torch.from_numpy(self.i1[idx]).to(dev),
                torch.from_numpy(self.i2[idx]).to(dev),
                torch.from_numpy(self.y[idx]).to(dev))

# ---- models(与上一版相同,含 B1/B6 修复)----
class DscriptB(nn.Module):
    """自检版(省略 map_predict 头)。dscript 格正式数字以 cross_dscript.py 20 种子为准。"""
    def __init__(self, D=EPS, d=100, h=50, w=7):
        super().__init__()
        self.proj = nn.Linear(D, d); self.drop = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(2 * d, h, 1); self.bn1 = nn.BatchNorm2d(h)
        self.conv = nn.Conv2d(h, 1, w, padding=w // 2); self.bn2 = nn.BatchNorm2d(1)
    def forward(self, v1, v2):                              # (B,D)
        x1 = F.relu(self.drop(self.proj(v1))); x2 = F.relu(self.drop(self.proj(v2)))
        m = torch.cat([(x1 - x2).abs().unsqueeze(1),
                       (x1 * x2).unsqueeze(1)], -1).permute(0, 3, 1, 2)
        c = F.relu(self.bn2(self.conv(F.relu(self.bn1(self.conv2(m))))))
        return torch.sigmoid(c.mean(dim=(1, 2, 3)))
class Baseline2dB(nn.Module):
    # B7: 外积图 1 通道(conv in=1,修崩溃)
    # B9: 输入 LN(与其他头内部归一化对等)
    # B10: L=1 下 amax 读出退化为 f((maxa)(maxb),(mina)(minb)) —— 假设类塌缩,
    #      常数 0.5 即类内最优;改全双线性读出 a^T M b(= 外积图+全局核conv+sum池化)
    def __init__(self, D=EPS, h3=64):
        super().__init__()
        h, h2 = D // 4, D // 16
        self.ln = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, h); self.fc2 = nn.Linear(h, h2); self.fc3 = nn.Linear(h2, h3)
        self.readout = nn.Linear(h3 * h3, 1)                  # B10
        self.sig = nn.Sigmoid()
    def forward(self, v1, v2):                              # (B,1,D) via TokenWrap
        v1, v2 = self.ln(v1), self.ln(v2)
        a = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(v1)))))).squeeze(1)   # (B,h3)
        b = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(v2)))))).squeeze(1)   # (B,h3)
        mat = torch.einsum('bi,bj->bij', a, b)              # (B,h3,h3) 外积图保留
        return self.sig(self.readout(mat.flatten(1))).view(-1)

class TUnAW(nn.Module):
    def __init__(self): super().__init__(); self.core = build_tuna()
    def forward(self, v1, v2):                              # (B,1,D)×2
        return self.core(torch.stack([v1, v2], dim=1)).view(-1)   # (B,2,1,D)

class FC2W(nn.Module):
    def __init__(self): super().__init__(); self.core = FC2_20_2Dense(embed_dim=EPS)
    def forward(self, v1, v2):                              # (B,D)
        return self.core(torch.stack([v1, v2], dim=1)).view(-1)   # (B,2,D)

class TokenWrap(nn.Module):
    def __init__(self, core): super().__init__(); self.core = core
    def forward(self, v1, v2): return self.core(v1.unsqueeze(1), v2.unsqueeze(1))

MODEL_REGISTRY = {
    "fc2":        (lambda: FC2W(),               "pool"),
    "dscript":    (lambda: DscriptB(),           "pool"),
    "baseline2d": (lambda: Baseline2dB(),        "token"),
    "selfatt":    (lambda: AttB(cross=False),    "token"),
    "crossatt":   (lambda: AttB(cross=True),     "token"),
    "richoux":    (lambda: RichouxPP(),          "token"),
    "tuna":       (lambda: TUnAW(),              "token"),
}

# ---- train / eval(含 B3/B4/B8 修复)----
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def predict(model, st, idx, dev, bs=1024):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        v1, v2, _ = st.batch(idx[i:i + bs], dev)
        out.append(model(v1, v2).float().cpu().numpy())
    return np.concatenate(out)

def train_deep(model, tr, va, te, seed, dev, epochs=25, bs=256, lr=1e-3, patience=5):
    model.to(dev)
    fix_scale(model, dev)                       # B8: 必须在 .to(dev) 之后
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCELoss(); rng = np.random.default_rng(seed)
    n = len(tr); best_acc, best_state, best_ep, bad = -1, None, -1, 0
    for ep in range(1, epochs + 1):
        model.train(); perm = rng.permutation(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            if len(j) == 1: continue            # B4
            v1, v2, y = tr.batch(j, dev)
            loss = lossf(model(v1, v2).clamp(1e-6, 1 - 1e-6), y)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = accuracy_score(va.y, (predict(model, va, np.arange(len(va)), dev) > .5).astype(int))
        if acc > best_acc:
            best_acc, best_ep, bad = acc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    p = predict(model, te, np.arange(len(te)), dev); y = te.y.astype(int)
    return dict(acc=accuracy_score(y, p > .5),
                precision=precision_score(y, p > .5, zero_division=0),
                recall=recall_score(y, p > .5, zero_division=0),
                f1=f1_score(y, p > .5, zero_division=0),
                aupr=average_precision_score(y, p), best_epoch=best_ep)

def run_rfc(tr, te, seed, n_pca=40, n_trees=300):
    Xtr = np.concatenate([tr.i1, tr.i2], 1); Xte = np.concatenate([te.i1, te.i2], 1)
    if n_pca:
        ip = IncrementalPCA(n_components=n_pca, batch_size=8192).fit(Xtr)
        Xtr, Xte = ip.transform(Xtr), ip.transform(Xte)
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1).fit(Xtr, tr.y)
    p = rf.predict_proba(Xte)[:, 1]; y = te.y.astype(int)
    return dict(acc=accuracy_score(y, p > .5),
                precision=precision_score(y, p > .5, zero_division=0),
                recall=recall_score(y, p > .5, zero_division=0),
                f1=f1_score(y, p > .5, zero_division=0),
                aupr=average_precision_score(y, p), best_epoch=-1)

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["fc2", "baseline2d", "rfc", "selfatt", "crossatt", "richoux", "tuna"])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--arms", nargs="+", default=["prep", "simclr"])
    ap.add_argument("--emb_pattern", default="embeddings_{arm}.npz")
    ap.add_argument("--blocks", default="Intra1,Intra0,Intra2",
                    help="train,val,test 对应的 block;与 cross_dscript.py 保持一致!")
    ap.add_argument("--out", default=str(SIMOUT / "cross_results_full.csv"))
    args = ap.parse_args()

    b = args.blocks.split(",")
    blocks = {"train": b[0], "val": b[1], "test": b[2]}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = {a: load_emb(SIMOUT / args.emb_pattern.format(arm=a)) for a in args.arms}
    pairs = {r: load_pairs(r, blocks) for r in ("train", "val", "test")}

    rows = []
    for arm in args.arms:
        st = {r: Stack(pairs[r], emb[arm], tag=arm) for r in ("train", "val", "test")}
        for name in args.models:
            for seed in args.seeds:
                t0 = time.time(); set_seed(seed)            # B3
                if name == "rfc":
                    m = run_rfc(st["train"], st["test"], seed)
                else:
                    build, kind = MODEL_REGISTRY[name]
                    model = build()
                    if kind == "token": model = TokenWrap(model)  # B1
                    m = train_deep(model, st["train"], st["val"], st["test"], seed, dev)
                rows.append(dict(arm=arm, model=name, seed=seed, **m))
                print(f"[{arm:>6}/{name:>10}/s{seed}] acc={m['acc']:.4f} aupr={m['aupr']:.4f} "
                      f"ep={m['best_epoch']} ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print("\n===== 配对汇总 (AUPR) =====")
    for name, g in df.groupby("model"):
        pv = g.pivot(index="seed", columns="arm", values="aupr").dropna()
        if {"prep", "simclr"} <= set(pv) and len(pv) > 1:
            d = pv["simclr"] - pv["prep"]
            t, tp = stats.ttest_rel(pv["simclr"], pv["prep"])
            try: _, wp = stats.wilcoxon(pv["simclr"], pv["prep"])
            except ValueError: wp = float("nan")
            print(f"{name:>12}: Δ={d.mean():+.4f}±{d.std():.4f} | t={t:.2f} p={tp:.4f} "
                  f"| W p={wp:.4f} | 同向 {(d>0).sum()}/{len(d)}")

if __name__ == "__main__":
    main()
