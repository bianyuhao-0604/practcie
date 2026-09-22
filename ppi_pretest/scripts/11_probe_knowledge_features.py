# scripts/11_probe_knowledge_features.py — 知识特征泄露探针
# 问题: text / genome 单独能"知道"多少标签? 若 text-only ≈ R7 全模型 → 警报 (可能直接编码答案)
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import DATA, FEATURES
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from sklearn.metrics import roc_auc_score as _ras, average_precision_score as _aps
    def AUROC(y, p): return float(_ras(y, p))
    def AUPRC(y, p): return float(_aps(y, p))
    MSRC = "sklearn"
except Exception:
    def AUROC(y, p):
        r = pd.Series(p).rank().to_numpy(); y = np.asarray(y, float)
        n1, n0 = y.sum(), len(y) - y.sum()
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    def AUPRC(y, p):
        o = np.argsort(-np.asarray(p)); yy = np.asarray(y, float)[o]
        tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
        return float(np.trapz(tp / k, tp / yy.sum()))
    MSRC = "manual(近似)"


def load_pairs(csv, id2row):
    df = pd.read_csv(csv)
    ok = df.pidA.isin(id2row) & df.pidB.isin(id2row)
    df = df[ok]
    return (df.pidA.map(id2row).to_numpy(np.int64),
            df.pidB.map(id2row).to_numpy(np.int64),
            df.label.astype(float).to_numpy())


def pair_feat(X, ia, ib):
    a, b = X[ia], X[ib]
    return np.hstack([np.abs(a - b), a * b]).astype(np.float64)


def fit_lr(Ftr, ytr, iters=2000, lr=0.1):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    Xs = np.hstack([(Ftr - mu) / sd, np.ones((len(Ftr), 1))])
    w = np.zeros(Xs.shape[1]); m = np.zeros_like(w); v = np.zeros_like(w)
    y = ytr * 2 - 1
    for t in range(1, iters + 1):
        z = np.clip(Xs @ w, -30, 30)
        g = Xs.T @ (-y / (1 + np.exp(y * z))) / len(y) + 1e-3 * w
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return mu, sd, w


def apply_lr(mu, sd, w, F):
    Fs = np.hstack([(F - mu) / (sd + 1e-9), np.ones((len(F), 1))])
    return 1 / (1 + np.exp(-np.clip(Fs @ w, -30, 30)))


def main():
    ids = (DATA / "proteins.txt").read_text().split()
    id2row = {p: i for i, p in enumerate(ids)}
    Xt = np.stack([np.load(FEATURES / "text" / f"{p}.npy") for p in ids]).astype(np.float64)
    Xg = np.stack([np.load(FEATURES / "genome" / f"{p}.npy") for p in ids]).astype(np.float64)
    ia, ib, y = load_pairs(DATA / "pairs_tr.csv", id2row)
    ja, jb, yt = load_pairs(DATA / "pairs_te.csv", id2row)
    print(f"蛋白 {len(ids)} | text {Xt.shape} | genome {Xg.shape} | "
          f"train {len(y)} (正例率 {y.mean():.3f}) | test {len(yt)} (正例率 {yt.mean():.3f}) | 指标: {MSRC}")

    def run(name, X):
        mu, sd, w = fit_lr(pair_feat(X, ia, ib), y)
        p = apply_lr(mu, sd, w, pair_feat(X, ja, jb))
        print(f"  {name:22s} AUPRC={AUPRC(yt, p):.4f}  AUROC={AUROC(yt, p):.4f}")

    print("—— 各知识特征单独的判别力 (逻辑回归, 对称对特征) ——")
    run("text only", Xt)
    run("genome only", Xg)
    run("text + genome", np.hstack([Xt, Xg]))
    print("—— 洗牌对照 (特征随机换到别的蛋白, 应≈0.5) ——")
    rr = np.random.default_rng(0)
    run("text shuffled", Xt[rr.permutation(len(ids))])
    run("genome shuffled", Xg[rr.permutation(len(ids))])
    print("""
判读 (参照: R7 全模型 test_auprc≈0.637, no_global≈0.606, R1≈0.581):
  text-only >= 0.63  → 文本信息量≈全模型, 警报: 可能直接编码标签 → 原文 grep 审计 + hard-negative 复测
  0.55 ~ 0.62        → 有信号但非答案, 像功能先验 (T3 嫌疑, 由 hard-negative 裁决)
  ≈ 0.50             → text 无增量; 若 R7 增益仍在, 查 genome-only
  shuffled 明显≠0.5  → 管线有 bug, 本探针结果不可信""")


if __name__ == "__main__":
    main()
