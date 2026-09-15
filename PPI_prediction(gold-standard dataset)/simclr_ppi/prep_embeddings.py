"""Stage 0: pkl → 统一预处理嵌入库。
修复 (1,1280)→(1280)；蛋白级缩放统计仅来自 Intra1 唯一蛋白（已验证与原作者
管线指纹 r=0.9999 对齐）；可选 ZCA 白化（统计同样仅来自 Intra1）。
产出 PREP_NPZ: { ids, X, mu, sd, W }"""
import pickle, numpy as np, pandas as pd
from config import PKL, SPLIT_DIR, PREP_NPZ, WHITEN, WHITEN_EPS

def read_pairs(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["a", "b"])
    df["a"], df["b"] = df["a"].astype(str).str.strip(), df["b"].astype(str).str.strip()
    return df.dropna()

def main():
    print("加载 pkl ...")
    with open(PKL, "rb") as f:
        emb = pickle.load(f)
    ids = list(emb.keys())
    X = np.stack([np.asarray(emb[k], np.float32).reshape(-1) for k in ids])
    N, D = X.shape
    print(f"蛋白数 {N}, 维度 {D}"); assert D == 1280

    bad = np.isnan(X).any(1) | np.isinf(X).any(1) | (np.abs(X).max(1) < 1e-6)
    print(f"NaN/Inf/全零: {int(np.isnan(X).any(1).sum())}/{int(np.isinf(X).any(1).sum())}/{int((np.abs(X).max(1)<1e-6).sum())}")
    if bad.any():
        ids, X = [i for i, b in zip(ids, bad) if not b], X[~bad]
        print(f"剔除异常 {int(bad.sum())} 条")

    id2row = {k: i for i, k in enumerate(ids)}
    tr = pd.concat([read_pairs(SPLIT_DIR / "Intra1_pos_rr.txt"),
                    read_pairs(SPLIT_DIR / "Intra1_neg_rr.txt")], ignore_index=True)
    tr_ids = pd.unique(pd.concat([tr["a"], tr["b"]]))
    miss = [p for p in tr_ids if p not in id2row]
    assert not miss, f"Intra1 缺嵌入蛋白 {len(miss)} 条, 例: {miss[:5]}"
    print(f"Intra1 唯一蛋白 {len(tr_ids)}, 覆盖率 100%")

    rows = np.array([id2row[p] for p in tr_ids])
    Xtr  = X[rows].astype(np.float64)
    mu, sd = Xtr.mean(0), np.clip(Xtr.std(0), 1e-6, None)
    Xs = ((X - mu) / sd).astype(np.float32)
    print(f"缩放后 train 蛋白 max|μ| = {np.abs(Xs[rows]).mean(0).max():.2e} (应~1e-6)")

    W = np.eye(D, dtype=np.float32)
    if WHITEN:
        Xc = Xs[rows].astype(np.float64)
        ev, evec = np.linalg.eigh(Xc.T @ Xc / (len(Xc) - 1))
        W = (evec @ np.diag(1 / np.sqrt(ev + WHITEN_EPS)) @ evec.T).astype(np.float32)
        Xs = (Xs @ W).astype(np.float32)
        print("已做 ZCA 白化")

    np.savez(PREP_NPZ, ids=np.array(ids), X=Xs, mu=mu.astype(np.float32),
             sd=sd.astype(np.float32), W=W)
    print(f"已保存 {PREP_NPZ}")

if __name__ == "__main__":
    main()
