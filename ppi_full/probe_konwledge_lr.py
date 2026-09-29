"""知识 LR 探针（E 行裁判）。P0-5(1536 维)、D5(洗牌同预算) 已落实。"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from data import FeatureStore, load_pairs
from config import Paths, save_json

def pair_feats(store, df):
    X = np.zeros((len(df), 4 * 128 + 4 * 256), np.float32)   # 1536
    for i, r in enumerate(df.itertuples()):
        ia, ib = store.idx(r.pid_a), store.idx(r.pid_b)
        ta = store.text[ia].astype(np.float32) if (store.text is not None and store.has_text[ia]) else np.zeros(128, np.float32)
        tb = store.text[ib].astype(np.float32) if (store.text is not None and store.has_text[ib]) else np.zeros(128, np.float32)
        ga = store.genome[ia].astype(np.float32) if (store.genome is not None and store.has_genome[ia]) else np.zeros(256, np.float32)
        gb = store.genome[ib].astype(np.float32) if (store.genome is not None and store.has_genome[ib]) else np.zeros(256, np.float32)
        X[i] = np.concatenate([ta, tb, np.abs(ta - tb), ta * tb,
                               ga, gb, np.abs(ga - gb), ga * gb])
    return X

def fit_sgd(Xtr, ytr, Xte, seed, epochs=6):
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=seed)
    rng = np.random.RandomState(seed)
    classes = np.unique(ytr)
    for _ in range(epochs):
        idx = rng.permutation(len(Xtr))
        clf.partial_fit(Xtr[idx], ytr[idx], classes=classes)
    return clf.predict_proba(Xte)[:, 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-tag", default="R7")
    a = ap.parse_args()
    paths = Paths()
    store = FeatureStore(paths.feat_dir)
    tr = load_pairs(os.path.join(paths.data_dir, "train.csv"))
    te = load_pairs(os.path.join(paths.data_dir, "test.csv"))
    ytr, yte = tr.label.values, te.label.values
    Xtr, Xte = pair_feats(store, tr), pair_feats(store, te)

    p_real = fit_sgd(Xtr, ytr, Xte, a.seed, epochs=6)
    auprc = average_precision_score(yte, p_real)

    pid_pool = np.array(sorted(set(tr.pid_a) | set(tr.pid_b)))
    perm = np.random.RandomState(a.seed).permutation(len(pid_pool))
    remap = dict(zip(pid_pool, pid_pool[perm]))
    tr_s = tr.copy()
    tr_s["pid_a"] = tr_s.pid_a.map(remap)
    tr_s["pid_b"] = tr_s.pid_b.map(remap)
    p_shuf = fit_sgd(pair_feats(store, tr_s), ytr, Xte, a.seed + 1, epochs=6)
    shuf = average_precision_score(yte, p_shuf)

    res = dict(knowledge_lr_auprc=float(auprc), shuffled_auprc=float(shuf),
               note="E-judge: 若 knowledge_lr ≥ FULL − 0.02 → 结论降格")
    try:
        full = json.load(open(os.path.join(paths.out_dir, a.full_tag, f"seed{a.seed}", "metrics.json")))
        res["full_test_auprc"] = full["test"]["auprc"]
        res["full_minus_probe"] = full["test"]["auprc"] - float(auprc)
        res["verdict"] = "DEGRADED(知识特征主导)" if full["test"]["auprc"] - auprc < 0.02 \
            else "OK(模型在知识特征之上有增量)"
    except FileNotFoundError:
        res["verdict"] = "FULL 未运行，仅记录探针值"
    print(json.dumps(res, indent=2, ensure_ascii=False))
    save_json(res, os.path.join(paths.out_dir, "probe_knowledge_lr.json"))

if __name__ == "__main__":
    main()
