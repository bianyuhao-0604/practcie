"""统计检验：Wilcoxon(exact) + Cohen's dz + 精确 McNemar + Stouffer + bootstrap CI。
probs 依赖 engine 的 1-D 输出（N1）。"""
import glob, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

def load_runs(out_dir, tag):
    ms, preds = [], []
    for f in sorted(glob.glob(os.path.join(out_dir, tag, "seed*", "metrics.json"))):
        ms.append(__import__("json").load(open(f)))
        preds.append(np.load(os.path.join(os.path.dirname(f), "predictions.npz"), allow_pickle=True))
    assert ms, f"no runs for {tag}"
    for i in range(1, len(preds)):
        assert np.array_equal(preds[0]["labels"], preds[i]["labels"]), \
            f"{tag}: test label order differs across seeds"
    return ms, preds

def bootstrap_auprc_ci(y, p, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    n = len(y)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, y[i].size):
            continue
        vals.append(average_precision_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def mcnemar_exact(p1, p2, labels):
    c1 = (np.asarray(p1) >= 0.5) == (np.asarray(labels) == 1)
    c2 = (np.asarray(p2) >= 0.5) == (np.asarray(labels) == 1)
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    if b + c == 0:
        return 1.0, b, c
    return float(stats.binomtest(min(b, c), b + c, 0.5).pvalue), b, c

def stouffer(ps):
    ps = np.clip(np.asarray(ps, float), 1e-12, 1 - 1e-12)
    z = stats.norm.isf(ps)
    Z = float(np.sum(z) / np.sqrt(len(z)))
    return float(stats.norm.sf(Z))

def compare(out_dir, tag_a, tag_b):
    """返回 a 相对 b 的增量（delta>0 ⇒ a 更好）。"""
    ma, pa = load_runs(out_dir, tag_a)
    mb, pb = load_runs(out_dir, tag_b)
    n = min(len(ma), len(mb))
    a = np.array([m["test"]["auprc"] for m in ma])[:n]
    b = np.array([m["test"]["auprc"] for m in mb])[:n]
    w_p = float(stats.wilcoxon(a, b, method="exact").pvalue) if n >= 2 else float("nan")
    d = a - b
    dz = float(d.mean() / d.std(ddof=1)) if n > 1 and d.std(ddof=1) > 0 else float("nan")
    mps = [mcnemar_exact(pa[i]["probs"], pb[i]["probs"], pa[i]["labels"])[0] for i in range(n)]
    lo, hi = bootstrap_auprc_ci(pa[0]["labels"], pa[0]["probs"])
    return dict(delta=float(a.mean() - b.mean()),
                auprc_a=f"{a.mean():.4f}±{a.std():.4f}",
                auprc_b=f"{b.mean():.4f}±{b.std():.4f}",
                wilcoxon_p=w_p, cohens_dz=dz,
                mcnemar_per_seed=[round(x, 4) for x in mps],
                mcnemar_stouffer_p=stouffer(mps) if n else float("nan"),
                boot95_a=(lo, hi))

def summarize(out_dir, tags, baseline=None, out_csv=None):
    rows = []
    if baseline:
        for t in tags:
            if t == baseline:
                continue
            try:
                rows.append(dict(comparison=f"{t} vs {baseline}", **compare(out_dir, t, baseline)))
            except AssertionError as e:
                print(e)
    else:
        for a, b in zip(tags[:-1], tags[1:]):
            try:
                rows.append(dict(comparison=f"{b} vs {a}", **compare(out_dir, b, a)))
            except AssertionError as e:
                print(e)
    for t in tags:
        ms, pr = load_runs(out_dir, t)
        lo, hi = bootstrap_auprc_ci(pr[0]["labels"], pr[0]["probs"])
        rows.append(dict(tag=t,
                         auprc=f"{np.mean([m['test']['auprc'] for m in ms]):.4f}"
                               f"±{np.std([m['test']['auprc'] for m in ms]):.4f}",
                         boot95=(lo, hi),
                         acc=f"{np.mean([m['test']['acc'] for m in ms]):.4f}"))
    if out_csv:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows
