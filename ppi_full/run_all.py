"""总编排（v2.3）：ladder / ablation / search / baseline / stats / mc。
本轮修复：N2（run_tagged 返回各 seed best_val 列表，search 排序可用）。
补丁携带：P10(非平凡性检查)、D4(ckpt 前置检查)。"""
import argparse, json, os, random, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (R_LEVELS, ABLATIONS, build_model_cfg, build_train_cfg,
                    TrainCfg, Paths, save_json, load_json)
from data import FeatureStore, load_pairs
from engine import train_one_run, set_seed, mc_dropout_eval
import stats as st

def feature_nontrivial(store, name, k=256):
    arr = getattr(store, name, None)
    if arr is None:
        return False
    if name in ("text", "genome"):
        v = np.asarray(arr[:min(store.n, k)], np.float32)
    else:
        v = np.asarray(arr[:min(store.n, k), :64], np.float32)
    return bool(np.abs(v).max() > 1e-6)                   # P10：全零即视为缺失

def run_tagged(tag, level, overrides, seeds, tbase):
    paths = Paths()
    mcfg = build_model_cfg(level, overrides)
    tcfg = build_train_cfg(overrides, tbase)
    store = FeatureStore(paths.feat_dir)
    for name in ("struct", "iface", "surf", "text", "genome"):
        if getattr(mcfg.sw, name) == 1 and not feature_nontrivial(store, name):
            raise RuntimeError(f"{tag}: switch_{name}=1 但特征 '{name}' 缺失或全零 "
                               f"(zero-provider 陷阱 / 未 collect)")
    if mcfg.use_ssl and not os.path.exists(paths.adapter_ckpt):
        raise RuntimeError(f"{tag}: use_ssl=True 但 adapter ckpt 不存在: {paths.adapter_ckpt} "
                           f"—— 先运行 ssl_pretrain.py")
    vals = []
    for seed in seeds:
        vals.append(train_one_run(level, seed, mcfg, tcfg, paths, tag=tag,
                                  adapter_ckpt=paths.adapter_ckpt if mcfg.use_ssl else None))
    return vals                                           # N2

def cmd_ladder(a):
    tbase = TrainCfg(lr=a.lr, bs=a.bs)
    overrides = load_json(a.hyperparams)["best_overrides"] if a.hyperparams else None
    for lv in a.levels:
        run_tagged(lv, lv, overrides, a.seeds, tbase)

def cmd_ablation(a):
    tbase = TrainCfg(lr=a.lr, bs=a.bs)
    overrides = load_json(a.hyperparams)["best_overrides"] if a.hyperparams else None
    for name in a.names:
        merged = dict(ABLATIONS[name])
        if overrides:
            merged.update({k: v for k, v in overrides.items() if k not in merged})
        run_tagged(name, "R7", merged, a.seeds, tbase)

def cmd_search(a):
    """R7 上 40 组随机搜索；其余级复用最优超参（best_hyperparams.json）。8GB：--trials 10–15。"""
    space = dict(lr=[5e-5, 1e-4, 3e-4], dropout=[0.1, 0.2, 0.3],
                 label_smooth=[0.0, 0.05, 0.1], mixup_prob=[0.0, 0.25, 0.5],
                 beta_init=[0.0, 0.5, 1.0], iface_tau=[0.25, 0.5, 0.7])
    rng = random.Random(0)
    paths = Paths()
    results = []
    for t in range(a.trials):
        lr = rng.choice(space["lr"]); do = rng.choice(space["dropout"])
        ls = rng.choice(space["label_smooth"]); mp = rng.choice(space["mixup_prob"])
        bi = rng.choice(space["beta_init"]); tau = rng.choice(space["iface_tau"])
        tag = f"SEARCH{t:02d}"
        try:
            vals = run_tagged(tag, "R7",
                              dict(dropout=do, beta_init=bi, iface_tau=tau,
                                   train__lr=lr, train__label_smooth=ls, train__mixup_prob=mp),
                              [42], TrainCfg())
            results.append(dict(tag=tag, val_auprc=vals[0], lr=lr, dropout=do, label_smooth=ls,
                                mixup_prob=mp, beta_init=bi, iface_tau=tau))
        except Exception as e:
            print(f"{tag} failed: {e}")
    results.sort(key=lambda r: -r["val_auprc"])
    best = results[0]
    best_overrides = dict(dropout=best["dropout"], beta_init=best["beta_init"],
                          iface_tau=best["iface_tau"], train__lr=best["lr"],
                          train__label_smooth=best["label_smooth"],
                          train__mixup_prob=best["mixup_prob"])
    save_json(dict(best=best, best_overrides=best_overrides, all=results),
              os.path.join(paths.out_dir, "best_hyperparams.json"))
    print("best:", best)

def cmd_baseline(a):
    """模块6（自包含部分）：RFC-mean / RFC-40。外部模型重跑为接口化 stub。"""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    paths = Paths()
    store = FeatureStore(paths.feat_dir)
    set_seed(42)
    tr, te = load_pairs(os.path.join(paths.data_dir, "train.csv")), load_pairs(os.path.join(paths.data_dir, "test.csv"))

    def mean_emb(df):
        X = np.zeros((len(df), 1280), np.float32)
        Ls = np.asarray(store.lengths)
        ia = np.array([store.pid2idx[p] for p in df.pid_a])
        ib = np.array([store.pid2idx[p] for p in df.pid_b])
        for i in range(len(df)):
            la, lb = int(Ls[ia[i]]), int(Ls[ib[i]])
            X[i] = 0.5 * (np.asarray(store.seq[ia[i], :la], np.float32).mean(0)
                          + np.asarray(store.seq[ib[i], :lb], np.float32).mean(0))
        return X

    Xtr, Xte = mean_emb(tr), mean_emb(te)
    ytr, yte = tr.label.values.astype(int), te.label.values.astype(int)
    pca40 = PCA(40, random_state=42).fit(Xtr)
    jobs = (("RFC-mean", Xtr, Xte), ("RFC-40", pca40.transform(Xtr), pca40.transform(Xte)))
    for name, Xt, Xe in jobs:
        out = os.path.join(paths.out_dir, name, "seed42")
        os.makedirs(out, exist_ok=True)
        rf = RandomForestClassifier(100, n_jobs=-1, random_state=42).fit(Xt, ytr)
        p = rf.predict_proba(Xe)[:, 1]
        save_json(dict(tag=name, test_auprc=float(average_precision_score(yte, p))),
                  os.path.join(out, "metrics.json"))
    print("RFC baselines done. 外部基线重跑(TUnA/TUnA-text-fused/D-SCRIPT-ESM-2/ProteinCLIP): "
          "clone 官方 repo → 以 data.py 的 csv 与 test 顺序为接口重跑 → 写入 "
          "out_dir/<name>/seed*/{metrics.json,predictions.npz} (probs 1-D, labels) → stats 直接汇总。")

def cmd_stats(a):
    rows = st.summarize(Paths().out_dir, a.tags, a.baseline,
                        out_csv=os.path.join(Paths().out_dir, "summary.csv"))
    for r in rows:
        print(r)

def cmd_mc(a):
    paths = Paths()
    for tag in a.tags:
        for seed in a.seeds:
            mc_dropout_eval(paths, tag, seed, T=a.T)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ladder")
    p.add_argument("--levels", nargs="+", default=list(R_LEVELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 2024])
    p.add_argument("--lr", type=float, default=5e-5)       # 8GB 默认；24GB+ 用 1e-4
    p.add_argument("--bs", type=int, default=16)           # 8GB 默认；24GB+ 用 64
    p.add_argument("--hyperparams", default=None)
    p = sub.add_parser("ablation")
    p.add_argument("--names", nargs="+", default=list(ABLATIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 2024])
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--hyperparams", default=None)
    p = sub.add_parser("search")
    p.add_argument("--trials", type=int, default=40)
    p = sub.add_parser("baseline")
    p = sub.add_parser("stats")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--baseline", default=None)
    p = sub.add_parser("mc")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 2024])
    p.add_argument("--T", type=int, default=10)
    a = ap.parse_args()
    dict(ladder=cmd_ladder, ablation=cmd_ablation, search=cmd_search,
         baseline=cmd_baseline, stats=cmd_stats, mc=cmd_mc)[a.cmd](a)

if __name__ == "__main__":
    main()
