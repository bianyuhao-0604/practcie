import os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, ".")
from config import Paths, save_json

FEAT = "workdir/features"
ANNOT = ["length","n_pfam","n_interpro","n_pdb","go_f","go_c","n_kw","n_tm","n_signal","n_secondary","reviewed"]
STRUCT = ["n_res","mean_plddt","frac_lt50","frac_lt70","frac_lt90","has_struct"]
LOG1P = {"length","n_pfam","n_interpro","n_pdb","go_f","go_c","n_kw","n_tm","n_signal","n_secondary","n_res"}

node = pd.read_csv(f"{FEAT}/features_node.tsv", sep="\t", na_values=["NA"]).set_index("primary")
for c in ANNOT + STRUCT:
    node[c] = pd.to_numeric(node[c], errors="coerce").fillna(0.0)
for c in LOG1P:
    node[c] = np.log1p(node[c].clip(lower=0))

pid2idx = {p: int(i) for p, i in json.load(open(f"{FEAT}/meta.json"))["pid2idx"].items()}
M = {"annot": node[ANNOT].to_numpy(np.float64),
     "struct": node[STRUCT].to_numpy(np.float64)}
M["annot+struct"] = np.hstack([M["annot"], M["struct"]])

def load(split):
    df = pd.read_csv(f"workdir/data/{split}.csv", dtype={"pid_a": str, "pid_b": str})
    ia = df.pid_a.map(pid2idx).to_numpy()
    ib = df.pid_b.map(pid2idx).to_numpy()
    assert not (np.isnan(ia).any() or np.isnan(ib).any()), f"{split} pid outside store"
    return ia, ib, df.label.to_numpy(np.float64)

D = {s: load(s) for s in ["train", "val", "test"]}

def pairize(split, X):
    ia, ib, _ = D[split]
    Xa, Xb = X[ia], X[ib]
    return np.hstack([np.abs(Xa - Xb), Xa * Xb])   # order-invariant pair transform

def run(X):
    Xtr, ytr = pairize("train", X), D["train"][2]
    Xva, yva = pairize("val", X),   D["val"][2]
    Xte, yte = pairize("test", X),  D["test"][2]
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    pv = lr.predict_proba(sc.transform(Xva))[:, 1]
    pt = lr.predict_proba(sc.transform(Xte))[:, 1]
    return average_precision_score(yva, pv), average_precision_score(yte, pt), roc_auc_score(yte, pt)

rows = {}
for name in ["annot", "struct", "annot+struct"]:
    va, te, auc = run(M[name])
    rows[name] = (va, te, auc)
    print(f"PROBE block={name:13s} AUPRC val={va:.4f} test={te:.4f} AUROC test={auc:.4f}")

res = dict(
    knowledge_lr_auprc=float(rows["annot+struct"][1]),
    auprc_annot_test=float(rows["annot"][1]),   auprc_annot_val=float(rows["annot"][0]),
    auprc_struct_test=float(rows["struct"][1]), auprc_struct_val=float(rows["struct"][0]),
    auprc_combined_val=float(rows["annot+struct"][0]), auroc_combined_test=float(rows["annot+struct"][2]),
    n_train=int(len(D["train"][2])), n_val=int(len(D["val"][2])), n_test=int(len(D["test"][2])),
    pair_transform="concat(|A-B|, A*B), order-invariant",
    note="E-judge: random=0.5 (balanced, degree-matched negatives); AUPRC >> 0.5 means knowledge carries transferable pair-level signal",
)
out = os.path.join(Paths().out_dir, "probe_knowledge_lr.json")
save_json(res, out)
print("OUT", out)
