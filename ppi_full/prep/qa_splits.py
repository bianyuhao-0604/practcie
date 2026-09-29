import os, sys, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import load_pairs, validate_splits

splits = {}
for name in ["train", "val", "test"]:
    df = load_pairs(f"workdir/data/{name}.csv")
    splits[name] = df
    pairs = {frozenset((a, b)) for a, b in zip(df.pid_a, df.pid_b)}
    loops = sum(a == b for a, b in zip(df.pid_a, df.pid_b))
    print(f"{name}: rows={len(df)} uniq_pairs={len(pairs)} dup={len(df)-len(pairs)} "
          f"self_loops={loops} pos={int(df.label.sum())} neg={int((df.label==0).sum())}")

validate_splits(splits)
print("LEAKAGE: none (protein-disjoint)")

for i, j in itertools.combinations(splits, 2):
    pi = {frozenset((a,b)) for a,b in zip(splits[i].pid_a, splits[i].pid_b)}
    pj = {frozenset((a,b)) for a,b in zip(splits[j].pid_a, splits[j].pid_b)}
    print(f"pair_overlap {i}&{j}: {len(pi & pj)}")
