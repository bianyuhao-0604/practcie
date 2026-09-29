import csv, json, os
import numpy as np
import pandas as pd

MAX_LEN = 1022
FEAT = "workdir/features"

rows = list(csv.DictReader(open(f"{FEAT}/features_node.tsv"), delimiter="\t"))
pids = [r["primary"] for r in rows]
n = len(pids)
assert len(set(pids)) == n, "duplicate primary in features_node.tsv"
pid2idx = {p: i for i, p in enumerate(pids)}

n_alias = 0
for line in open(f"{FEAT}/uniprot_primary_map.txt"):
    t = line.rstrip("\n").split("\t")
    if len(t) >= 2 and t[1] and t[0] not in pid2idx and t[1] in pid2idx:
        pid2idx[t[0]] = pid2idx[t[1]]
        n_alias += 1

lengths = np.zeros(n, np.int64)
for i, r in enumerate(rows):
    L = int(r["n_res"]) if r["has_struct"] == "1" else int(r["length"])
    lengths[i] = max(min(L, MAX_LEN), 1)

plddt = np.zeros((n, MAX_LEN), np.float16)
miss_npz = []
for i, p in enumerate(pids):
    f = f"{FEAT}/struct_npz/{p}.npz"
    if not os.path.exists(f):
        miss_npz.append(p); continue
    z = np.load(f)["plddt"].astype(np.float16)
    Lc = min(z.size, MAX_LEN)
    plddt[i, :Lc] = z[:Lc]

json.dump({"pid2idx": pid2idx, "n": n,
           "has_text": [False]*n, "has_genome": [False]*n},
          open(f"{FEAT}/meta.json", "w"))
np.save(f"{FEAT}/lengths.npy", lengths)
np.save(f"{FEAT}/plddt.npy", plddt)

print(f"BUILD n={n} aliases={n_alias} plddt={plddt.shape} {plddt.dtype} missing_npz={len(miss_npz)}")

tot = set()
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"workdir/data/{split}.csv", dtype=str)
    m = set(df.pid_a) | set(df.pid_b)
    miss = {p for p in m if p not in pid2idx}
    tot |= miss
    print(f"{split}: rows={len(df)} unique_pids={len(m)} missing={len(miss)} sample={sorted(miss)[:5]}")

if tot:
    open(f"{FEAT}/_csv_missing_pids.txt", "w").write("\n".join(sorted(tot)) + "\n")
    print(f"TOTAL missing={len(tot)} -> _csv_missing_pids.txt")
else:
    print("CSV coverage: 100% — PairDataset will not assert")
