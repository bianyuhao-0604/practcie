import os, sys
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "features")
from config import FeatureModels
from extract_seq_str import extract_esm2

FEAT  = "workdir/features"
FASTA = "workdir/data/sequences_primary.fasta"
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 3

primaries = [l.split("\t")[0] for l in open(f"{FEAT}/features_node.tsv").readlines()[1:]]
todo = []
for p in primaries:
    f = os.path.join(FEAT, "seq_" + p + ".npy")
    ok = False
    if os.path.exists(f) and os.path.getsize(f) > 256:
        try:
            z = np.load(f, mmap_mode="r")
            ok = (z.ndim == 2 and z.shape[1] == 1280 and z.dtype == np.float16)
        except Exception:
            ok = False
    if not ok:
        todo.append(p)
print(f"RESUME todo={len(todo)}/{len(primaries)} bs={bs}", flush=True)
if todo:
    extract_esm2(todo, FASTA, FEAT, FeatureModels(), bs)
n_ok = sum(1 for p in primaries
           if os.path.exists(os.path.join(FEAT, "seq_" + p + ".npy"))
           and os.path.getsize(os.path.join(FEAT, "seq_" + p + ".npy")) > 256)
print(f"EXTRACT_SEQ_DONE files_ok={n_ok}/{len(primaries)}", flush=True)
