import os, sys, glob, shutil
import numpy as np

FEAT = "workdir/features"
RAW  = "workdir/data/sequences.fasta"
OUT  = "workdir/data/sequences_primary.fasta"
STD  = set("ACDEFGHIKLMNPQRSTVWY")

try:
    import esm
    ver = "unknown"
    try:
        import importlib.metadata as md
        ver = md.version("fair-esm")
    except Exception:
        pass
    fn = esm.pretrained.__dict__.get("esm2_t33_650M_UR50D")
    print(f"ENV fair-esm={ver} esm2_entry={'OK' if fn else 'MISSING'}")
    if fn is None:
        sys.exit(1)
except Exception as e:
    print("ENV FAIL import esm:", repr(e)[:80])
    sys.exit(1)

ck = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "esm2_t33_650M_UR50D.pt")
print("ENV weights_cached=", os.path.exists(ck), "(False = first run downloads ~2.7GB)")
du = shutil.disk_usage(os.path.abspath(FEAT))
print(f"ENV disk_free={du.free/2**30:.1f}GB (seq stage peak ~47GB)")

primaries = [l.split("\t")[0] for l in open(f"{FEAT}/features_node.tsv").readlines()[1:]]
assert len(set(primaries)) == len(primaries)
P = set(primaries)
print(f"PRIMARIES n={len(primaries)}")

entries, name, buf = [], None, []
for line in open(RAW):
    if line.startswith(">"):
        if name is not None:
            entries.append((name, "".join(buf)))
        name, buf = line[1:], []
    else:
        buf.append(line.strip())
if name is not None:
    entries.append((name, "".join(buf)))

def k_plain(h):
    t = h.split()
    return t[0] if t else ""
def k_pipe(h):
    t = h.split()
    tok = t[0] if t else ""
    p = tok.split("|")
    return p[1] if len(p) >= 3 else ""

plain = sum(1 for h, s in entries if k_plain(h) in P)
pipe  = sum(1 for h, s in entries if k_pipe(h) in P)
mode = "pipe" if pipe > plain else "plain"
kf = k_pipe if mode == "pipe" else k_plain
seqs = {}
for h, s in entries:
    k = kf(h)
    if k in P and k not in seqs and len(s) > 0:
        seqs[k] = s
print(f"FASTA entries={len(entries)} mode={mode} plain_hits={plain} pipe_hits={pipe} covered={len(seqs)}/{len(P)}")

missing = sorted(P - set(seqs))
print("MISSING", len(missing), missing[:12])
if missing:
    import requests
    for acc in missing:
        try:
            r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=30)
            s = "".join(x.strip() for x in r.text.splitlines()[1:] if not x.startswith(">"))
            if r.ok and len(s) > 0:
                seqs[acc] = s
                print(f"  fetched {acc} len={len(s)}")
            else:
                print(f"  FETCH_FAIL {acc} status={r.status_code}")
        except Exception as e:
            print(f"  FETCH_ERR {acc} {repr(e)[:60]}")
    missing = sorted(P - set(seqs))

if missing:
    print("NOT_READY missing_after_fetch=", len(missing), missing[:12])
    sys.exit(1)

from collections import Counter
cnt = Counter()
for p in primaries:
    cnt.update(ch for ch in seqs[p] if ch not in STD)
print("NONSTD", dict(cnt) if cnt else "none (ESM2 handles U/X/B/Z/O natively)")

with open(OUT, "w", newline="\n") as f:
    for p in primaries:
        s = seqs[p]
        f.write(">" + p + "\n")
        for i in range(0, len(s), 60):
            f.write(s[i:i+60] + "\n")
print("WROTE", OUT, "entries", len(primaries))

old = np.load(f"{FEAT}/lengths.npy")
newL = np.array([min(len(seqs[p]), 1022) for p in primaries], np.int64)
idx = list(np.nonzero(newL != old)[0])
print(f"LENGTHS diff_rows={len(idx)}")
for i in idx[:12]:
    print(f"  {primaries[i]} {int(old[i])} -> {int(newL[i])}")
np.save(f"{FEAT}/lengths.npy", newL)

have = set(os.path.basename(x)[4:-4] for x in glob.glob(f"{FEAT}/seq_*.npy"))
print(f"RESUME existing_seq_npy={len(have & P)}")
print("READY")
