import csv, re, os, sys
import requests

TSV  = "workdir/features/features_annot.tsv"
PMAP = "workdir/features/uniprot_primary_map.txt"
DIR  = "workdir/uniprot"

sys.path.insert(0, "prep")
from parse_uniprot import worker

rows = list(csv.DictReader(open(TSV), delimiter="\t"))
bad = [r["acc"] for r in rows if int(r["length"]) == 0]
print("len_zero:", bad, flush=True)

new = {}
for acc in bad:
    p = f"{DIR}/{acc}.txt"
    head = open(p, encoding="utf-8", errors="ignore").read(40) if os.path.exists(p) else ""
    if not head.startswith("ID   "):
        print(f"{acc}: bad file (head={head[:20]!r}) -> refetch", flush=True)
        try:
            resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.txt", timeout=30)
            print(f"  refetch status={resp.status_code} bytes={len(resp.text)}", flush=True)
            if resp.ok and resp.text.startswith("ID   "):
                open(p, "w", encoding="utf-8").write(resp.text)
        except Exception as e:
            print("  refetch failed:", repr(e)[:80], flush=True)
    r = worker(p)
    new[acc] = r
    print(f"  reparsed: primary={r[1]} name={r[2]} rev={r[3]} len={r[4]}", flush=True)

out = []
for r in rows:
    a = r["acc"]
    if a in new:
        w = new[a]
        out.append("\t".join(map(str, w[:14])))
    else:
        out.append("\t".join(r[h] for h in ["acc","primary","name","reviewed","length","n_pfam","n_interpro","n_pdb","go_f","go_c","n_kw","n_tm","n_signal","n_secondary"]))
open(TSV, "w").write("acc\tprimary\tname\treviewed\tlength\tn_pfam\tn_interpro\tn_pdb\tgo_f\tgo_c\tn_kw\tn_tm\tn_signal\tn_secondary\n" + "\n".join(out) + "\n")

lines = []
for line in open(PMAP):
    acc = line.split("\t")[0]
    lines.append(f"{new[acc][0]}\t{new[acc][1]}\t{new[acc][13]}\n" if acc in new else line)
open(PMAP, "w").writelines(lines)

still = sum(1 for line in open(TSV).readlines()[1:] if line.split("\t")[4] == "0")
print(f"FIXED len_zero_remaining={still}", flush=True)
