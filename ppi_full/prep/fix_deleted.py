import csv, sys, requests

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
    live = ""
    try:
        r = requests.get("https://rest.uniprot.org/uniprotkb/search",
                         params={"query": f"accession:{acc}", "format": "tsv",
                                 "fields": "accession,id,organism_name,length"}, timeout=30)
        slines = [l for l in r.text.strip().splitlines() if l.strip()]
        if len(slines) > 1:
            live = slines[1].split("\t")[0]
    except Exception as e:
        print(acc, "search err", repr(e)[:60], flush=True)

    if live:
        resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{live}.txt", timeout=30)
        if resp.ok and resp.text.startswith("ID   "):
            open(f"{DIR}/{acc}.txt", "w", encoding="utf-8").write(resp.text)
            w = worker(f"{DIR}/{acc}.txt")
            new[acc] = w
            print(f"{acc}: MERGED -> primary {live} | {w[2]} len={w[4]}", flush=True)
            continue

    try:
        r2 = requests.get(f"https://rest.uniprot.org/unisave/{acc}?format=txt", timeout=30)
        if r2.ok and r2.text.startswith("ID   "):
            open(f"{DIR}/{acc}.txt", "w", encoding="utf-8").write(r2.text)
            w = worker(f"{DIR}/{acc}.txt")
            new[acc] = w
            print(f"{acc}: DELETED -> salvaged last release | {w[2]} len={w[4]}", flush=True)
            continue
        print(f"{acc}: unisave odd (status={r2.status_code} head={r2.text[:20]!r})", flush=True)
    except Exception as e:
        print(acc, "unisave err", repr(e)[:60], flush=True)

    new[acc] = (acc, "", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "dead")
    print(f"{acc}: UNRECOVERABLE", flush=True)

COLS = ["acc","primary","name","reviewed","length","n_pfam","n_interpro","n_pdb","go_f","go_c","n_kw","n_tm","n_signal","n_secondary"]
out = []
for r in rows:
    a = r["acc"]
    if a in new:
        out.append("\t".join(map(str, new[a][:14])))
    else:
        out.append("\t".join(r[h] for h in COLS))
open(TSV, "w").write("\t".join(COLS) + "\n" + "\n".join(out) + "\n")

plines = []
for line in open(PMAP):
    acc = line.split("\t")[0]
    plines.append(f"{new[acc][0]}\t{new[acc][1]}\t{new[acc][13]}\n" if acc in new else line)
open(PMAP, "w").writelines(plines)

dead = sorted(a for a, w in new.items() if w[4] == 0)
open("workdir/features/_uniprot_dead_accessions.txt", "w").write("\n".join(dead) + "\n")

still = sum(1 for line in open(TSV).readlines()[1:] if line.split("\t")[4] == "0")
merged = sum(1 for w in new.values() if w[4] and w[1] not in ("", w[0]))
print(f"FIXED2 len_zero_remaining={still} dead={dead} merged_live={merged}", flush=True)
