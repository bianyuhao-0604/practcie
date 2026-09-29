import os, re, glob
from multiprocessing import Pool

IN, OUTD = "workdir/uniprot", "workdir/features"

def worker(path):
    acc0 = os.path.basename(path)[:-4]
    try:
        primary = ""; sec = 0; name = ""; rev = 0; L = 0
        n_pfam = n_ipr = n_pdb = go_f = go_c = n_kw = n_tm = n_sig = 0
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("AC   ") and not primary:
                    t = [x.strip() for x in line[5:].split(";") if x.strip()]
                    primary = t[0]; sec = len(t) - 1
                elif line.startswith("ID   ") and not name:
                    p = line.split()
                    name = p[1]
                    rev = 1 if len(p) > 2 and p[2].startswith("Reviewed") else 0
                    mm = re.search(r"(\d+)\s+AA", line); L = int(mm.group(1)) if mm else 0
                elif line.startswith("DR   "):
                    db = line[5:].split(";")[0]
                    if db == "Pfam": n_pfam += 1
                    elif db == "InterPro": n_ipr += 1
                    elif db == "PDB": n_pdb += 1
                    elif db == "GO":
                        f2 = line.split(";")
                        if len(f2) > 2:
                            go_f += f2[2].strip().startswith("F:")
                            go_c += f2[2].strip().startswith("C:")
                elif line.startswith("KW   "):
                    n_kw += len([x for x in line[5:].split(";") if x.strip() and x.strip() != "."])
                elif line.startswith("FT   TRANSMEM"): n_tm += 1
                elif line.startswith("FT   SIGNAL"):    n_sig += 1
        return acc0, primary, name, rev, L, n_pfam, n_ipr, n_pdb, go_f, go_c, n_kw, n_tm, n_sig, sec, ""
    except Exception as e:
        return acc0, "", "", 0, 0, 0,0,0,0,0,0,0,0, 0, repr(e)[:80]

if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IN, "*.txt")))
    print(f"files={len(files)}", flush=True)
    rows, errs = [], []
    with Pool(8) as p:
        for k, r in enumerate(p.imap_unordered(worker, files, chunksize=32), 1):
            rows.append(r)
            if r[14]: errs.append(r)
            if k % 2000 == 0 or k == len(files):
                print(f"[{k}/{len(files)}] err={len(errs)}", flush=True)
    with open(f"{OUTD}/features_annot.tsv", "w") as f:
        f.write("acc\tprimary\tname\treviewed\tlength\tn_pfam\tn_interpro\tn_pdb\tgo_f\tgo_c\tn_kw\tn_tm\tn_signal\tn_secondary\n")
        for r in sorted(rows):
            if not r[14]:
                f.write("\t".join(map(str, r[:14])) + "\n")
    with open(f"{OUTD}/uniprot_primary_map.txt", "w") as f:
        for r in sorted(rows):
            if not r[14]:
                f.write(f"{r[0]}\t{r[1]}\t{r[13]}\n")
    ok = [r for r in rows if not r[14]]
    print(f"DONE parsed={len(ok)} errors={len(errs)} primary_changed={sum(1 for r in ok if r[1]!=r[0])} reviewed={sum(r[3] for r in ok)} len_zero={sum(1 for r in ok if r[4]==0)}", flush=True)
