import os, glob, time
from multiprocessing import Pool
import numpy as np

CIF_DIR = "workdir/afdb"
NPZ_DIR = "workdir/features/struct_npz"
TSV     = "workdir/features/features_struct.tsv"

def parse_cif(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    n = len(lines); i = 0
    while i < n:
        if lines[i].startswith("loop_"):
            j = i + 1; headers = []
            while j < n and lines[j].lstrip().startswith("_"):
                headers.append(lines[j].split()[0]); j += 1
            if headers and headers[0].startswith("_atom_site."):
                b_i  = headers.index("_atom_site.B_iso_or_equiv")
                id_i = headers.index("_atom_site.label_atom_id")
                sq_i = headers.index("_atom_site.label_seq_id")
                xs_i = headers.index("_atom_site.Cartn_x")
                ys_i = headers.index("_atom_site.Cartn_y")
                zs_i = headers.index("_atom_site.Cartn_z")
                seqid, pl, xyz = [], [], []
                while j < n and not lines[j].startswith("#"):
                    t = lines[j].split()
                    if len(t) == len(headers) and t[id_i] == "CA":
                        try:
                            seqid.append(int(float(t[sq_i])))
                            pl.append(float(t[b_i]))
                            xyz.append((float(t[xs_i]), float(t[ys_i]), float(t[zs_i])))
                        except ValueError: pass
                    j += 1
                return seqid, pl, xyz
            i = j
        else:
            i += 1
    return [], [], []

def worker(path):
    acc = os.path.basename(path)[:-4]
    try:
        seqid, pl, xyz = parse_cif(path)
        if not pl:
            return acc, 0, 0, 0, 0, 0, "empty"
        a = np.asarray(pl, dtype=np.float32)
        np.savez_compressed(os.path.join(NPZ_DIR, acc + ".npz"),
                            seq=np.asarray(seqid, dtype=np.int32),
                            plddt=a, ca=np.asarray(xyz, dtype=np.float32))
        return acc, len(a), float(a.mean()), float((a < 50).mean()), float((a < 70).mean()), float((a < 90).mean()), ""
    except Exception as e:
        return acc, 0, 0, 0, 0, 0, repr(e)[:80]

if __name__ == "__main__":
    os.makedirs(NPZ_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TSV), exist_ok=True)
    files = sorted(glob.glob(os.path.join(CIF_DIR, "*.cif")))
    print(f"files={len(files)} workers=8", flush=True)
    rows, errs, t0 = [], [], time.time()
    with Pool(8) as p:
        for k, r in enumerate(p.imap_unordered(worker, files, chunksize=16), 1):
            rows.append(r)
            if r[6]: errs.append(r)
            if k % 500 == 0 or k == len(files):
                print(f"[{k}/{len(files)}] err={len(errs)} elapsed={time.time()-t0:.0f}s", flush=True)
    with open(TSV, "w") as f:
        f.write("acc\tn_res\tmean_plddt\tfrac_lt50\tfrac_lt70\tfrac_lt90\n")
        for acc, nres, m, f50, f70, f90, _ in sorted(rows):
            if nres:
                f.write(f"{acc}\t{nres}\t{m:.2f}\t{f50:.4f}\t{f70:.4f}\t{f90:.4f}\n")
    lens = [r[1] for r in rows if r[1]]
    gmean = sum(r[2] * r[1] for r in rows if r[1]) / sum(lens)
    print(f"DONE parsed={len(lens)} errors={len(errs)} mean_len={sum(lens)/len(lens):.0f} global_mean_plddt={gmean:.2f}", flush=True)
    for r in errs[:10]: print("ERR", r[0], r[6], flush=True)
