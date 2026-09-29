import csv, os

FEAT = "workdir/features"
SALVAGED = ["A9Z1Z3","P01562","Q3SXR2","Q5RGS3","Q5SWW7","Q6RUI8","Q7Z4U5","Q8N268","Q8TAB7","Q8WV35"]
open(f"{FEAT}/_salvaged_from_unisave.txt","w").write("\n".join(SALVAGED)+"\n")

pmap = {}
for line in open(f"{FEAT}/uniprot_primary_map.txt"):
    t = line.rstrip("\n").split("\t")
    if len(t) >= 2 and t[1]:
        pmap[t[0]] = t[1]

amap = {}
cand = ["workdir/afdb_id_map.txt", "workdir/features/afdb_id_map.txt", "afdb_id_map.txt", "prep/afdb_id_map.txt"]
found = next((c for c in cand if os.path.exists(c)), None)
print("afdb_id_map:", found)
if found:
    lines = open(found).read().splitlines()
    print("  sample:", lines[:3])
    for line in lines:
        t = line.split()
        if len(t) >= 2:
            amap[t[0]] = t[1]

def to_primary(acc):
    if acc in pmap and pmap[acc]:
        return pmap[acc]
    if acc in amap:
        return pmap.get(amap[acc], amap[acc])
    return acc

srows = list(csv.DictReader(open(f"{FEAT}/features_struct.tsv"), delimiter="\t"))
sfeat, unmatched = {}, []
for r in srows:
    if r["acc"] not in pmap and r["acc"] not in amap:
        unmatched.append(r["acc"])
    prim = to_primary(r["acc"])
    if prim not in sfeat:
        sfeat[prim] = (r["n_res"], r["mean_plddt"], r["frac_lt50"], r["frac_lt70"], r["frac_lt90"])

arows = list(csv.DictReader(open(f"{FEAT}/features_annot.tsv"), delimiter="\t"))
nodes, coll = {}, 0
for r in arows:
    prim = r["primary"] or to_primary(r["acc"])
    if prim in nodes:
        coll += 1
        continue
    nodes[prim] = r

COLS_S = ["n_res","mean_plddt","frac_lt50","frac_lt70","frac_lt90"]
with open(f"{FEAT}/features_node.tsv","w") as f:
    f.write("primary\tacc_uniprot\tname\treviewed\tlength\tn_pfam\tn_interpro\tn_pdb\tgo_f\tgo_c\tn_kw\tn_tm\tn_signal\tn_secondary\thas_struct\t" + "\t".join(COLS_S) + "\n")
    for prim in sorted(nodes):
        r = nodes[prim]
        s = sfeat.get(prim)
        vals = list(s) if s else ["NA"]*5
        f.write("\t".join([prim, r["acc"], r["name"], r["reviewed"], r["length"], r["n_pfam"], r["n_interpro"], r["n_pdb"], r["go_f"], r["go_c"], r["n_kw"], r["n_tm"], r["n_signal"], r["n_secondary"], "1" if s else "0"] + vals) + "\n")

wsum = sum(1 for p in nodes if p in sfeat)
print(f"JOIN annot_rows={len(arows)} unique_primary={len(nodes)} collapsed={coll}")
print(f"JOIN struct_rows={len(srows)} resolved={len(sfeat)} unmatched={len(unmatched)} {unmatched[:5]}")
print(f"JOIN nodes_with_struct={wsum} without_struct={len(nodes)-wsum}")
print("OUT workdir/features/features_node.tsv")
