#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""consolidate_rescue.py v2 — 救援成果并入 + 6 文件覆盖率报告（沙箱已验证）"""
import json, csv, shutil
from pathlib import Path

MONO      = Path("af_output/monomers")
META_FILE = Path("af_output/metadata/monomer_metadata.json")
RMONO     = Path("af_output_missing/monomers")
REPORT    = Path("af_output_missing/rescue_report.csv")

SPLIT_FILES = {
    "train": ["Intra1_pos_rr.txt", "Intra1_neg_rr.txt"],
    "val":   ["Intra0_pos_rr.txt", "Intra0_neg_rr.txt"],
    "test":  ["Intra2_pos_rr.txt", "Intra2_neg_rr.txt"],
}

# ---- 1. 读报告分类 ----
af_r, pdb_r, no_struct, no_seq = [], [], [], []
with open(REPORT, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        st = row.get("structure", "")
        if   "源: AF"  in st: af_r.append(row["acc"])
        elif "源: PDB" in st: pdb_r.append(row["acc"])
        else:                 no_struct.append(row["acc"])
        if row.get("seq_saved") != "True":
            no_seq.append(row["acc"])
print(f"[分类] AF合并救回: {len(af_r)} | PDB实验: {len(pdb_r)} | "
      f"仅序列: {len(no_struct)} | 无序列: {no_seq}")

# ---- 2. 复制结构 ----
copied = 0
for acc in af_r:
    for suf in ["_pdb.pdb", "_plddt.json"]:
        src = RMONO / f"{acc}{suf}"
        if src.exists():
            shutil.copy2(src, MONO / src.name); copied += 1
for acc in pdb_r:
    src = RMONO / f"{acc}_pdb.pdb"
    if src.exists():
        shutil.copy2(src, MONO / src.name); copied += 1
print(f"[复制] {copied} 个文件并入 af_output/monomers/")

# ---- 3. 更新 metadata（splits 已由 rescue 脚本保留, 此处只打 source 标签）----
meta = json.loads(META_FILE.read_text(encoding="utf-8"))
for acc in af_r:
    pj = MONO / f"{acc}_plddt.json"
    if pj.exists() and meta.get(acc):
        vals = json.loads(pj.read_text())
        if vals:
            meta[acc]["mean_plddt"] = round(sum(vals)/len(vals), 1)
        meta[acc]["source"] = "rescue_af_merged"
for acc in pdb_r:
    if meta.get(acc):
        meta[acc]["source"] = "rescue_pdb_experimental"
META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False)
                     , encoding="utf-8")
print(f"[元数据] source 标签已更新 → {META_FILE}")

# ---- 4. 6 文件覆盖率（序列 + 结构双口径）----
def parse(p):
    toks = p.read_text().split()
    if len(toks) % 2: toks = toks[:-1]
    return [(toks[i], toks[i+1]) for i in range(0, len(toks), 2)]

def has_seq(a):
    m = meta.get(a); return bool(m and m.get("sequence") and len(m["sequence"]) >= 30)
def has_pdb(a): return (MONO / f"{a}_pdb.pdb").exists()

print(f"\n{'='*68}")
print(f" 最终覆盖率报告（按官方 6 文件划分）")
print(f"{'='*68}")
rows = []
for split, files in SPLIT_FILES.items():
    for fn in files:
        ps = parse(Path(fn))
        n_seq  = sum(1 for a, b in ps if has_seq(a) and has_seq(b))
        n_full = sum(1 for a, b in ps if has_seq(a) and has_seq(b)
                     and has_pdb(a) and has_pdb(b))
        rows.append([split, fn, len(ps), n_seq, n_full,
                     f"{100*n_seq/len(ps):.1f}%", f"{100*n_full/len(ps):.1f}%"])
        print(f"  {split:<6}{fn:<22} 序列 {n_seq:>6}/{len(ps):<6}"
              f" ({100*n_seq/len(ps):.1f}%) |"
              f" 结构 {n_full:>6}/{len(ps):<6} ({100*n_full/len(ps):.1f}%)")

csv_path = Path("af_output") / "split_coverage_final.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "file", "n_pairs", "seq_covered", "full_covered",
                "seq_pct", "full_pct"])
    w.writerows(rows)

n_seq_prot  = sum(1 for v in meta.values() if v and v.get("sequence"))
n_pdb_total = len(list(MONO.glob("*_pdb.pdb")))
n_lddt      = len(list(MONO.glob("*_plddt.json")))
print(f"\n[INFO] 序列蛋白: {n_seq_prot} | 结构: {n_pdb_total} (带 pLDDT: {n_lddt})")
print(f"[INFO] 覆盖表: {csv_path}")
print(f"{'='*68}")
