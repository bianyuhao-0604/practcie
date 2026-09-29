#!/usr/bin/env python3
"""verify_download.py — 验证 af_output 下载质量（沙箱已测试）"""
import json, random, statistics
from pathlib import Path

BASE = Path("af_output")
MONO = BASE / "monomers"
META = BASE / "metadata" / "monomer_metadata.json"

pdb_files  = list(MONO.glob("*_pdb.pdb"))
cif_files  = list(MONO.glob("*_cif.cif"))
lddt_files = list(MONO.glob("*_plddt.json"))
print(f"[文件数] PDB: {len(pdb_files)}  CIF: {len(cif_files)}  pLDDT: {len(lddt_files)}")

# 异常检测
empty_pdb, bad_pdb, empty_plddt = [], [], []
for pdb in pdb_files:
    if pdb.stat().st_size == 0:
        empty_pdb.append(pdb.name); continue
    text = pdb.read_text(errors="ignore")
    if "ATOM" not in text or " CA " not in text:
        bad_pdb.append(pdb.name)

for l in lddt_files:
    try:
        data = json.loads(l.read_text())
        if not isinstance(data, list) or len(data) == 0:
            empty_plddt.append(l.name)
    except Exception:
        empty_plddt.append(l.name)

sizes = [p.stat().st_size for p in pdb_files if p.stat().st_size > 0]
print(f"[异常] 空 PDB: {len(empty_pdb)} | 损坏: {len(bad_pdb)} | 空pLDDT: {len(empty_plddt)}")
if bad_pdb: print(f"       损坏列表(前10): {bad_pdb[:10]}")
if empty_pdb: print(f"       空文件(前10): {empty_pdb[:10]}")
if sizes:
    print(f"[大小] 总量: {sum(sizes)/1e9:.2f} GB | 中位数: {statistics.median(sizes)/1024:.0f} KB")

# 元数据交叉验证
with open(META) as f:
    meta = json.load(f)
hits = {k for k, v in meta.items() if v is not None}
downloaded = {p.stem.replace("_pdb", "") for p in pdb_files}
missing = hits - downloaded
print(f"[交叉] 命中: {len(hits)} | 下载: {len(downloaded)} | 命中但缺文件: {len(missing)}")

# pLDDT 抽样
random.seed(0)
valid = [p for p in lddt_files if p.stat().st_size > 10]
sample = random.sample(valid, min(20, len(valid)))
low = []
for l in sample:
    vals = json.loads(l.read_text())
    m = sum(vals)/len(vals) if vals else 0
    if m < 50: low.append((l.stem, round(m, 1)))
print(f"[pLDDT抽样 {len(sample)} 个] 平均值<50(低质量): {len(low)} {low[:5]}")

# 磁盘
def dir_size(d):
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
print(f"[磁盘] af_output 总占用: {dir_size(BASE)/1e9:.2f} GB")

ok = len(pdb_files) - len(empty_pdb) - len(bad_pdb)
pct = 100 * ok / max(1, len(pdb_files))
print(f"\n{'='*40}\n结论: {ok}/{len(pdb_files)} ({pct:.1f}%) PDB 可用\n{'='*40}")
