# diagnose_state.py
import json
from pathlib import Path

meta_file = Path("af_output/metadata/monomer_metadata.json")
print(f"metadata 存在: {meta_file.exists()}")
if meta_file.exists():
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    print(f"metadata 条目: {len(meta)}")

mono = Path("af_output/monomers")
print(f"PDB 结构: {len(list(mono.glob('*_pdb.pdb')))}")
print(f"pLDDT 文件: {len(list(mono.glob('*_plddt.json')))}")
resc = Path("af_output_missing/monomers")
print(f"救援目录结构: {len(list(resc.glob('*.pdb'))) if resc.exists() else 0}")
