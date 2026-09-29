#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rescue_missing104.py — 救援 AFDB 未命中的 104 个蛋白
==========================================================
在 rescue_missing35.py 基础上的修复:
  [FIX-1] MISS_FILE 改为 not_in_afdb_all.txt (全数据集 6 文件版)
  [FIX-2] 写回 metadata 时保留 download_all_splits.py 写入的 splits 标签
          (否则救援蛋白失去 split 归属, 正式实验按 split 取样时会被跳过!)

运行: python rescue_missing104.py
"""
import requests, json, time, csv
from pathlib import Path

AF_OUTPUT  = Path("af_output")
MONO_DIR   = AF_OUTPUT / "monomers"
META_FILE  = AF_OUTPUT / "metadata" / "monomer_metadata.json"
MISS_FILE  = AF_OUTPUT / "logs" / "not_in_afdb_all.txt"   # [FIX-1]

RESCUE_ROOT = Path("af_output_missing")
RESCUE_MONO = RESCUE_ROOT / "monomers"
RESCUE_ROOT.mkdir(parents=True, exist_ok=True)

s = requests.Session()
s.headers.update({"User-Agent": "rescue-missing/1.0"})
TIMEOUT = 30

def uni_get(acc):
    try:
        r = s.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json",
                  timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 200:
            d = r.json()
            return {
                "primary":  d.get("primaryAccession", acc),
                "entry":    d.get("uniProtkbId", ""),
                "len":      d.get("sequence", {}).get("length", 0),
                "sequence": d.get("sequence", {}).get("value", ""),
                "pdbs":     [x["id"] for x in d.get("uniProtKBCrossReferences", [])
                             if x.get("database") == "PDB"],
                "desc":     d.get("proteinDescription", {}).get(
                                "recommendedName", {}).get("fullName", {}).get("value", ""),
            }
        return None
    except Exception as e:
        print(f"  [ERR] {acc}: {type(e).__name__}")
        return None

def af_check(acc):
    try:
        r = s.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                return data[0]
        return None
    except Exception:
        return None

def download(url, dest):
    try:
        r = s.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest.with_suffix(dest.suffix + ".part"), "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            dest.with_suffix(dest.suffix + ".part").rename(dest)
            return True
        print(f"    [HTTP {r.status_code}] {url}")
    except Exception as e:
        print(f"    [ERR] {type(e).__name__}: {e}")
    return False

def extract_plddt(pdb_path):
    vals, seen = [], set()
    try:
        for line in pdb_path.read_text(errors="ignore").splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                key = line[22:27]
                if key not in seen:
                    seen.add(key)
                    vals.append(round(float(line[60:66]), 2))
    except Exception:
        return None
    return vals or None

def main():
    missing = [l.strip() for l in MISS_FILE.read_text().splitlines() if l.strip()]
    print(f"[INFO] 待救援: {len(missing)} 个 accession\n")

    with open(META_FILE, encoding="utf-8") as f:
        metadata = json.load(f)

    report, rescued_struct, rescued_seq, dead = [], 0, 0, 0

    for acc in missing:
        rec = {"acc": acc, "cause": "", "action": "", "structure": "", "seq_saved": False}
        print(f"--- {acc} ---")

        info = uni_get(acc)
        if info is None:
            rec["cause"], rec["action"] = "UniProt 废弃/不存在", "从数据集剔除"
            metadata[acc] = None
            dead += 1
            report.append(rec); continue

        primary, seq, L = info["primary"], info["sequence"], info["len"]
        merged = (primary != acc)

        # 路径1: 已合并 → 主条目 AF 结构
        af_meta = None
        if merged:
            af_meta = af_check(primary)
            if af_meta:
                entry_id = af_meta.get("entryId", f"AF-{primary}-F1")
                if download(af_meta.get("pdbUrl") or
                            f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v6.pdb",
                            RESCUE_MONO / f"{acc}_pdb.pdb"):
                    plddt = extract_plddt(RESCUE_MONO / f"{acc}_pdb.pdb")
                    if plddt:
                        (RESCUE_MONO / f"{acc}_plddt.json").write_text(json.dumps(plddt))
                    rec.update(cause=f"已合并 → {primary}", action="下载主条目 AF 结构",
                               structure=f"{acc}_pdb.pdb (源: AF-{primary})")
                    rescued_struct += 1
        # 路径2: PDB 实验结构
        if not af_meta and info["pdbs"]:
            pdb_id = info["pdbs"][0]
            if download(f"https://files.rcsb.org/download/{pdb_id}.pdb",
                        RESCUE_MONO / f"{acc}_pdb.pdb"):
                rec.update(cause=f"AF无结构({'超长' if L > 2700 else '未收录'})",
                           action=f"下载 PDB 实验结构 {pdb_id}",
                           structure=f"{acc}_pdb.pdb (源: PDB {pdb_id}, 无pLDDT)")
                rescued_struct += 1
        # 路径3: 标记
        if not rec["structure"]:
            rec.update(cause=f"AF无结构({'超长' if L > 2700 else '未收录'}), 无PDB",
                       action="待本地 ColabFold" if L <= 2700 else "超长,建议剔除或截断")
        # [FIX-2] 序列写回: 保留 splits!
        if seq and len(seq) >= 30:
            old_splits = (metadata.get(acc) or {}).get("splits")
            entry = {
                "entryId": f"RESCUE-{acc}", "gene": None,
                "description": info["desc"] or info["entry"],
                "organism": "Homo sapiens", "sequence": seq, "length": L,
                "mean_plddt": None, "frac_very_low": None, "frac_low": None,
                "frac_conf": None, "frac_very_high": None,
                "pdbUrl": None, "cifUrl": None, "latestVersion": None,
                "source": "rescue_uniprot",
            }
            if old_splits:                       # ← 关键修复: 保留 split 归属
                entry["splits"] = old_splits
            metadata[acc] = entry
            rec["seq_saved"] = True
            rescued_seq += 1
        else:
            metadata[acc] = None
        report.append(rec)
        time.sleep(0.3)

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    csv_path = RESCUE_ROOT / "rescue_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["acc","cause","action","structure","seq_saved"])
        w.writeheader(); w.writerows(report)

    print(f"\n{'='*58}")
    print(f" 救援结果汇总")
    print(f"{'='*58}")
    print(f" 拿回结构: {rescued_struct} / {len(missing)}")
    print(f" 拿回序列: {rescued_seq} / {len(missing)} (splits 已保留)")
    print(f" 彻底无信息: {dead}")
    print(f" 报告: {csv_path}")
    print(f"{'='*58}")
    print("\n[下一步] python consolidate_rescue.py")

if __name__ == "__main__":
    main()
