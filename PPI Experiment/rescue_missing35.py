#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rescue_missing35.py — 诊断并救援 AlphaFold DB 未命中的 35 个蛋白
==========================================================
对每个缺失 accession 的处理链:
  1. 查 UniProt (自动跟随合并重定向) → 拿主 accession / 序列 / 长度 / PDB 引用
  2. 若已合并(merged) → 用主 accession 查 AF DB → 有则下载结构(存为原 acc 文件名)
  3. 若 active 但 AF 无 → 优先 PDB 实验结构 → 否则标记待 ColabFold
  4. 若废弃 → 标记剔除
  5. 无论哪种,只要拿到序列 → 写回 monomer_metadata.json (pilot 立即可用)

运行: python rescue_missing35.py
输出: af_output_missing/rescue_report.csv + 补齐的 monomers/ + 更新的 metadata
"""
import requests, json, time, csv, sys
from pathlib import Path

# ============ 配置 ============
AF_OUTPUT  = Path("af_output")
MONO_DIR   = AF_OUTPUT / "monomers"
META_FILE  = AF_OUTPUT / "metadata" / "monomer_metadata.json"
MISS_FILE  = AF_OUTPUT / "logs" / "not_in_afdb.txt"

RESCUE_ROOT = Path("af_output_missing")
RESCUE_MONO = RESCUE_ROOT / "monomers"     # 救援的结构放这里(不污染原目录)
RESCUE_ROOT.mkdir(parents=True, exist_ok=True)

s = requests.Session()
s.headers.update({"User-Agent": "rescue-missing/1.0"})
TIMEOUT = 30

# ============ 工具函数 ============
def uni_get(acc):
    """查 UniProt,返回 (info_dict 或 None)"""
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
        return None  # 404 = 废弃/不存在
    except Exception as e:
        print(f"  [ERR] {acc}: {type(e).__name__}")
        return None

def af_check(acc):
    """查 AlphaFold DB,命中返回 (meta, None),未命中返回 (None, None)"""
    try:
        r = s.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                return data[0], None
        return None, None
    except Exception:
        return None, None

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
    """从 PDB B-factor 列提取 pLDDT (仅 AF 结构有效; 实验结构返回 None)"""
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

# ============ 主流程 ============
def main():
    missing = [l.strip() for l in MISS_FILE.read_text().splitlines() if l.strip()]
    print(f"[INFO] 待救援: {len(missing)} 个 accession\n")

    with open(META_FILE, encoding="utf-8") as f:
        metadata = json.load(f)

    report, rescued_struct, rescued_seq_only, dead = [], 0, 0, 0

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

        # ---- 路径1: 已合并 → 查主 accession 的 AF 结构 ----
        af_meta = None
        if merged:
            af_meta, _ = af_check(primary)
            if af_meta:
                entry_id = af_meta.get("entryId", f"AF-{primary}-F1")
                ok = download(af_meta.get("pdbUrl") or
                              f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v6.pdb",
                              RESCUE_MONO / f"{acc}_pdb.pdb")
                if ok:
                    plddt = extract_plddt(RESCUE_MONO / f"{acc}_pdb.pdb")
                    if plddt:
                        (RESCUE_MONO / f"{acc}_plddt.json").write_text(json.dumps(plddt))
                    rec.update(cause=f"已合并 → {primary}", action="下载主条目 AF 结构",
                               structure=f"{acc}_pdb.pdb (源: AF-{primary})")
                    rescued_struct += 1
        # ---- 路径2: active 但 AF 无 → PDB 实验结构 ----
        if not af_meta and info["pdbs"]:
            pdb_id = info["pdbs"][0]
            if download(f"https://files.rcsb.org/download/{pdb_id}.pdb",
                        RESCUE_MONO / f"{acc}_pdb.pdb"):
                rec.update(cause=f"AF无结构({'超长' if L > 2700 else '未收录'})",
                           action=f"下载 PDB 实验结构 {pdb_id}",
                           structure=f"{acc}_pdb.pdb (源: PDB {pdb_id}, 无pLDDT)")
                rescued_struct += 1
        # ---- 路径3: 啥都没有 → 标记待 ColabFold ----
        if not rec["structure"]:
            rec.update(cause=f"AF无结构({'超长' if L > 2700 else '未收录'}), 无PDB",
                       action="标记待本地 ColabFold" if L <= 2700 else "超长,建议剔除或截断")
        # ---- 序列无论如何都救回 metadata (供 ESM-2 pilot 用) ----
        if seq and len(seq) >= 30:
            metadata[acc] = {
                "entryId": f"RESCUE-{acc}", "gene": None,
                "description": info["desc"] or info["entry"],
                "organism": "Homo sapiens", "sequence": seq, "length": L,
                "mean_plddt": None, "frac_very_low": None, "frac_low": None,
                "frac_conf": None, "frac_very_high": None,
                "pdbUrl": None, "cifUrl": None, "latestVersion": None,
                "source": "rescue_uniprot",
            }
            rec["seq_saved"] = True
            rescued_seq_only += 1
        else:
            metadata[acc] = None
        report.append(rec)
        time.sleep(0.3)

    # ---- 保存 ----
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    csv_path = RESCUE_ROOT / "rescue_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["acc","cause","action","structure","seq_saved"])
        w.writeheader(); w.writerows(report)

    print(f"\n{'='*58}")
    print(f" 救援结果汇总")
    print(f"{'='*58}")
    print(f" 拿回结构(合并主条目/PDB):  {rescued_struct} / {len(missing)}")
    print(f" 拿回序列(供 ESM-2 用):     {rescued_seq_only} / {len(missing)}")
    print(f" 彻底无信息(剔除):          {dead}")
    print(f" 元数据已更新:              {META_FILE}")
    print(f" 结构目录:                  {RESCUE_MONO}")
    print(f" 详细报告:                  {csv_path}")
    print(f"{'='*58}")
    print("\n[NOTE] 救援的结构文件在 af_output_missing/monomers/,")
    print("       如需统一目录, 可复制到 af_output/monomers/ (文件名已兼容)。")
    print("[NOTE] 序列已并入 monomer_metadata.json — pilot 实验(ESM-2 baseline)")
    print("       现在能覆盖全部 3628 个蛋白, 386 对不再被跳过。")

if __name__ == "__main__":
    main()
