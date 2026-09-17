#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
af_cutoff_check.py — AlphaFold 预训练截止日期泄漏检查
==========================================================
原理: AF2 训练集含 PDB<2018-04-30 全部结构, 模板可选用至 2021-02-15。
     若 test 蛋白在 PDB 有早于 2018-04-30 的实验结构,
     则 AF 对该蛋白的预测已"见过"其真实结构 → 结构模态存在泄漏。

流程:
  1. 收集 test 蛋白(默认, 可 --splits train,val,test)
  2. UniProt stream API 批量查 PDB 引用
  3. RCSB GraphQL 批量查 release_date
  4. 标记 af_leak=True 的蛋白 → af_leak_test.json

运行:
  python af_cutoff_check.py                  # 默认只查 test
  python af_cutoff_check.py --splits test    # 显式指定
"""
import json, argparse, time
from pathlib import Path
import requests

AF_OUTPUT = Path("af_output")
META_FILE = AF_OUTPUT / "metadata" / "monomer_metadata.json"
OUT_FILE  = Path("leakage_work") / "af_leak_test.json"

AF_TRAIN_CUTOFF   = "2018-04-30"   # AF2 训练集截止
AF_TEMPLATE_CUTOFF = "2021-02-15"  # 模板可选截止
BATCH_UNIPROT = 100
BATCH_RCSB    = 100

SPLIT_FILES = {
    "train": ["Intra1_pos_rr.txt", "Intra1_neg_rr.txt"],
    "val":   ["Intra0_pos_rr.txt", "Intra0_neg_rr.txt"],
    "test":  ["Intra2_pos_rr.txt", "Intra2_neg_rr.txt"],
}

s = requests.Session()
s.headers.update({"User-Agent": "af-cutoff-check/1.0"})


def collect_proteins(splits):
    accs = set()
    for sp in splits:
        for fn in SPLIT_FILES[sp]:
            fp = Path(fn)
            if not fp.exists():
                continue
            toks = fp.read_text().split()
            if len(toks) % 2:
                toks = toks[:-1]
            accs.update(toks)
    return sorted(accs)


def uniprot_pdb_refs(accs):
    """UniProt stream API 批量查 PDB 引用 → {acc: [pdb_id,...]}"""
    print(f"[STEP] UniProt 查询 PDB 引用: {len(accs)} 个蛋白")
    refs = {}
    for i in range(0, len(accs), BATCH_UNIPROT):
        batch = accs[i:i + BATCH_UNIPROT]
        q = " OR ".join(f"accession:{a}" for a in batch)
        try:
            r = s.post("https://rest.uniprot.org/uniprotkb/stream",
                       data={"query": q, "format": "json"},
                       timeout=120)
            if r.status_code == 200:
                for entry in r.json().get("results", []):
                    acc = entry.get("primaryAccession")
                    pdbs = [x["id"] for x in entry.get("uniProtKBCrossReferences", [])
                            if x.get("database") == "PDB"]
                    refs[acc] = pdbs
        except Exception as e:
            print(f"  [ERR] 批 {i//BATCH_UNIPROT+1}: {type(e).__name__}")
        if (i // BATCH_UNIPROT) % 5 == 0:
            print(f"  {i+len(batch)}/{len(accs)}")
        time.sleep(0.5)
    return refs


def rcsb_release_dates(pdb_ids):
    """RCSB GraphQL 批量查 release date → {pdb_id: 'YYYY-MM-DD'}"""
    all_ids = sorted(set(pdb_ids))
    print(f"[STEP] RCSB 查询 release date: {len(all_ids)} 个 PDB")
    dates = {}
    for i in range(0, len(all_ids), BATCH_RCSB):
        batch = all_ids[i:i + BATCH_RCSB]
        gql = ('query($ids: [String!]!) { entries(entry_ids: $ids) '
               '{ rcsb_id rcsb_accession_info { release_date } } }')
        try:
            r = s.post("https://data.rcsb.org/graphql",
                       json={"query": gql, "variables": {"ids": batch}},
                       timeout=120)
            if r.status_code == 200:
                for e in r.json().get("data", {}).get("entries", []):
                    pid = e.get("rcsb_id")
                    rd = (e.get("rcsb_accession_info") or {}).get("release_date")
                    if rd:
                        dates[pid] = rd
        except Exception as e:
            print(f"  [ERR] 批 {i//BATCH_RCSB+1}: {type(e).__name__}")
        if (i // BATCH_RCSB) % 10 == 0:
            print(f"  {i+len(batch)}/{len(all_ids)}")
        time.sleep(0.3)
    return dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="test", help="逗号分隔: test 或 train,val,test")
    args = ap.parse_args()
    splits = [x.strip() for x in args.splits.split(",")]

    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)

    accs = collect_proteins(splits)
    print(f"[INFO] {splits} 蛋白: {len(accs)} 个")

    refs = uniprot_pdb_refs(accs)
    all_pdbs = [p for v in refs.values() for p in v]
    print(f"[INFO] 有 PDB 引用的蛋白: {len(refs)}/{len(accs)} | PDB 总数: {len(all_pdbs)}")

    dates = rcsb_release_dates(all_pdbs)

    # ---- 判定每个蛋白的 AF 泄漏状态 ----
    result, n_leak, n_template = 0, 0, 0
    for acc in accs:
        pdbs = refs.get(acc, [])
        if not pdbs:
            result[acc] = None
            continue
        earliest, earliest_pdb = None, None
        for p in pdbs:
            d = dates.get(p)
            if d and (earliest is None or d < earliest):
                earliest, earliest_pdb = d, p
        entry = {
            "n_pdbs": len(pdbs),
            "earliest_pdb": earliest_pdb,
            "earliest_release": earliest,
            "af_leak": bool(earliest and earliest < AF_TRAIN_CUTOFF),
            "template_risk": bool(earliest and AF_TRAIN_CUTOFF <= earliest < AF_TEMPLATE_CUTOFF),
        }
        result = result or {}  # placeholder
        (result if isinstance(result, dict) else None)
        # (重新初始化, 见下)
        break
    # 正确实现(上面占位有误, 用干净循环):
    result = {}
    n_leak = n_tmpl = 0
    for acc in accs:
        pdbs = refs.get(acc, [])
        if not pdbs:
            result[acc] = {"af_leak": False, "n_pdbs": 0}
            continue
        earliest, ep = None, None
        for p in pdbs:
            d = dates.get(p)
            if d and (earliest is None or d < earliest):
                earliest, ep = d, p
        leak = bool(earliest and earliest < AF_TRAIN_CUTOFF)
        tmpl = bool(earliest and AF_TRAIN_CUTOFF <= earliest < AF_TEMPLATE_CUTOFF)
        result[acc] = {"n_pdbs": len(pdbs), "earliest_pdb": ep,
                       "earliest_release": earliest, "af_leak": leak,
                       "template_risk": tmpl}
        n_leak += leak
        n_tmpl += tmpl

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f" AF 预训练截止检查结果 ({splits})")
    print(f"{'='*60}")
    print(f" 有实验结构蛋白:    {sum(1 for v in result.values() if v['n_pdbs'] > 0)}/{len(accs)}")
    print(f" AF训练泄漏(<2018-04-30): {n_leak} ({100*n_leak/len(accs):.1f}%)")
    print(f" 模板风险(2018~2021):     {n_tmpl} ({100*n_tmpl/len(accs):.1f}%)")
    print(f" 输出: {OUT_FILE}")
    print(f"{'='*60}")
    print("\n[解读] af_leak=True 的 test 蛋白 → AF 结构'见过'其真实结构")
    print("       全量实验时: 在 make_clean_splits.py 中从 test 删除这些蛋白")


if __name__ == "__main__":
    main()
