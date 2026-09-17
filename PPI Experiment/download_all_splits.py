#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_all_splits.py — Intra0/1/2 完整数据集（pos+neg）蛋白单体结构下载
==========================================================
数据集划分（官方负样本, 每个 split 都有）:
  train = Intra1_pos_rr.txt + Intra1_neg_rr.txt
  val   = Intra0_pos_rr.txt + Intra0_neg_rr.txt
  test  = Intra2_pos_rr.txt + Intra2_neg_rr.txt

特性:
  1. 增量复用: 已在 monomer_metadata.json 的蛋白跳过 API 查询,
     已有 PDB 文件的跳过下载（断点续传, 可随时中断重跑）
  2. 负样本蛋白一并下载（之前只处理过 Intra0_pos, Intra0_neg 是全新蛋白）
  3. 元数据写入 "splits" 标签（正式实验按官方划分取样）
  4. 泄漏检查: 任何 split 的 neg 对 vs 全部 pos 对, 必须零重叠
  5. AFDB 未命中 → 待救援列表（复用 rescue 逻辑）
  6. v2.0 全部修复保留: entry_id 命名 / 异常打印 / v6 URL / .part 原子写入

运行:
  python download_all_splits.py --skip-cif --max-workers 4
"""
import os, sys, json, csv, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
AF_API_BASE  = "https://alphafold.ebi.ac.uk/api/prediction/{}"
AF_FILES_DIR = "https://alphafold.ebi.ac.uk/files"
OUTPUT_ROOT  = Path("af_output")
MONO_DIR     = OUTPUT_ROOT / "monomers"
META_DIR     = OUTPUT_ROOT / "metadata"
LOGS_DIR     = OUTPUT_ROOT / "logs"

# 完整 6 文件配置（文件不存在自动跳过并提示）
SPLIT_FILES = {
    "train": ["Intra1_pos_rr.txt", "Intra1_neg_rr.txt"],
    "val":   ["Intra0_pos_rr.txt", "Intra0_neg_rr.txt"],
    "test":  ["Intra2_pos_rr.txt", "Intra2_neg_rr.txt"],
}

REQUEST_TIMEOUT  = 30
DOWNLOAD_RETRIES = 3

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SplitDownload/1.0"})


def safe_filename(acc): return "".join(c if c.isalnum() or c in "._-" else "_" for c in acc)


# ============================================================
# Step 1: 解析全部 6 个文件
# ============================================================
def parse_pair_file(p: Path):
    toks = p.read_text().split()
    if len(toks) % 2:
        print(f"[WARN] {p.name} token 数为奇数, 丢弃最后一个: {toks[-1]}")
        toks = toks[:-1]
    return [(toks[i], toks[i+1]) for i in range(0, len(toks), 2)]


def parse_all_splits():
    pairs_by_split, acc_splits = {}, {}
    total = 0
    print("[STEP] 解析数据集文件 (完整 6 文件配置) ...")
    for split, files in SPLIT_FILES.items():
        pairs_by_split[split] = []
        for fn in files:
            fp = Path(fn)
            if not fp.exists():
                print(f"  {split:<6} {fn}: ⚠️ 文件不存在, 跳过")
                continue
            ps = parse_pair_file(fp)
            pairs_by_split[split].append((fn, ps))
            total += len(ps)
            for a, b in ps:
                for acc in (a, b):
                    acc_splits.setdefault(acc, set()).add(split)
            label = "pos" if "pos" in fn else "neg"
            print(f"  {split:<6} {fn} ({label}): {len(ps):>8} 对")
    return pairs_by_split, acc_splits, total


def leak_check(pairs_by_split):
    """任何 split 的 neg 对不得与任何 split 的 pos 对重叠"""
    all_pos, leaks = set(), []
    for split, files in pairs_by_split.items():
        for fn, ps in files:
            if "neg" in fn:
                continue
            all_pos |= {frozenset(p) for p in ps}
    for split, files in pairs_by_split.items():
        for fn, ps in files:
            if "neg" not in fn:
                continue
            for p in ps:
                if frozenset(p) in all_pos:
                    leaks.append((split, fn, p))
    if leaks:
        print(f"[INFO] 泄漏检查: {len(leaks)} 对重叠  ⚠️ 请排查: {leaks[:5]}")
    else:
        print(f"[INFO] 泄漏检查 (neg∩pos 跨全部 split): 0 对  ✓ 干净")
    return leaks


# ============================================================
# Step 2: 增量 API 查询
# ============================================================
def query_af_metadata(acc):
    url = AF_API_BASE.format(acc)
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0]
                return None
            elif r.status_code == 404:
                return None
            print(f"  [HTTP {r.status_code}] {acc} (尝试 {attempt+1})")
        except Exception as e:
            print(f"  [查询异常 {attempt+1}/{DOWNLOAD_RETRIES}] {acc}: {type(e).__name__}: {e}")
            time.sleep(1)
    return None


def batch_query_incremental(all_accs):
    META_DIR.mkdir(parents=True, exist_ok=True)
    meta_file = META_DIR / "monomer_metadata.json"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    new_accs = sorted(a for a in all_accs if a not in meta)
    reuse    = len(all_accs) - len(new_accs)
    print(f"\n[STEP] API 查询 (增量): 新 {len(new_accs)} 个 | 复用已有 {reuse} 个")
    print("[INFO] 注: Intra0_neg / Intra1 / Intra2 的蛋白均为新查询对象")
    for acc in tqdm(new_accs, desc="API 查询"):
        m = query_af_metadata(acc)
        meta[acc] = None if m is None else {
            "entryId":   m.get("entryId"), "gene": m.get("gene"),
            "description": m.get("uniprotDescription"),
            "organism":  m.get("organismScientificName"),
            "sequence":  m.get("sequence"),
            "length":    len(m.get("sequence") or ""),
            "mean_plddt": m.get("globalMetricValue"),
            "frac_very_low": m.get("fractionPlddtVeryLow"),
            "frac_low":  m.get("fractionPlddtLow"),
            "frac_conf": m.get("fractionPlddtConfident"),
            "frac_very_high": m.get("fractionPlddtVeryHigh"),
            "pdbUrl":    m.get("pdbUrl"), "cifUrl": m.get("cifUrl"),
            "latestVersion": m.get("latestVersion"), "source": "afdb",
        }
        time.sleep(0.2)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    misses = [k for k in all_accs if meta.get(k) is None]
    print(f"[INFO] 命中: {len(all_accs) - len(misses)}/{len(all_accs)} | 未命中: {len(misses)}")
    if misses:
        miss_file = LOGS_DIR / "not_in_afdb_all.txt"
        miss_file.parent.mkdir(parents=True, exist_ok=True)
        miss_file.write_text("\n".join(sorted(misses)))
        print(f"[INFO] 未命中列表: {miss_file} (救援: 复用 rescue 脚本, 改输入文件名即可)")
    return meta, misses


# ============================================================
# Step 3: 下载（断点续传, v2.0 修复版）
# ============================================================
def download_file(url, dest, retries=DOWNLOAD_RETRIES):
    if not url:
        return False
    for attempt in range(retries):
        try:
            r = SESSION.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                tmp.rename(dest)
                return True
            elif r.status_code == 404:
                print(f"  [404] {url}")
                return False
            print(f"  [HTTP {r.status_code}] 尝试{attempt+1}: {url}")
        except Exception as e:
            print(f"  [下载异常 {attempt+1}/{retries}] {url}: {type(e).__name__}: {e}")
            time.sleep(2)
    return False


def extract_plddt_from_pdb(pdb_path):
    vals, seen = [], set()
    try:
        for line in pdb_path.read_text(errors="ignore").splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                key = line[22:27]
                if key not in seen:
                    seen.add(key)
                    vals.append(round(float(line[60:66]), 2))
    except Exception as e:
        print(f"  [pLDDT提取失败] {pdb_path.name}: {e}")
        return None
    return vals or None


def download_monomer(acc, m, skip_cif=False):
    """v2.0 修复版: entry_id 唯一变量名 + API URL 优先 + .part 原子写入"""
    if m is None:
        return False
    entry_id = m.get("entryId", f"AF-{acc}-F1")
    pdb_url  = m.get("pdbUrl") or f"{AF_FILES_DIR}/{entry_id}-model_v6.pdb"

    pdb_dest  = MONO_DIR / f"{safe_filename(acc)}_pdb.pdb"
    lddt_dest = MONO_DIR / f"{safe_filename(acc)}_plddt.json"

    ok = (pdb_dest.exists() and pdb_dest.stat().st_size > 0) \
         or download_file(pdb_url, pdb_dest)

    if not skip_cif:
        cif_url = m.get("cifUrl") or f"{AF_FILES_DIR}/{entry_id}-model_v6.cif"
        cif_dest = MONO_DIR / f"{safe_filename(acc)}_cif.cif"
        if not (cif_dest.exists() and cif_dest.stat().st_size > 0):
            download_file(cif_url, cif_dest)

    if ok and not lddt_dest.exists():
        plddt = extract_plddt_from_pdb(pdb_dest)
        if plddt:
            lddt_dest.write_text(json.dumps(plddt))
    return ok


def batch_download(meta, workers=4, skip_cif=False):
    hits = {k: v for k, v in meta.items() if v is not None}
    MONO_DIR.mkdir(parents=True, exist_ok=True)
    todo = {k: v for k, v in hits.items()
            if not (MONO_DIR / f"{safe_filename(k)}_pdb.pdb").exists()}
    print(f"\n[STEP] 下载单体: 待下载 {len(todo)} 个 | 已有跳过 {len(hits)-len(todo)} 个")
    if not todo:
        print("[INFO] 全部已有, 无需下载")
        return

    ok_count, fails = 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_monomer, a, m, skip_cif): a for a, m in todo.items()}
        with tqdm(total=len(todo), desc="单体下载") as pbar:
            for f in as_completed(futs):
                acc = futs[f]
                try:
                    if f.result():
                        ok_count += 1
                    else:
                        fails.append(acc)
                except Exception as e:
                    print(f"\n  [ERROR] {acc}: {type(e).__name__}: {e}")
                    fails.append(acc)
                pbar.update(1)
    print(f"[INFO] 下载成功: {ok_count}/{len(todo)}")
    if fails:
        (LOGS_DIR / "download_failed.txt").write_text("\n".join(fails))
        print(f"[INFO] 失败列表: {LOGS_DIR}/download_failed.txt")


# ============================================================
# Step 4: split 标签 + 覆盖率报告
# ============================================================
def finalize(meta, acc_splits, pairs_by_split):
    for acc, splits in acc_splits.items():
        if meta.get(acc):
            meta[acc]["splits"] = sorted(splits)
    meta_file = META_DIR / "monomer_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    def ok_pair(a, b):
        ma, mb = meta.get(a), meta.get(b)
        return bool(ma and mb and ma.get("sequence") and mb.get("sequence"))

    print(f"\n{'='*58}\n 覆盖率报告（蛋白对双端序列齐全）\n{'='*58}")
    rows = []
    for split in ("train", "val", "test"):
        for fn, ps in pairs_by_split.get(split, []):
            n = sum(1 for a, b in ps if ok_pair(a, b))
            rows.append([split, fn, len(ps), n, f"{100*n/len(ps):.1f}%"])
            print(f"  {split:<6} {fn:<22} {n:>8}/{len(ps):<8} ({100*n/len(ps):.1f}%)")

    csv_path = OUTPUT_ROOT / "split_coverage.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "n_pairs", "n_covered", "coverage"])
        w.writerows(rows)

    n_prot = sum(1 for a in acc_splits if meta.get(a) and meta[a].get("sequence"))
    n_stru = len(list(MONO_DIR.glob("*_pdb.pdb")))
    print(f"\n[INFO] 蛋白: 序列可用 {n_prot}/{len(acc_splits)} | 结构 {n_stru}")
    print(f"[INFO] 标签文件: {meta_file} (splits 字段)")
    print(f"[INFO] 覆盖表: {csv_path}")
    print(f"{'='*58}")


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Intra0/1/2 完整数据集 (pos+neg) 单体结构下载")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--skip-cif", action="store_true",
                    help="跳过 CIF 下载（省 ~2/3 磁盘和时间, PDB/pLDDT 不受影响）")
    args = ap.parse_args()

    for d in (MONO_DIR, META_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    pairs_by_split, acc_splits, total = parse_all_splits()
    leak_check(pairs_by_split)
    print(f"\n[INFO] 蛋白对总数: {total} | 唯一蛋白: {len(acc_splits)}")

    meta, misses = batch_query_incremental(set(acc_splits))
    batch_download(meta, workers=args.max_workers, skip_cif=args.skip_cif)
    finalize(meta, acc_splits, pairs_by_split)


if __name__ == "__main__":
    main()
