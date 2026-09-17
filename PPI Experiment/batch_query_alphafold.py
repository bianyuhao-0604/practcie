#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_query_alphafold.py  v2.0 (修复版)
=========================================
功能:
  1. 解析蛋白对文件（如 Intra0_pos_rr.txt，空格分隔、两两成对的 UniProt accession）
  2. 批量查询 AlphaFold DB API，获取单体元数据（含序列、pLDDT 统计）
  3. 批量下载单体结构（PDB + CIF + 从 PDB 提取的残基级 pLDDT）
  4. [实验性] 尝试下载 NVIDIA AF3 同源二聚体复合物（失败不影响主流程，可 --skip-complex 跳过）
  5. 为蛋白对生成 ColabFold/AlphaFold-Multimer 输入 FASTA
  6. 输出汇总 CSV

v2.0 修复清单:
  [FIX-1][关键] download_monomer 中 entryId / entry_id 变量名不一致导致 NameError
                （v1 全部 0/3593 下载成功的根因），现统一为 entry_id，
                且优先使用 API 响应中返回的 pdbUrl / cifUrl
  [FIX-2][关键] 所有异常不再被静默吞掉，失败时打印具体错误类型和信息
  [FIX-3][适配] AlphaFold DB 已升级 model_v6，URL 构造改用 v6
  [FIX-4][改进] 置信度数据改为从 PDB 的 B-factor 列提取 pLDDT（存为 {acc}_plddt.json）
                旧版的独立 summary_confidences.json 已不在 API 响应中
  [FIX-5][修复] main() 的 global 声明移至函数开头（v1 报 SyntaxError 的原因）
  [FIX-6][健壮] 下载使用 .part 临时文件 + 原子重命名，避免半截文件被断点续传误判为完成

运行:
  python batch_query_alphafold.py Intra0_pos_rr.txt
  python batch_query_alphafold.py Intra0_pos_rr.txt --skip-complex --max-workers 8
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# ============================================================
# 全局配置
# ============================================================
AF_API_BASE  = "https://alphafold.ebi.ac.uk/api/prediction/{}"
AF_FILES_DIR = "https://alphafold.ebi.ac.uk/files"

MAX_WORKERS      = 4        # 并发线程数（可用 --max-workers 覆盖）
SLEEP_BETWEEN    = 0.2      # API 查询间隔秒数（可用 --sleep 覆盖）
REQUEST_TIMEOUT  = 30
DOWNLOAD_RETRIES = 3

OUTPUT_ROOT = Path("af_output")
SUB_DIRS = {
    "monomer":   OUTPUT_ROOT / "monomers",               # {acc}_pdb.pdb / {acc}_cif.cif / {acc}_plddt.json
    "homodimer": OUTPUT_ROOT / "homodimers",             # 同源二聚体复合物（实验性）
    "fasta":     OUTPUT_ROOT / "fasta_for_prediction",   # 待本地预测的 FASTA
    "metadata":  OUTPUT_ROOT / "metadata",               # monomer_metadata.json
    "logs":      OUTPUT_ROOT / "logs",                   # not_in_afdb.txt / monomer_download_failed.txt
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BatchAFQuery/2.0"})


def setup_dirs():
    for d in SUB_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def safe_filename(acc: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in acc)


# ============================================================
# Step 1: 元数据查询
# ============================================================
def query_af_metadata(acc: str, session) -> dict:
    """查询单个 accession 的 AlphaFold 元数据。返回 dict（命中）或 None（未命中）。"""
    url = AF_API_BASE.format(acc)
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                return None
            elif r.status_code == 404:
                return None
            else:
                # FIX-2: 打印非预期状态码
                print(f"  [HTTP {r.status_code}] {acc} (尝试 {attempt + 1})")
        except Exception as e:
            # FIX-2: 打印具体异常
            print(f"  [查询异常 {attempt + 1}/{DOWNLOAD_RETRIES}] {acc}: "
                  f"{type(e).__name__}: {e}")
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(1)
    return None


def batch_query_monomers(unique_accs, session):
    """批量查询元数据，返回 {acc: meta_or_None}"""
    print(f"\n[STEP] 批量查询 {len(unique_accs)} 个 accession 的单体元数据 ...")
    meta_dict = {}
    for acc in tqdm(unique_accs, desc="API 查询"):
        meta_dict[acc] = query_af_metadata(acc, session)
        if SLEEP_BETWEEN > 0:
            time.sleep(SLEEP_BETWEEN)

    hits   = {k: v for k, v in meta_dict.items() if v is not None}
    misses = [k for k, v in meta_dict.items() if v is None]
    print(f"[INFO] 命中: {len(hits)} 个")
    print(f"[INFO] 未命中 (不在 AlphaFold DB): {len(misses)} 个")
    if misses:
        miss_file = SUB_DIRS["logs"] / "not_in_afdb.txt"
        miss_file.write_text("\n".join(misses))
        print(f"[INFO] 未命中列表已保存到: {miss_file}")

    # 保存精简元数据（含序列，供 pilot 实验和 FASTA 生成使用）
    slim = {}
    for acc, m in meta_dict.items():
        if m is None:
            slim[acc] = None
        else:
            slim[acc] = {
                "entryId":       m.get("entryId"),
                "gene":          m.get("gene"),
                "description":   m.get("uniprotDescription"),
                "organism":      m.get("organismScientificName"),
                "sequence":      m.get("sequence"),
                "length":        len(m.get("sequence") or ""),
                "mean_plddt":    m.get("globalMetricValue"),
                "frac_very_low": m.get("fractionPlddtVeryLow"),
                "frac_low":      m.get("fractionPlddtLow"),
                "frac_conf":     m.get("fractionPlddtConfident"),
                "frac_very_high": m.get("fractionPlddtVeryHigh"),
                "pdbUrl":        m.get("pdbUrl"),
                "cifUrl":        m.get("cifUrl"),
                "latestVersion": m.get("latestVersion"),
            }
    meta_file = SUB_DIRS["metadata"] / "monomer_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 元数据已保存到: {meta_file}")
    return meta_dict


# ============================================================
# Step 2: 单体下载（FIX-1 / FIX-2 / FIX-3 / FIX-6 核心区域）
# ============================================================
def download_file(url: str, dest: Path, session, retries: int = DOWNLOAD_RETRIES) -> bool:
    """
    带重试与错误打印的文件下载。
    FIX-6: 先写 .part 临时文件，成功后原子重命名，避免半截文件被误判为已完成。
    FIX-2: 所有失败路径都会打印原因，不再静默。
    """
    if not url:
        return False
    for attempt in range(retries):
        try:
            r = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                tmp.rename(dest)  # 原子替换
                return True
            elif r.status_code == 404:
                print(f"  [404] {url}")
                return False
            else:
                print(f"  [HTTP {r.status_code}] 尝试 {attempt + 1}: {url}")
        except Exception as e:
            print(f"  [下载异常 {attempt + 1}/{retries}] {url}: "
                  f"{type(e).__name__}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return False


def extract_plddt_from_pdb(pdb_path: Path):
    """
    FIX-4: 从 AlphaFold PDB 的 B-factor 列（CA 原子）提取残基级 pLDDT。
    AlphaFold 把 pLDDT 存在 B-factor 字段（列 61-66）。
    """
    plddt, seen = [], set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    res_key = line[22:27]  # chain + resSeq + icode，用于残基去重
                    if res_key not in seen:
                        seen.add(res_key)
                        plddt.append(round(float(line[60:66]), 2))
    except Exception as e:
        print(f"  [pLDDT 提取失败] {pdb_path.name}: {type(e).__name__}: {e}")
        return None
    return plddt if plddt else None


def download_monomer(acc: str, meta: dict, session) -> bool:
    """
    下载单个蛋白: PDB + CIF，并从 PDB 提取 pLDDT 存为 JSON。

    FIX-1（核心）: v1 在这里同时用了 entryId（未定义）和 entry_id（已定义），
    dict.get(key, default) 的 default 是立即求值的 f-string，
    导致每行都抛 NameError，被上层 except 静默吞掉 -> 0/3593 成功。
    现在统一只用 entry_id，且优先直接使用 API 返回的 URL（FIX-3: v6）。
    """
    if meta is None:
        return False

    entry_id = meta.get("entryId", f"AF-{acc}-F1")  # 唯一变量名，全文只用这一个

    # 优先使用 API 返回的 URL；缺失时才用 entry_id 构造（v6）
    pdb_url = meta.get("pdbUrl") or f"{AF_FILES_DIR}/{entry_id}-model_v6.pdb"
    cif_url = meta.get("cifUrl") or f"{AF_FILES_DIR}/{entry_id}-model_v6.cif"

    pdb_dest  = SUB_DIRS["monomer"] / f"{safe_filename(acc)}_pdb.pdb"
    cif_dest  = SUB_DIRS["monomer"] / f"{safe_filename(acc)}_cif.cif"
    lddt_dest = SUB_DIRS["monomer"] / f"{safe_filename(acc)}_plddt.json"

    # PDB 是必需品（断点续传：已存在且非空则跳过）
    pdb_ok = (pdb_dest.exists() and pdb_dest.stat().st_size > 0) \
             or download_file(pdb_url, pdb_dest, session)

    # CIF 尽力而为（失败不判整体失败，但会记录）
    if not (cif_dest.exists() and cif_dest.stat().st_size > 0):
        download_file(cif_url, cif_dest, session)  # 失败仅打印，不返回 False

    # FIX-4: 从 PDB 提取 pLDDT（有 PDB 且尚未提取时）
    if pdb_ok and not lddt_dest.exists():
        plddt = extract_plddt_from_pdb(pdb_dest)
        if plddt is not None:
            lddt_dest.write_text(json.dumps(plddt))

    return pdb_ok


def batch_download_monomers(meta_dict, session):
    """多线程批量下载单体结构"""
    hits = {k: v for k, v in meta_dict.items() if v is not None}
    print(f"\n[STEP] 批量下载 {len(hits)} 个蛋白的单体结构 (PDB/CIF/pLDDT) ...")
    SUB_DIRS["monomer"].mkdir(parents=True, exist_ok=True)

    ok_count, fail_list = 0, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_monomer, acc, meta, session): acc
                   for acc, meta in hits.items()}
        with tqdm(total=len(hits), desc="单体下载") as pbar:
            for future in as_completed(futures):
                acc = futures[future]
                try:
                    if future.result():
                        ok_count += 1
                    else:
                        fail_list.append(acc)
                except Exception as e:
                    # FIX-2（核心）: v1 这里是 except Exception: fail_list.append(acc)
                    # 什么都不打印，NameError 被完全掩盖。现在打印具体错误。
                    print(f"\n  [ERROR] {acc}: {type(e).__name__}: {e}")
                    fail_list.append(acc)
                pbar.update(1)

    print(f"[INFO] 单体下载成功: {ok_count} / {len(hits)}")
    if fail_list:
        fail_file = SUB_DIRS["logs"] / "monomer_download_failed.txt"
        fail_file.write_text("\n".join(fail_list))
        print(f"[INFO] 失败列表已保存到: {fail_file}")
        print("[INFO] 失败原因请向上翻终端输出（每个失败都打印了具体错误）")
    return ok_count, fail_list


# ============================================================
# Step 3（实验性）: 同源二聚体复合物下载
# ============================================================
def download_homodimer_complex(acc: str, session) -> bool:
    """
    尝试从 NVIDIA AF3 协作数据集下载同源二聚体。
    [注意] 此处 URL 为推测路径，命中率可能很低（可能 404）。
    失败是预期行为，请用本地 ColabFold 预测兜底。
    """
    first = acc[0]
    urls = [
        f"https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvidia/AF3/{first}/{acc}.cif.gz",
        f"https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvidia/{first}/{acc}.cif.gz",
    ]
    dest = SUB_DIRS["homodimer"] / f"{safe_filename(acc)}_{safe_filename(acc)}.cif.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for u in urls:
        if download_file(u, dest, session, retries=1):
            return True
    return False


def batch_download_homodimers(homo_pairs, session):
    accs = sorted({a for a, _ in homo_pairs})
    print(f"\n[STEP] [实验性] 尝试下载 {len(accs)} 个同源二聚体复合物 (NVIDIA AF3) ...")
    print("[WARN] 该数据源 URL 为推测路径，未命中属正常现象；未命中的对请走本地预测")
    ok = 0
    for acc in tqdm(accs, desc="同源二聚体"):
        if download_homodimer_complex(acc, session):
            ok += 1
        time.sleep(0.1)
    print(f"[INFO] 同源二聚体命中: {ok} / {len(accs)}")
    return ok


# ============================================================
# Step 4: FASTA 生成（供本地 ColabFold / AlphaFold-Multimer）
# ============================================================
def generate_pair_fasta(pairs, meta_dict):
    """
    生成多聚体输入 FASTA。ColabFold 格式: 链之间用 ':' 分隔。
    >{accA}--{accB}
    {seqA}:{seqB}
    """
    fa_hetero = SUB_DIRS["fasta"] / "hetero_pairs_for_multimer.fasta"
    fa_homo   = SUB_DIRS["fasta"] / "homodimers_for_multimer.fasta"

    n_h, skip_h = 0, 0
    with open(fa_hetero, "w") as f:
        for a, b in pairs:
            if a == b:
                continue
            ma, mb = meta_dict.get(a), meta_dict.get(b)
            sa = (ma or {}).get("sequence") if ma else None
            sb = (mb or {}).get("sequence") if mb else None
            if not sa or not sb:
                skip_h += 1
                continue
            f.write(f">{a}--{b}\n{sa}:{sb}\n")
            n_h += 1

    n_d, skip_d = 0, 0
    with open(fa_homo, "w") as f:
        for a, b in pairs:
            if a != b:
                continue
            m = meta_dict.get(a)
            s = (m or {}).get("sequence")
            if not s:
                skip_d += 1
                continue
            f.write(f">{a}--{a}\n{s}:{s}\n")
            n_d += 1

    print(f"\n[STEP] 生成预测用 FASTA ...")
    print(f"[INFO] 异源二聚体: {fa_hetero} ({n_h} 对，跳过 {skip_h} 对缺序列)")
    print(f"[INFO] 同源二聚体: {fa_homo} ({n_d} 对，跳过 {skip_d} 对缺序列)")
    print(f"[INFO] 本地运行示例:")
    print(f"    colabfold_batch {fa_hetero} af_output/fasta_for_prediction/results/")


# ============================================================
# Step 5: 汇总
# ============================================================
def write_summary(pairs, meta_dict):
    path = OUTPUT_ROOT / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_a", "pair_b", "type",
                    "a_in_afdb", "b_in_afdb",
                    "a_pdb_file", "b_pdb_file",
                    "a_mean_plddt", "b_mean_plddt"])
        for a, b in pairs:
            ma, mb = meta_dict.get(a), meta_dict.get(b)
            w.writerow([
                a, b,
                "homodimer" if a == b else "heterodimer",
                ma is not None, mb is not None,
                (SUB_DIRS["monomer"] / f"{safe_filename(a)}_pdb.pdb").exists(),
                (SUB_DIRS["monomer"] / f"{safe_filename(b)}_pdb.pdb").exists(),
                (ma or {}).get("globalMetricValue", ""),
                (mb or {}).get("globalMetricValue", ""),
            ])
    print(f"\n[INFO] 汇总表: {path}")


# ============================================================
# 输入解析
# ============================================================
def parse_pair_file(path: Path):
    """解析蛋白对文件（空白符分隔的 accession 流，两两成对）"""
    with open(path) as f:
        tokens = f.read().split()
    n_raw = len(tokens)
    if n_raw % 2 != 0:
        print(f"[WARN] token 数为奇数 ({n_raw})，丢弃最后一个: {tokens[-1]}")
        tokens = tokens[:-1]
    raw_pairs = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]

    seen, uniq_pairs = set(), []
    for p in raw_pairs:
        if p not in seen:
            seen.add(p)
            uniq_pairs.append(p)
    unique_accs = sorted({a for p in uniq_pairs for a in p})

    print(f"[INFO] 原始 token 数: {n_raw}")
    print(f"[INFO] 解析得到蛋白对: {len(raw_pairs)} 个")
    print(f"[INFO] 去重后蛋白对:  {len(uniq_pairs)} 个")
    print(f"[INFO] 唯一 accession: {len(unique_accs)} 个")
    return uniq_pairs, unique_accs


def classify_pairs(pairs):
    homo = [p for p in pairs if p[0] == p[1]]
    hetero = [p for p in pairs if p[0] != p[1]]
    print(f"[INFO] 同源二聚体 (A==A): {len(homo)} 对")
    print(f"[INFO] 异源二聚体 (A!=B): {len(hetero)} 对")
    return homo, hetero


# ============================================================
# 主流程
# ============================================================
def main():
    # FIX-5: global 声明必须在任何使用之前
    # （v1 把它放在 argparse 之后，而 argparse 的 default=MAX_WORKERS 已构成"使用"，报 SyntaxError）
    global MAX_WORKERS, SLEEP_BETWEEN

    parser = argparse.ArgumentParser(
        description="批量查询/下载 AlphaFold 结构（单体+复合物）v2.0 修复版")
    parser.add_argument("input", type=str,
                        help="蛋白对列表文件路径，如 Intra0_pos_rr.txt")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                        help=f"并发线程数（默认 {MAX_WORKERS}）")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN,
                        help=f"API 查询间隔秒数（默认 {SLEEP_BETWEEN}）")
    parser.add_argument("--skip-complex", action="store_true",
                        help="跳过实验性的同源二聚体下载（推荐：该数据源 URL 未验证）")
    args = parser.parse_args()

    MAX_WORKERS = args.max_workers
    SLEEP_BETWEEN = args.sleep

    print(f"[INFO] 配置: workers={MAX_WORKERS}, sleep={SLEEP_BETWEEN}s")

    setup_dirs()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    pairs, unique_accs = parse_pair_file(input_path)
    homo_pairs, hetero_pairs = classify_pairs(pairs)

    # ---- Step 1: 查询元数据 ----
    meta_dict = batch_query_monomers(unique_accs, SESSION)

    # ---- Step 2: 下载单体 ----
    ok_count, fail_list = batch_download_monomers(meta_dict, SESSION)

    # ---- Step 3: 同源二聚体（实验性，可跳过） ----
    if not args.skip_complex and homo_pairs:
        batch_download_homodimers(homo_pairs, SESSION)
    elif args.skip_complex:
        print("\n[INFO] 已按 --skip-complex 跳过实验性复合物下载")

    # ---- Step 4: 生成 FASTA ----
    generate_pair_fasta(pairs, meta_dict)

    # ---- Step 5: 汇总 ----
    write_summary(pairs, meta_dict)

    # ---- 结束统计 ----
    print(f"\n{'=' * 60}")
    print(f" 全部完成")
    print(f"{'=' * 60}")
    hits = sum(1 for v in meta_dict.values() if v is not None)
    print(f" 元数据命中:        {hits} / {len(unique_accs)}")
    print(f" 单体下载成功:      {ok_count} / {hits}")
    if fail_list:
        print(f" 单体下载失败:      {len(fail_list)} (见 logs/monomer_download_failed.txt)")
    print(f" 输出目录:          {OUTPUT_ROOT.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
