#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_query_alphafold.py
========================
根据蛋白对列表批量查询/下载 AlphaFold 结构：
  1) 解析 Intra0_pos_rr.txt（两个连续 UniProt accession = 一个蛋白对）
  2) 去重、分类（同源二聚体 A==A / 异源二聚体 A!=B）
  3) 单体结构批量下载（PDB + mmCIF + 置信度JSON）
  4) 同源二聚体复合物批量下载（AlphaFold DB 已收录）
  5) 异源二聚体复合物：下载 AlphaFold DB NVIDIA 数据集 accession 列表并比对
  6) 未命中的异源蛋白对 → 生成 FASTA 文件，供 ColabFold/AlphaFold-Multimer 本地批量预测
  7) 输出汇总 CSV

依赖：pip install requests tqdm
运行：python batch_query_alphafold.py Intra0_pos_rr.txt
"""

import os
import sys
import csv
import json
import time
import shutil
import hashlib
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

import requests
from tqdm import tqdm

# ============================================================
# 0. 全局配置
# ============================================================
AF_BASE          = "https://alphafold.ebi.ac.uk"
AF_API_SINGLE    = f"{AF_BASE}/api/prediction/{{acc}}"          # 单体元数据 API
AF_FILES_DIR     = f"{AF_BASE}/files"                           # 直接拼接下载 URL
AF_MODEL_VERSION = "v4"                                         # 当前默认模型版本
AF_FTP_NVIDIA    = "https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvidia/"

REQUEST_TIMEOUT  = 30
SLEEP_BETWEEN    = 0.2                                          # 礼貌限速
MAX_WORKERS      = 4                                            # 并发线程数

OUTPUT_ROOT      = Path("af_output")
SUB_DIRS = {
    "monomer":     OUTPUT_ROOT / "monomers",
    "homodimer":   OUTPUT_ROOT / "homodimers",
    "heterodimer": OUTPUT_ROOT / "heterodimers",
    "fasta":       OUTPUT_ROOT / "fasta_for_colabfold",
    "meta":        OUTPUT_ROOT / "metadata",
    "logs":        OUTPUT_ROOT / "logs",
}

# ============================================================
# 1. 工具函数
# ============================================================
def setup_dirs():
    """创建输出目录结构"""
    for d in SUB_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def parse_pair_file(path: Path):
    """
    解析蛋白对文件：空格/换行分隔，两个连续 accession 为一个蛋白对。
    返回：去重后的蛋白对列表 [(A, B), ...] 和唯一 accession 集合
    """
    with open(path, "r", encoding="utf-8") as f:
        tokens = f.read().split()

    if len(tokens) % 2 != 0:
        print(f"[WARN] 文件 token 数为奇数 ({len(tokens)})，忽略最后一个孤立 ID: {tokens[-1]}")
        tokens = tokens[:-1]

    pairs = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]
    # 去重（保持顺序）
    seen = set()
    uniq_pairs = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            uniq_pairs.append(p)

    unique_accs = sorted({a for pair in uniq_pairs for a in pair})
    print(f"[INFO] 原始 token 数: {len(tokens)}")
    print(f"[INFO] 解析得到蛋白对: {len(pairs)} 个")
    print(f"[INFO] 去重后蛋白对:  {len(uniq_pairs)} 个")
    print(f"[INFO] 唯一 accession: {len(unique_accs)} 个")
    return uniq_pairs, unique_accs


def classify_pairs(pairs):
    """分类：同源二聚体 / 异源二聚体"""
    homo = [p for p in pairs if p[0] == p[1]]
    hetero = [p for p in pairs if p[0] != p[1]]
    print(f"[INFO] 同源二聚体 (A==A): {len(homo)} 对")
    print(f"[INFO] 异源二聚体 (A!=B): {len(hetero)} 对")
    return homo, hetero


def safe_filename(name: str) -> str:
    """防止 accession 中有非法字符"""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """带重试的下载"""
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
                return True
            elif r.status_code == 404:
                return False
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
    return False


# ============================================================
# 2. AlphaFold DB API 查询
# ============================================================
def query_af_metadata(acc: str, session: requests.Session):
    """
    查询单体元数据，返回 dict 或 None（404/未收录）
    """
    url = AF_API_SINGLE.format(acc=acc)
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
        return None
    except Exception:
        return None


def batch_query_monomers(unique_accs, session):
    """批量查询单体元数据（带进度条和限速）"""
    print(f"\n[STEP] 批量查询 {len(unique_accs)} 个 accession 的单体元数据 ...")
    results = OrderedDict()
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(query_af_metadata, acc, session): acc for acc in unique_accs}
        with tqdm(total=len(unique_accs), desc="API 查询") as pbar:
            for future in as_completed(futures):
                acc = futures[future]
                try:
                    meta = future.result()
                    results[acc] = meta
                    if meta is None:
                        errors.append(acc)
                except Exception as e:
                    results[acc] = None
                    errors.append(acc)
                pbar.update(1)
                time.sleep(SLEEP_BETWEEN / MAX_WORKERS)

    print(f"[INFO] 命中: {len(unique_accs) - len(errors)} 个")
    print(f"[INFO] 未命中 (不在 AlphaFold DB): {len(errors)} 个")
    if errors:
        with open(SUB_DIRS["logs"] / "not_in_afdb.txt", "w") as f:
            f.write("\n".join(errors))
        print(f"[INFO] 未命中列表已保存到: {SUB_DIRS['logs'] / 'not_in_afdb.txt'}")

    # 保存元数据 JSON
    with open(SUB_DIRS["meta"] / "monomer_metadata.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


# ============================================================
# 3. 下载单体结构
# ============================================================
def download_monomer(acc: str, meta: dict, session) -> bool:
    """下载单个蛋白的单体 PDB + mmCIF + 置信度 JSON"""
    if meta is None:
        return False

    entry_id  = meta.get("entryId", f"AF-{acc}-F1")
    pdb_url   = meta.get("pdbUrl",  f"{AF_FILES_DIR}/{entry_id}-model_{AF_MODEL_VERSION}.pdb")
    cif_url   = meta.get("cifUrl",  f"{AF_FILES_DIR}/{entry_id}.cif")
    json_url  = meta.get("summaryConfidence",  f"{AF_FILES_DIR}/{entry_id}-summary_confidences_{AF_MODEL_VERSION}.json")

    ok = True
    for url, suffix in [(pdb_url, ".pdb"), (cif_url, ".cif"), (json_url, "_conf.json")]:
        dest = SUB_DIRS["monomer"] / f"{safe_filename(acc)}{suffix}"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if not download_file(url, dest):
            ok = False
    return ok


def batch_download_monomers(meta_dict, session):
    """并发下载所有命中的单体结构"""
    hits = {k: v for k, v in meta_dict.items() if v is not None}
    print(f"\n[STEP] 批量下载 {len(hits)} 个蛋白的单体结构 (PDB/CIF/confidence) ...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_monomer, acc, meta, session): acc
                   for acc, meta in hits.items()}
        ok_count, fail_list = 0, []
        with tqdm(total=len(hits), desc="单体下载") as pbar:
            for future in as_completed(futures):
                acc = futures[future]
                try:
                    if future.result():
                        ok_count += 1
                    else:
                        fail_list.append(acc)
                except Exception:
                    fail_list.append(acc)
                pbar.update(1)

    print(f"[INFO] 单体下载成功: {ok_count} / {len(hits)}")
    if fail_list:
        with open(SUB_DIRS["logs"] / "monomer_download_failed.txt", "w") as f:
            f.write("\n".join(fail_list))
    return hits


# ============================================================
# 4. 同源二聚体复合物下载
# ============================================================
def download_homodimer(acc: str, meta: dict, session) -> bool:
    """
    同源二聚体：AlphaFold DB 中同一 accession 的 entry 包含其homodimer构象。
    实际上 AlphaFold DB 对人类蛋白的条目是单体预测；
    但 NVIDIA 数据集 / AF3 复合物数据集里有同源二聚体。
    这里走 NVIDIA FTP 下载路径（见下方 heterodimer 检索逻辑），
    或者通过 AF3 API（如果用户有凭据）。
    简化处理：单体结构 + 用户自行用 ColabFold 跑同源二聚体。
    但我们先尝试 NVIDIA FTP 下载对应同源二聚体文件。
    """
    # NVIDIA 数据集文件命名模式（v2.0+）：
    # https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvidia/AF3/{first_char}/{acc}.cif.gz
    acc = acc.upper()
    ftp_base = AF_FTP_NVIDIA + "AF3"
    url = f"{ftp_base}/{acc[0]}/{acc}.cif.gz"
    dest = SUB_DIRS["homodimer"] / f"{safe_filename(acc)}_homodimer.cif.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return True
    return download_file(url, dest)


def batch_download_homodimers(homo_pairs, session):
    """批量下载同源二聚体复合物（从 NVIDIA FTP）"""
    unique_homo = sorted({p[0] for p in homo_pairs})
    print(f"\n[STEP] 批量下载 {len(unique_homo)} 个同源二聚体复合物 (NVIDIA FTP) ...")

    ok_count, fail_list = 0, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_homodimer, acc, None, session): acc
                   for acc in unique_homo}
        with tqdm(total=len(unique_homo), desc="同源二聚体下载") as pbar:
            for future in as_completed(futures):
                acc = futures[future]
                try:
                    if future.result():
                        ok_count += 1
                    else:
                        fail_list.append(acc)
                except Exception:
                    fail_list.append(acc)
                pbar.update(1)

    print(f"[INFO] 同源二聚体命中: {ok_count} / {len(unique_homo)}")
    if fail_list:
        with open(SUB_DIRS["logs"] / "homodimer_not_found.txt", "w") as f:
            f.write("\n".join(fail_list))
        print(f"[INFO] 未命中的同源二聚体需本地预测，列表: {SUB_DIRS['logs'] / 'homodimer_not_found.txt'}")


# ============================================================
# 5. 异源二聚体现成复合物检索
# ============================================================
def download_heterodimer_index(session):
    """
    下载 NVIDIA 数据集的 heterodimer accession 列表（如果存在索引文件）
    常见格式：
      - complexes_metadata.csv / heterodimers.tsv
    如果找不到索引文件，返回 None，用户需自己下载 FTP 目录扫描
    """
    # 尝试常见索引文件名
    candidate_names = [
        "heterodimers.tsv",
        "heterodimer_list.txt",
        "complexes_metadata.tsv",
        "pred Complex_index.tsv",
    ]
    for name in candidate_names:
        url = AF_FTP_NVIDIA + name
        dest = SUB_DIRS["meta"] / name
        if download_file(url, dest):
            print(f"[INFO] 找到异源二聚体索引: {name}")
            return dest

    # 尝试下载目录列表（FTP HTML 索引）
    r = session.get(AF_FTP_NVIDIA, timeout=60)
    if r.status_code == 200:
        with open(SUB_DIRS["meta"] / "ftp_index.html", "w") as f:
            f.write(r.text)
        print(f"[INFO] FTP 目录索引已保存到: {SUB_DIRS['meta'] / 'ftp_index.html'}")
        print(f"[INFO] 请人工查看该文件，确认异源二聚体列表文件名")
        return None

    print(f"[WARN] 无法访问 NVIDIA FTP，跳过异源二聚体现成复合物检索")
    return None


def check_heterodimer_in_db(hetero_pairs, index_file, session):
    """
    检查异源蛋白对是否已在 AlphaFold DB 复合物数据集中。
    如果索引可用，返回命中的蛋白对集合 + 未命中集合；
    否则，全部返回为"需本地预测"。
    """
    if index_file is None or not index_file.exists():
        print(f"[INFO] 无索引可用，全部 {len(hetero_pairs)} 对异源蛋白需本地预测")
        return set(), set(hetero_pairs)

    # 读索引
    db_keys = set()
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                a, b = parts[0].upper(), parts[1].upper()
                # 用 frozenset 避免顺序差异
                db_keys.add(frozenset([a, b]))

    hits, missing = set(), set()
    for pair in hetero_pairs:
        key = frozenset(pair)
        if key in db_keys:
            hits.add(pair)
        else:
            missing.add(pair)

    print(f"[INFO] 异源二聚体已在 AlphaFold DB: {len(hits)} 对")
    print(f"[INFO] 异源二聚体需本地预测:     {len(missing)} 对")
    return hits, missing


def download_heterodimer_pair(acc_a: str, acc_b: str, session) -> bool:
    """
    下载已收录的异源二聚体复合物。
    NVIDIA FTP 命名模式（推测）：{A}_{B}.cif.gz 或 {B}_{A}.cif.gz
    """
    a, b = sorted([acc_a.upper(), acc_b.upper()])
    ftp_base = AF_FTP_NVIDIA + "AF3"
    candidates = [
        f"{ftp_base}/{a[0]}/{a}_{b}.cif.gz",
        f"{ftp_base}/{b[0]}/{a}_{b}.cif.gz",
        f"{ftp_base}/{a[0]}/{b}_{a}.cif.gz",
        f"{ftp_base}/{b[0]}/{b}_{a}.cif.gz",
    ]
    dest = SUB_DIRS["heterodimer"] / f"{safe_filename(a)}_{safe_filename(b)}.cif.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return True

    for url in candidates:
        if download_file(url, dest):
            return True
    return False


def batch_download_heterodimers(hetero_hits, session):
    """批量下载已收录的异源二聚体复合物"""
    if not hetero_hits:
        return
    print(f"\n[STEP] 批量下载 {len(hetero_hits)} 个已收录的异源二聚体复合物 ...")
    ok_count, fail_list = 0, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_heterodimer_pair, a, b, session): (a, b)
                   for a, b in hetero_hits}
        with tqdm(total=len(hetero_hits), desc="异源二聚体下载") as pbar:
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    if future.result():
                        ok_count += 1
                    else:
                        fail_list.append(pair)
                except Exception:
                    fail_list.append(pair)
                pbar.update(1)
    print(f"[INFO] 异源二聚体下载成功: {ok_count} / {len(hetero_hits)}")
    if fail_list:
        with open(SUB_DIRS["logs"] / "heterodimer_download_failed.txt", "w") as f:
            for a, b in fail_list:
                f.write(f"{a}_{b}\n")


# ============================================================
# 6. 生成 FASTA 供 ColabFold 批量预测
# ============================================================
def fetch_sequence(acc: str, session) -> str:
    """从 UniProt REST API 拉取蛋白序列"""
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            seq = "".join(lines[1:])  # 去掉 >header 行
            return seq
    except Exception:
        pass
    return ""


def generate_fasta_for_prediction(pairs_to_predict, session):
    """
    对需要本地预测的蛋白对（同源未命中 + 异源未命中），
    从 UniProt 拉取序列并生成 FASTA 文件供 ColabFold 批量使用。
    ColabFold 格式：一条 FASTA 记录包含两条序列，用":"分隔
    """
    print(f"\n[STEP] 为 {len(pairs_to_predict)} 个蛋白对生成 FASTA (ColabFold 格式) ...")

    # 缓存序列
    seq_cache = {}

    def get_seq(acc):
        if acc not in seq_cache:
            seq_cache[acc] = fetch_sequence(acc, session)
            time.sleep(0.1)
        return seq_cache[acc]

    fa_path = SUB_DIRS["fasta"] / "pairs_for_prediction.fasta"
    with open(fa_path, "w") as f:
        for idx, (a, b) in enumerate(tqdm(pairs_to_predict, desc="生成FASTA"), 1):
            seq_a = get_seq(a)
            seq_b = get_seq(b)
            if not seq_a or not seq_b:
                continue
            header = f">{safe_filename(a)}_{safe_filename(b)}|pair{idx}"
            # ColabFold 多链格式：序列用 ":" 分隔
            seq_combined = f"{seq_a}:{seq_b}"
            f.write(f"{header}\n{seq_combined}\n")

    print(f"[INFO] FASTA 文件已生成: {fa_path}")
    print(f"[INFO] 可用命令批量预测:")
    print(f"    colabfold_batch --model-type alphafold2_multimer_v3 \\")
    print(f"                    {fa_path} af_output/colabfold_results/")
    return fa_path


# ============================================================
# 7. 汇总输出
# ============================================================
def write_summary_csv(pairs, meta_dict, homo_hits, hetero_hits, hetero_missing):
    """生成最终汇总 CSV"""
    csv_path = OUTPUT_ROOT / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "acc_A", "acc_B", "type",
            "A_in_afdb", "B_in_afdb",
            "complex_source", "complex_file",
            "needs_local_prediction"
        ])

        for i, (a, b) in enumerate(pairs, 1):
            pair_id = f"P{i:05d}"
            a_in = "yes" if meta_dict.get(a) is not None else "no"
            b_in = "yes" if meta_dict.get(b) is not None else "no"

            if a == b:
                ptype = "homodimer"
                if (a, a) in homo_hits or frozenset([a]) in {frozenset([x]) for x in homo_hits}:
                    src = "AlphaFold_DB"
                    cfile = f"{safe_filename(a)}_homodimer.cif.gz"
                    need_local = "no"
                else:
                    src = "Local_prediction"
                    cfile = ""
                    need_local = "yes"
            else:
                ptype = "heterodimer"
                if (a, b) in hetero_hits or frozenset([a, b]) in {frozenset(p) for p in hetero_hits}:
                    src = "AlphaFold_DB"
                    sorted_pair = sorted([a, b])
                    cfile = f"{safe_filename(sorted_pair[0])}_{safe_filename(sorted_pair[1])}.cif.gz"
                    need_local = "no"
                else:
                    src = "Local_prediction"
                    cfile = ""
                    need_local = "yes"

            writer.writerow([pair_id, a, b, ptype, a_in, b_in, src, cfile, need_local])

    print(f"\n[INFO] 汇总表已生成: {csv_path}")
    return csv_path


# ============================================================
# 8. 主函数
# ============================================================
def main():
    global MAX_WORKERS, SLEEP_BETWEEN   # ← 移到这里，必须放在所有使用之前

    parser = argparse.ArgumentParser(description="批量查询/下载 AlphaFold 结构（单体+复合物）")
    parser.add_argument("input", type=str, help="蛋白对列表文件路径，如 Intra0_pos_rr.txt")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="并发线程数")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN, help="请求间隔秒数")
    parser.add_argument("--skip-hetero-index", action="store_true",
                        help="跳过下载异源二聚体索引（加快启动）")
    args = parser.parse_args()

    MAX_WORKERS = args.max_workers
    SLEEP_BETWEEN = args.sleep
    # ... 后面其余代码保持不变



    setup_dirs()
    session = requests.Session()
    session.headers.update({"User-Agent": "BatchAFQuery/1.0"})

    # ---- Step 1: 解析蛋白对 ----
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    pairs, unique_accs = parse_pair_file(input_path)
    homo_pairs, hetero_pairs = classify_pairs(pairs)

    # ---- Step 2: 批量查询单体元数据 ----
    meta_dict = batch_query_monomers(unique_accs, session)

    # ---- Step 3: 批量下载单体结构 ----
    batch_download_monomers(meta_dict, session)

    # ---- Step 4: 同源二聚体 ----
    # 尝试从 NVIDIA FTP 下载同源二聚体复合物
    batch_download_homodimers(homo_pairs, session)
    # 检查哪些同源二聚体下载成功
    homo_hits = set()
    homo_fail = set()
    for (a, _) in homo_pairs:
        dest = SUB_DIRS["homodimer"] / f"{safe_filename(a)}_homodimer.cif.gz"
        if dest.exists() and dest.stat().st_size > 0:
            homo_hits.add((a, a))
        else:
            homo_fail.add((a, a))

    # ---- Step 5: 异源二聚体 ----
    hetero_hits, hetero_missing = set(), set(hetero_pairs)
    if not args.skip_hetero_index:
        index_file = download_heterodimer_index(session)
        if index_file:
            hetero_hits, hetero_missing = check_heterodimer_in_db(hetero_pairs, index_file, session)
            batch_download_heterodimers(hetero_hits, session)

    # ---- Step 6: 生成 FASTA 供本地预测 ----
    pairs_to_predict = list(homo_fail) + list(hetero_missing)
    if pairs_to_predict:
        fasta_path = generate_fasta_for_prediction(pairs_to_predict, session)
        # 自动生成 ColabFold 运行脚本
        script_path = OUTPUT_ROOT / "run_colabfold.sh"
        with open(script_path, "w") as f:
            f.write(f"""#!/bin/bash
# ColabFold 批量预测脚本（自动生成）
# 运行前请确保已安装 colabfold-batch

colabfold_batch \\
    --model-type alphafold2_multimer_v3 \\
    --num-recycle 3 \\
    --num-models 5 \\
    {fasta_path} \\
    af_output/colabfold_results/

echo "预测完成！结果在 af_output/colabfold_results/"
echo "用 ipTM/pDockQ2 过滤可信互作（建议 ipTM>=0.6 或 pDockQ2>=0.23）"
""")
        os.chmod(script_path, 0o755)
        print(f"[INFO] ColabFold 运行脚本已生成: {script_path}")

    # ---- Step 7: 汇总 ----
    write_summary_csv(pairs, meta_dict, homo_hits, hetero_hits, hetero_missing)

    # ---- 最终统计 ----
    print("\n" + "=" * 60)
    print(" 最终统计")
    print("=" * 60)
    print(f"  蛋白对总数:              {len(pairs)}")
    print(f"  同源二聚体:              {len(homo_pairs)}")
    print(f"  异源二聚体:              {len(hetero_pairs)}")
    print(f"  唯一 accession:          {len(unique_accs)}")
    print(f"  单体在 AlphaFold DB:     {sum(1 for v in meta_dict.values() if v)}")
    print(f"  同源二聚体已收录:        {len(homo_hits)}")
    print(f"  同源二聚体需本地预测:    {len(homo_fail)}")
    print(f"  异源二聚体已收录:        {len(hetero_hits)}")
    print(f"  异源二聚体需本地预测:    {len(hetero_missing)}")
    print("=" * 60)
    print(f"\n所有输出文件保存在: {OUTPUT_ROOT.absolute()}/")


if __name__ == "__main__":
    main()
