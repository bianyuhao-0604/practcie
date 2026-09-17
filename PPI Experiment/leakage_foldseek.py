#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leakage_foldseek.py — 结构层面泄漏评估与清洁切分
==========================================================
四部分:
  Part 1: 跨split身份检查（免费, 秒级, 无需foldseek）
  Part 2: 准备结构目录 + 运行 Foldseek all-vs-all（需安装foldseek, ~20分钟）
  Part 3: 解析比对结果, 统计 test↔train 结构泄漏
  Part 4: 选项A清洁切分 — 从 train 删除与 test 相似(lDDT≥阈值)的蛋白

依赖: foldseek (conda install -c bioconda foldseek)

运行:
  python leakage_foldseek.py --skip-foldseek          # 只跑Part1(立即)
  python leakage_foldseek.py --lddt-thr 55            # 完整流程
  python leakage_foldseek.py --lddt-thr 70            # 宽松版
"""
import os, sys, json, shutil, argparse, subprocess
from pathlib import Path
from collections import defaultdict

AF_OUTPUT  = Path("af_output")
MONO_DIR   = AF_OUTPUT / "monomers"
META_FILE  = AF_OUTPUT / "metadata" / "monomer_metadata.json"

WORK_DIR   = Path("leakage_work")
STRUCT_DIR = WORK_DIR / "structures"
FS_DB      = WORK_DIR / "fs_db"
FS_RESULT  = WORK_DIR / "leak_result.tsv"

SPLIT_FILES = {
    "train": ["Intra1_pos_rr.txt", "Intra1_neg_rr.txt"],
    "val":   ["Intra0_pos_rr.txt", "Intra0_neg_rr.txt"],
    "test":  ["Intra2_pos_rr.txt", "Intra2_neg_rr.txt"],
}


def parse_pair_file(p: Path):
    toks = p.read_text().split()
    if len(toks) % 2:
        toks = toks[:-1]
    return [(toks[i], toks[i+1]) for i in range(0, len(toks), 2)]


# ============================================================
# Part 1: 跨 split 身份检查（免费）
# ============================================================
def check_cross_split(meta):
    print("\n" + "=" * 60)
    print(" Part 1: 跨split身份检查")
    print("=" * 60)
    cross = {a: m["splits"] for a, m in meta.items()
             if m and m.get("splits") and len(m["splits"]) > 1}
    print(f"[结果] 跨split蛋白: {len(cross)} 个")
    if not cross:
        print("       ✓ 官方切分是蛋白级别的, 无身份泄漏")
    else:
        # 按组合分组统计
        combo = defaultdict(list)
        for a, sp in cross.items():
            combo["+".join(sp)].append(a)
        for c, accs in sorted(combo.items()):
            print(f"       {c}: {len(accs)} 个 (如 {accs[:3]})")
    return cross


# ============================================================
# Part 2: Foldseek all-vs-all
# ============================================================
def run_foldseek(threads, lddt_thr):
    print("\n" + "=" * 60)
    print(" Part 2: Foldseek all-vs-all 结构比对")
    print("=" * 60)

    # 检查 foldseek
    try:
        v = subprocess.run(["foldseek", "--version"], capture_output=True,
                           text=True, timeout=30)
        print(f"[INFO] foldseek: {v.stdout.strip() or v.stderr.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[ERROR] foldseek 未安装!")
        print("  安装: conda install -c bioconda foldseek")
        print("  或:   https://github.com/steineggerlab/foldseek/releases")
        sys.exit(1)

    # 结构目录: 硬链接(省空间)或复制
    if STRUCT_DIR.exists() and any(STRUCT_DIR.glob("*.pdb")):
        n = len(list(STRUCT_DIR.glob("*.pdb")))
        print(f"[INFO] 结构目录已存在: {n} 个, 跳过准备")
    else:
        STRUCT_DIR.mkdir(parents=True, exist_ok=True)
        pdb_files = sorted(MONO_DIR.glob("*_pdb.pdb"))
        print(f"[INFO] 准备结构: {len(pdb_files)} 个 (硬链接优先)")
        n_link = n_copy = 0
        for pf in pdb_files:
            acc = pf.name[:-len("_pdb.pdb")]
            dst = STRUCT_DIR / f"{acc}.pdb"
            if dst.exists():
                continue
            try:
                os.link(pf, dst)      # Windows NTFS / Linux 均支持
                n_link += 1
            except OSError:
                shutil.copy2(pf, dst)
                n_copy += 1
        print(f"[INFO] 完成: 硬链接 {n_link} | 复制 {n_copy}")

    # createdb
    db = str(FS_DB)
    if not Path(db + ".0").exists():
        print("[INFO] foldseek createdb ...")
        r = subprocess.run(["foldseek", "createdb", str(STRUCT_DIR), db,
                            "--threads", str(threads)], capture_output=True)
        if r.returncode != 0:
            print(f"[ERROR] createdb 失败:\n{r.stderr.decode(errors='ignore')[-500:]}")
            sys.exit(1)

    # search (自比对)
    aln, tmp = db + "_aln", db + "_tmp"
    if not Path(aln).exists():
        print("[INFO] foldseek search (all-vs-all, 约10-30分钟) ...")
        r = subprocess.run(["foldseek", "search", db, db, aln, tmp,
                            "--threads", str(threads)], capture_output=True)
        if r.returncode != 0:
            print(f"[ERROR] search 失败:\n{r.stderr.decode(errors='ignore')[-500:]}")
            sys.exit(1)

    # createtsv
    if not FS_RESULT.exists():
        print("[INFO] foldseek createtsv ...")
        r = subprocess.run(["foldseek", "createtsv", db, db, aln, str(FS_RESULT),
                            "--format-output", "query,target,lddt,alnlen",
                            "--threads", str(threads)], capture_output=True)
        if r.returncode != 0:
            print(f"[ERROR] createtsv 失败:\n{r.stderr.decode(errors='ignore')[-500:]}")
            sys.exit(1)

    n_lines = sum(1 for _ in FS_RESULT.open())
    print(f"[INFO] 比对结果: {n_lines} 行 → {FS_RESULT}")


# ============================================================
# Part 3: 解析 + 泄漏统计
# ============================================================
def analyze_leakage(meta, lddt_thr):
    print("\n" + "=" * 60)
    print(f" Part 3: 泄漏统计 (lDDT ≥ {lddt_thr})")
    print("=" * 60)

    def get_splits(a):
        m = meta.get(a)
        return set(m["splits"]) if m and m.get("splits") else set()

    sim = defaultdict(set)
    n_edges = 0
    with FS_RESULT.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q, t = parts[0], parts[1]
            try:
                lddt = float(parts[2])
            except ValueError:
                continue
            if q == t:
                continue
            if lddt >= lddt_thr:
                sim[q].add(t)
                sim[t].add(q)
                n_edges += 1
    print(f"[INFO] 相似边(去自比对): {n_edges}")

    # test ↔ train 泄漏（核心）
    test_leak = defaultdict(set)
    for q, ts in sim.items():
        if "test" in get_splits(q):
            for t in ts:
                if "train" in get_splits(t):
                    test_leak[q].add(t)

    n_test = sum(1 for m in meta.values() if m and "test" in (m.get("splits") or []))
    print(f"[泄漏] test蛋白与train结构相似: {len(test_leak)}/{n_test} "
          f"({100*len(test_leak)/max(n_test,1):.1f}%)")

    # test ↔ val（次要, 影响模型选择）
    test_val = defaultdict(set)
    for q, ts in sim.items():
        if "test" in get_splits(q):
            for t in ts:
                if "val" in get_splits(t) and "train" not in get_splits(t):
                    test_val[q].add(t)
    print(f"[参考] test蛋白与val(纯val)相似: {len(test_val)} 个")

    if test_leak:
        # 相似物数量分布
        counts = sorted((len(ts) for ts in test_leak.values()), reverse=True)
        print(f"       每个泄漏test蛋白的train相似物数: "
              f"max={counts[0]}, median={counts[len(counts)//2]}")
        # 展示前5个
        for q in sorted(test_leak, key=lambda x: -len(test_leak[x]))[:5]:
            print(f"       {q} ← {len(test_leak[q])} 个train相似物")
    return test_leak, sim


# ============================================================
# Part 4: 选项A清洁切分（删除train侧泄漏蛋白）
# ============================================================
def make_structural_clean(meta, cross, test_leak):
    print("\n" + "=" * 60)
    print(" Part 4: 清洁切分生成（选项A: 删除train侧）")
    print("=" * 60)

    banned = set()
    for q, ts in test_leak.items():
        banned |= ts                       # 与test相似的train蛋白
    for a, sp in cross.items():
        if "test" in sp:
            banned.add(a)                  # 跨split且出现在test
    print(f"[INFO] train侧删除蛋白: {len(banned)} 个")

    report = {"banned_train": sorted(banned), "files": {}}
    for fn in SPLIT_FILES["train"]:
        fp = Path(fn)
        if not fp.exists():
            continue
        pairs = parse_pair_file(fp)
        kept = [p for p in pairs if p[0] not in banned and p[1] not in banned]
        out = fp.with_name(fp.stem + "_structclean.txt")
        out.write_text(" ".join(f"{a} {b}" for a, b in kept))
        report["files"][fn] = {"before": len(pairs), "after": len(kept)}
        print(f"  {fn}: {len(pairs)} → {len(kept)} 对 "
              f"({100*len(kept)/max(len(pairs),1):.1f}% 保留) → {out.name}")

    with open(WORK_DIR / "structural_leak.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] 报告: {WORK_DIR}/structural_leak.json")
    return report


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lddt-thr", type=float, default=55.0,
                    help="结构相似阈值(0-100, 55=PINDER严格版, 70=宽松版)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-foldseek", action="store_true",
                    help="只跑Part1跨split检查(免费)")
    args = ap.parse_args()

    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)

    cross = check_cross_split(meta)

    if args.skip_foldseek:
        print("\n[DONE] --skip-foldseek 模式结束。完整评估请去掉该参数重跑")
        return

    run_foldseek(args.threads, args.lddt_thr)
    test_leak, sim = analyze_leakage(meta, args.lddt_thr)
    make_structural_clean(meta, cross, test_leak)

    print("\n[解读指南]")
    print("  泄漏率 <5%  → 官方切分基本干净, 只需AF截止检查")
    print("  泄漏率 5-15% → 建议用 _structclean 切分跑全量实验")
    print("  泄漏率 >15% → 必须报告双套结果(原始/清洁)")


if __name__ == "__main__":
    main()
