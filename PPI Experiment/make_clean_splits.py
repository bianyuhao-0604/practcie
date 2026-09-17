#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_clean_splits.py — 汇总结构泄漏 + AF泄漏, 生成最终清洁切分
==========================================================
输入:
  leakage_work/structural_leak.json   (来自①leakage_foldseek.py)
  leakage_work/af_leak_test.json      (来自②af_cutoff_check.py)
输出:
  Intra1_pos_clean.txt / Intra1_neg_clean.txt   (train: 删结构泄漏蛋白)
  Intra0_pos_clean.txt / Intra0_neg_clean.txt   (val: 同上)
  Intra2_pos_clean.txt / Intra2_neg_clean.txt   (test: 删AF泄漏蛋白)
  leakage_work/clean_report.json

运行: python make_clean_splits.py
"""
import json
from pathlib import Path

WORK = Path("leakage_work")
SL   = WORK / "structural_leak.json"
AFL  = WORK / "af_leak_test.json"

SPLIT_FILES = {
    "train": ["Intra1_pos_rr.txt", "Intra1_neg_rr.txt"],
    "val":   ["Intra0_pos_rr.txt", "Intra0_neg_rr.txt"],
    "test":  ["Intra2_pos_rr.txt", "Intra2_neg_rr.txt"],
}


def parse(p: Path):
    toks = p.read_text().split()
    if len(toks) % 2:
        toks = toks[:-1]
    return [(toks[i], toks[i+1]) for i in range(0, len(toks), 2)]


def main():
    # ---- 读入泄漏数据 ----
    struct_leak = json.loads(SL.read_text()) if SL.exists() else None
    af_leak = json.loads(AFL.read_text()) if AFL.exists() else None

    if not struct_leak:
        print("[WARN] structural_leak.json 不存在 → train用原始切分")
    if not af_leak:
        print("[WARN] af_leak_test.json 不存在 → test用原始切分")

    banned_train = set(struct_leak["banned_train"]) if struct_leak else set()
    banned_test = {a for a, v in (af_leak or {}).items() if v.get("af_leak")}

    print(f"[INFO] train侧结构泄漏蛋白: {len(banned_train)}")
    print(f"[INFO] test侧AF泄漏蛋白:    {len(banned_test)}")

    # ---- 生成清洁切分 ----
    report = {}
    print(f"\n{'='*64}")
    print(f" 最终清洁切分")
    print(f"{'='*64}")
    for split, files in SPLIT_FILES.items():
        banned = banned_train if split in ("train", "val") else banned_test
        report[split] = {}
        for fn in files:
            fp = Path(fn)
            if not fp.exists():
                continue
            pairs = parse(fp)
            kept = [p for p in pairs
                    if p[0] not in banned and p[1] not in banned]
            out = fp.with_name(fp.stem + "_clean.txt")
            out.write_text(" ".join(f"{a} {b}" for a, b in kept))
            report[split][fn] = {"before": len(pairs), "after": len(kept),
                                 "banned": len(banned)}
            print(f"  {split:<6}{fn:<24} {len(pairs):>7} → {len(kept):>7} "
                  f"({100*len(kept)/max(len(pairs),1):.1f}%)")

    with open(WORK / "clean_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] 报告: {WORK}/clean_report.json")
    print("[INFO] pilot 使用清洁切分: python pilot_official.py --split-mode clean ...")


if __name__ == "__main__":
    main()
