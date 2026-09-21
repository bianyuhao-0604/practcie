# scripts/09_evaluate_all.py — 汇总 metrics_*.json → all_metrics.csv + Δ 计算
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import RESULTS
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in RESULTS.glob("metrics_*.json")]
    if not rows:
        sys.exit("!! no metrics_*.json found under results/")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "all_metrics.csv", index=False)

    piv = df.groupby(["r_stage", "ablation"])[["test_auprc", "test_acc"]] \
            .agg(["mean", "std"])
    print("\n===== summary (mean +/- std over seeds) =====")
    print(piv.round(4).to_string())

    r1 = df[(df.r_stage == "R1") & (df.ablation == "none")].test_auprc.mean()
    r7 = df[(df.r_stage == "R7") & (df.ablation == "none")].test_auprc.mean()
    print(f"\nDelta(R7 - R1) = {r7 - r1:+.4f}")
    print("verdict: >= +0.01 -> multi-modal direction supported; "
          "~0 or negative -> inspect per-run hist & ablations")

    ng = df[(df.r_stage == "R7") & (df.ablation == "no_global")].test_auprc.mean()
    if np.isfinite(ng):
        print(f"Delta(R7 - R7_no_global) = {r7 - ng:+.4f}  "
              f"(protein-level pathway contribution)")


if __name__ == "__main__":
    main()
