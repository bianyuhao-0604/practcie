import re, csv
from config import SIMCLR_LOG

pat = re.compile(
    r"ep\s+(\d+) loss ([\d.]+) \(inst ([\d.]+) sup ([\d.]+)\)(?: \| probe ([\d.]+))?")
rows = []
for ln in open("simclr.log", encoding="utf-8", errors="replace"):
    m = pat.search(ln)
    if m: rows.append({k: v for k, v in zip(
        ["ep", "loss", "inst", "sup", "probe"], m.groups()) if v})
with open(SIMCLR_LOG, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ep", "loss", "inst", "sup", "probe"], restval="")
    w.writeheader(); w.writerows(rows)

print(f"重建 simclr_log.csv: {len(rows)} 行 (含 probe {sum(1 for r in rows if 'probe' in r)} 行)")
