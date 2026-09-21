import sys, re, collections
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import *

UNIPROT_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")

def norm_id(s):
    """兼容 >sp|P12345|NAME / P12345-2 / taxon:P12345 等格式 → 纯 accession。"""
    s = str(s).strip()
    if "|" in s:                       # fasta 头 sp|P12345|NAME
        p = s.split("|")
        if len(p) >= 2 and p[1]:
            s = p[1]
    s = s.split("-")[0]                # 亚型 P12345-2 → P12345
    s = s.split(":")[-1]               # 带前缀形式
    return s

def is_uniprot(s):
    return bool(UNIPROT_RE.match(s))

def read_pair_file(fp, label):
    """解析 pos/neg 对文件：每行两个 UniProt accession。"""
    rows, bad, first = [], 0, None
    with open(fp, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if first is None:
                first = line
            parts = [p for p in re.split(r"[\t\s]+", line) if p]
            if len(parts) == 1 and "__" in parts[0]:
                parts = parts[0].split("__", 1)
            if len(parts) < 2:
                bad += 1
                continue
            a, b = norm_id(parts[0]), norm_id(parts[1])
            if is_uniprot(a) and is_uniprot(b):
                rows.append((a, b, label))
            else:
                bad += 1
    df = pd.DataFrame(rows, columns=["pidA", "pidB", "label"])
    print(f"[{fp.name}] label={label}  解析 {len(df)} 对, 丢弃 {bad} 行, 样例行: {first!r}")
    return df

def read_fasta(fp):
    seqs, name = {}, None
    with open(fp, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = norm_id(line[1:].split()[0])
                seqs.setdefault(name, [])
            elif name is not None:
                seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}

def discover(raw_dir):
    """按 Intra<n>_(pos|neg) 分组；fasta 单独识别。"""
    groups = collections.defaultdict(dict)
    fasta = None
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        m = re.match(r"intra(\d+)_(pos|neg)", n)
        if m:
            groups[int(m.group(1))][m.group(2)] = p
        elif n.endswith((".fasta", ".fa")) or "oneliner" in n:
            fasta = fasta or p
    return groups, fasta

def stratified_sample(df, n, seed):
    df = df.copy()
    df["len_max"] = df[["_lenA", "_lenB"]].max(axis=1)
    df["bucket"] = pd.cut(df["len_max"], [0, 300, 600, 10**9], labels=["s", "m", "l"])
    rng = np.random.RandomState(seed)
    parts = []
    for _, g in df.groupby(["label", "bucket"], observed=True):
        k = min(len(g), int(round(n * len(g) / len(df))))
        parts.append(g.sample(k, random_state=rng))
    out = pd.concat(parts).sample(frac=1, random_state=rng)
    return out.head(n).reset_index(drop=True)

if __name__ == "__main__":
    raw_dir = DATA / "raw"
    groups, fasta = discover(raw_dir)
    if fasta is None:
        sys.exit("!! 未找到 human_swissprot_oneliner.fasta")
    print("fasta:", fasta.name)

    # ---- 按 pos+neg 总行数自动映射 split ----
    totals = {}
    for k in sorted(groups):
        if not {"pos", "neg"}.issubset(groups[k]):
            print(f"!! Intra{k} 缺文件: {groups[k]}")
            continue
        n = sum(1 for _ in open(groups[k]["pos"], encoding="utf-8", errors="ignore")) + \
            sum(1 for _ in open(groups[k]["neg"], encoding="utf-8", errors="ignore"))
        totals[k] = n
        print(f"Intra{k}: pos+neg = {n}")
    def pick(target):
        cands = {k: n for k, n in totals.items() if abs(n - target) <= target * 0.02}
        if not cands:
            sys.exit(f"!! 没有总行数≈{target} 的 split, 现有: {totals}")
        return min(cands, key=lambda k: abs(totals[k] - target))
    k_tr, k_va, k_te = pick(ROW_TRAIN), pick(ROW_VAL), pick(ROW_TEST)
    print(f"映射: train=Intra{k_tr}, val=Intra{k_va}, test=Intra{k_te}\n")

    fa = read_fasta(fasta)
    print(f"fasta 蛋白数: {len(fa)}\n")

    for tag, k, n in [("tr", k_tr, TRAIN_N), ("va", k_va, VAL_N), ("te", k_te, TEST_N)]:
        pos = read_pair_file(groups[k]["pos"], 1)
        neg = read_pair_file(groups[k]["neg"], 0)
        df = pd.concat([pos, neg], ignore_index=True).drop_duplicates(
            subset=["pidA", "pidB"], keep="first")
        df["_lenA"] = df["pidA"].map(lambda p: len(fa.get(p, "")))
        df["_lenB"] = df["pidB"].map(lambda p: len(fa.get(p, "")))
        before = len(df)
        df = df[(df._lenA > 0) & (df._lenB > 0)].reset_index(drop=True)
        print(f"{tag}: 合并 {before} -> fasta 过滤后 {len(df)}")
        df = stratified_sample(df, n, SUBSET_SEED)
        df[["pidA", "pidB", "label"]].to_csv(DATA / f"pairs_{tag}.csv", index=False)
        print(f"  采样后 {len(df)}, 正例率 {df.label.mean():.3f}\n")

    # ---- 汇总需要特征的蛋白 & 写出序列 ----
    parts = []
    for tag in ["tr", "va", "te"]:
        d = pd.read_csv(DATA / f"pairs_{tag}.csv")
        parts += [d.pidA, d.pidB]
    ids = sorted(set(pd.concat(parts)))
    print(f"独立蛋白数: {len(ids)}")
    (DATA / "proteins.txt").write_text("\n".join(ids))
    with open(DATA / "sequences.fasta", "w") as f:
        for pid in ids:
            f.write(f">{pid}\n{fa[pid][:MAX_LEN]}\n")
    print("完成: data/pairs_{tr,va,te}.csv, data/proteins.txt, data/sequences.fasta")
