"""序列 fasta：整包拉人参考蛋白质组按 pid 过滤；同工型映射规范条目；缺失走 REST 兜底。
产出 sequences.fasta + _missing.txt。多物种数据集：REST 兜底可覆盖任意 pid（慢但可行）。"""
import argparse, gzip, os, time, urllib.request, urllib.error

def uniprot_id(header):
    h = header.split()[0]
    return h.split("|")[1] if "|" in h else h

def read_fasta_lines(lines):
    seqs, name, buf = {}, None, []
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = uniprot_id(line[1:]), []
        else:
            buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="./workdir/cache")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    gz = os.path.join(a.cache, "UP000005640_9606.fasta.gz")
    if not os.path.exists(gz):
        url = ("https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
               "knowledgebase/reference_proteomes/Eukaryota/UP000005640_9606.fasta.gz")
        urllib.request.urlretrieve(url, gz)
    with gzip.open(gz, "rt") as f:
        prot = read_fasta_lines(f)
    print(f"proteome loaded: {len(prot)} entries")
    pids = [l.strip() for l in open(a.pids) if l.strip()]
    got, missing = {}, []
    for p in pids:
        if p in prot:
            got[p] = prot[p]
        elif p.split("-")[0] in prot:
            got[p] = prot[p.split("-")[0]]
        else:
            missing.append(p)
    for i, p in enumerate(missing):
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{p}.fasta"
            with urllib.request.urlopen(url, timeout=30) as r:
                seqs = read_fasta_lines(r.read().decode().splitlines())
            if p in seqs:
                got[p] = seqs[p]
                missing[i] = None
        except urllib.error.HTTPError:
            pass
        time.sleep(0.3)
        if (i + 1) % 50 == 0:
            print(f"fallback {i + 1}/{len(missing)}")
    still = [p for p in missing if p]
    with open(a.out, "w") as f:
        for p, s in got.items():
            f.write(f">{p}\n{s}\n")
    with open(a.out + "_missing.txt", "w") as f:
        f.write("\n".join(still))
    print(f"fasta written: {len(got)}/{len(pids)}; missing={len(still)} → _missing.txt")
    if still:
        print("!! 存在无法映射的 pid，先人工复核（作废 accession / 非人类蛋白）再继续")

if __name__ == "__main__":
    main()
