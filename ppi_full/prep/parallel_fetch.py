"""通用断点续跑并行下载器（AFDB cif / UniProt txt 共用）。
已存在且非空即跳过；.part 原子替换；指数退避；失败清单落盘。
用法：
  python prep/parallel_fetch.py --urls workdir/urls/afdb_urls.txt    --outdir workdir/afdb --workers 12
  python prep/parallel_fetch.py --urls workdir/urls/uniprot_urls.txt --outdir workdir/uniprot --workers 6
"""
import argparse, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch(url, out_path, retries=4, timeout=60):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return ("skip", url)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ppi-prep/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            with open(out_path + ".part", "wb") as f:
                f.write(data)
            os.replace(out_path + ".part", out_path)
            return ("ok", url)
        except Exception as e:
            if attempt == retries - 1:
                return ("fail", f"{url}\t{e}")
            time.sleep(2 ** attempt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    urls = [l.strip() for l in open(a.urls) if l.strip()]
    jobs = [(u, os.path.join(a.outdir, u.rstrip("/").split("/")[-1])) for u in urls]
    ok = skip = 0
    fails = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, u, p): u for u, p in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            st, info = fut.result()
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fails.append(info)
            if i % 500 == 0:
                print(f"{i}/{len(jobs)} ok={ok} skip={skip} fail={len(fails)}")
    with open(os.path.join(a.outdir, "_failed.txt"), "w") as f:
        f.write("\n".join(fails))
    print(f"DONE ok={ok} skip={skip} fail={len(fails)} → _failed.txt")

if __name__ == "__main__":
    main()
