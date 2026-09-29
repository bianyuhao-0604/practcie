import os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

URLS = sys.argv[sys.argv.index("--urls") + 1]
OUT  = sys.argv[sys.argv.index("--outdir") + 1]
W    = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
os.makedirs(OUT, exist_ok=True)
FAILED = os.path.join(OUT, "_failed.txt")

tl = threading.local()
def sess():
    if not hasattr(tl, "s"):
        tl.s = requests.Session()
    return tl.s

def target_path(url):
    m = re.search(r"AF-([A-Za-z0-9]+)-F\d+-model_v\d+\.cif", url)
    name = f"{m.group(1)}.cif" if m else os.path.basename(url.split("?")[0])
    return os.path.join(OUT, name)

def fetch(url):
    out = target_path(url)
    if os.path.exists(out) and os.path.getsize(out) > 10_000:
        return "skip"
    tmp = out + ".part"
    for attempt in range(4):
        try:
            r = sess().get(url, timeout=(10, 120))
            if r.status_code == 404:
                return "fail"           # ?????????
            r.raise_for_status()
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, out)        # ??????????????
            return "ok"
        except Exception:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
            if attempt == 3:
                return "fail"
            time.sleep(1 + attempt)
    return "fail"

lines = [l.strip() for l in open(URLS) if l.strip()]
ok = skip = fail = 0
with ThreadPoolExecutor(W) as ex:
    futs = {ex.submit(fetch, u): u for u in lines}
    for fut in as_completed(futs):
        st = fut.result()
        if st == "fail":
            with open(FAILED, "a") as f:
                f.write(futs[fut] + "\n")
        ok += st == "ok"; skip += st == "skip"; fail += st == "fail"
        n = ok + skip + fail
        if n % 100 == 0 or n == len(lines):
            print(f"[{n}/{len(lines)}] ok={ok} skip={skip} fail={fail}", flush=True)
print(f"DONE ok={ok} skip={skip} fail={fail}", flush=True)
