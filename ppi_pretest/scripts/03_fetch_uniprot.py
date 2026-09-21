import sys, time, json, hashlib, urllib.parse, urllib.request, urllib.error
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import DATA, GENOME_DIM

BASE = "https://rest.uniprot.org"
UA = {"User-Agent": "ppi-pretest/0.1 (python-urllib)"}
FIELDS = "accession,cc_function,go_c,cc_subcellular_location"

def _read(resp):
    return resp.read().decode("utf-8", errors="ignore")

def request_with_retry(url, data=None, headers=None, tries=4, timeout=120):
    """带重试；HTTP 错误时打印响应体（定位 400 真因）。"""
    hdrs = dict(UA)
    if headers: hdrs.update(headers)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=hdrs)
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = ""
            try: body = e.read().decode("utf-8", errors="ignore")[:500]
            except Exception: pass
            print(f"  HTTP {e.code}: {body or e.reason}")
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(5 * (i + 1)); continue
            raise
        except Exception as e:
            print(f"  retry {i+1}: {e}"); last = e
            time.sleep(5 * (i + 1))
    raise last

def genome_hash(go_subcell):
    v = [0.0] * GENOME_DIM
    for tok in go_subcell.replace(",", " ").replace(";", " ").split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % GENOME_DIM
        v[h] = 1.0
    return v

def parse_tsv(text):
    """按表头解析 TSV → list[dict]；兼容 idmapping(有From列) 与 stream(有Accession列)。"""
    lines = text.strip().splitlines()
    if len(lines) < 2: return []
    hdr = lines[0].split("\t")
    idx = {"first": 0}
    for j, h in enumerate(hdr):
        hl = h.lower()
        if hl == "from": idx["from"] = j
        elif hl.startswith("accession") or hl == "entry": idx["acc"] = j
        elif hl.startswith("function"): idx["func"] = j
        elif "gene ontology (cellular" in hl: idx["go"] = j
        elif hl.startswith("subcellular"): idx["sub"] = j
    out = []
    for ln in lines[1:]:
        cols = ln.split("\t")
        g = lambda k: cols[idx[k]] if k in idx and idx[k] < len(cols) else ""
        out.append({"key": g("from") or g("acc") or g("first"),
                    "func": g("func"), "go": g("go"), "sub": g("sub")})
    return out

def fetch_via_idmapping(ids, chunk=1000):
    """官方推荐：POST 提交 ID 列表 → 轮询 → 流式取回。"""
    all_rows = []
    for s in range(0, len(ids), chunk):
        part = ids[s:s + chunk]
        data = urllib.parse.urlencode(
            {"from": "UniProtKB_AC-ID", "to": "UniProtKB", "ids": ",".join(part)}
        ).encode("utf-8")
        r = request_with_retry(f"{BASE}/idmapping/run", data=data,
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        job = json.loads(r.read().decode())["jobId"]
        print(f"  jobId={job} (提交 {len(part)} 个), 轮询中 ...")
        status = ""
        for _ in range(60):
            time.sleep(2)
            sj = json.loads(_read(request_with_retry(f"{BASE}/idmapping/{job}")))
            status = sj.get("jobStatus") or ("FINISHED" if "results" in sj else "")
            if status in ("FINISHED", "ERROR"): break
        if status != "FINISHED":
            raise RuntimeError(f"idmapping job 未完成: {status}")
        url = (f"{BASE}/idmapping/uniprotkb/results/stream/{job}"
               f"?format=tsv&fields={FIELDS}")
        rows = parse_tsv(_read(request_with_retry(url)))
        all_rows += rows
        print(f"  分片 {s}-{s+len(part)}: 返回 {len(rows)} 行")
    return all_rows

def fetch_via_get_small(ids):
    """兜底：每 25 个 accession 一次 GET；单批失败只损失该批。"""
    all_rows = []
    n_fail = 0
    for i in range(0, len(ids), 25):
        chunk = ids[i:i + 25]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = (f"{BASE}/uniprotkb/stream?query={urllib.parse.quote(q)}"
               f"&format=tsv&fields={FIELDS}")
        try:
            all_rows += parse_tsv(_read(request_with_retry(url)))
        except Exception as e:
            n_fail += 1
            print(f"  批 {i} 失败(跳过): {e}")
        if (i // 25) % 10 == 0:
            print(f"  GET 进度 {min(i+25, len(ids))}/{len(ids)}")
        time.sleep(1)
    print(f"  GET 完成, 失败批数 {n_fail}")
    return all_rows

def fill(rows):
    text_map, genome_map = {}, {}
    for r in rows:
        if not r["key"]: continue
        text_map[r["key"]] = " ".join((r["func"] or "").split())[:1500]
        genome_map[r["key"]] = genome_hash((r["go"] or "") + " " + (r["sub"] or ""))
    return text_map, genome_map

if __name__ == "__main__":
    ids = sorted(set((DATA / "proteins.txt").read_text().split()))
    print(f"待拉取 {len(ids)} 蛋白")
    text_map, genome_map = {}, {}
    try:
        print("[方式1] UniProt ID mapping 服务 ...")
        text_map, genome_map = fill(fetch_via_idmapping(ids))
    except Exception as e:
        print(f"[方式1] 失败: {e}\n[方式2] 切换小批量 GET 兜底 ...")
        text_map, genome_map = fill(fetch_via_get_small(ids))

    miss = [a for a in ids if not text_map.get(a)]
    print(f"\ntext 缺失 {len(miss)}/{len(ids)} (将用占位文本; 缺失样本前10: {miss[:10]})")
    np.savez_compressed(DATA / "uniprot_cache.npz",
                        ids=json.dumps(ids),
                        text=json.dumps({a: text_map.get(a, "") for a in ids}),
                        genome=json.dumps({a: genome_map.get(a, [0.0] * GENOME_DIM)
                                           for a in ids}))
    print("完成: data/uniprot_cache.npz")
