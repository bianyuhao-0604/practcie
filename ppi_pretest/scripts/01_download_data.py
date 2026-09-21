import sys, json, zipfile, urllib.request, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import DATA
from tqdm import tqdm

ARTICLE_ID = "21591618"
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
extract_dir = DATA / "raw"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def http_get(url, dest=None, min_size=100):
    """带 UA 的下载；dest=None 时返回 bytes。校验内容非 HTML。"""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) < min_size or data[:1] == b"<":
        head = data[:300].decode("utf-8", errors="ignore")
        raise RuntimeError(f"返回内容不是文件（疑似 HTML）:\n{head}")
    if dest is not None:
        dest.write_bytes(data)
        return len(data)
    return data

def is_zip(b: bytes) -> bool:
    return b[:2] == b"PK"          # zip 魔数 PK\x03\x04

def main():
    # ---- 1. 查询文件清单 ----
    print(f"查询 figshare 文章 {ARTICLE_ID} 的文件清单 ...")
    meta = json.loads(http_get(API_URL).decode("utf-8"))
    print(f"文章: {meta['title']}  (版本状态见 figshare 页面)")
    files = meta.get("files", [])
    if not files:
        sys.exit("!! API 未返回文件清单，请用下方'手动兜底'方案")
    for f in files:
        print(f"  - {f['name']}  ({f['size']/1e6:.1f} MB)")
        print(f"    {f['download_url']}")

    # ---- 2. 逐文件下载 ----
    dl_dir = DATA / "figshare_files"
    dl_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []
    for f in files:
        dest = dl_dir / f["name"]
        ok = dest.exists() and dest.stat().st_size == f["size"]
        if not ok:
            print(f"下载 {f['name']} ...")
            n = http_get(f["download_url"], dest=dest, min_size=f["size"] * 0.95)
            print(f"  完成 {n/1e6:.1f} MB")
        local_paths.append(dest)

    # ---- 3. 解压所有 zip 到 raw/，其余文件直接拷贝 ----
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    for p in local_paths:
        if is_zip(p.read_bytes()[:4]):
            print(f"解压 {p.name}")
            with zipfile.ZipFile(p) as z:
                z.extractall(extract_dir)
        else:
            shutil.copy2(p, extract_dir / p.name)

    (extract_dir / ".done").touch()
    print("\n包内文件清单:")
    for p in [x for x in extract_dir.rglob("*") if x.is_file()]:
        print("  ", p.relative_to(extract_dir), f"({p.stat().st_size/1e3:.0f} KB)")
    print("\n完成。下一步: python scripts\\02_parse_and_sample.py")

if __name__ == "__main__":
    main()
