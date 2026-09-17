# debug_download.py — 运行这个确认具体错误
import requests
from pathlib import Path

# 1. 先测试单个 API 调用
acc = "P05067"  # 用一个已知存在的蛋白
url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
r = requests.get(url, timeout=30)
print(f"API 状态: {r.status_code}")
meta = r.json()[0] if r.status_code == 200 else {}
print(f"meta keys: {list(meta.keys())}")

# 2. 测试实际下载
if meta:
    pdb_url = meta.get("pdbUrl")
    print(f"pdbUrl: {pdb_url}")
    if pdb_url:
        try:
            r2 = requests.get(pdb_url, stream=True, timeout=30)
            print(f"下载状态: {r2.status_code}")
            if r2.status_code == 200:
                # 3. 测试写入文件
                dest = Path("test_download.pdb")
                with open(dest, "wb") as f:
                    for chunk in r2.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✓ 文件写入成功: {dest.stat().st_size} bytes")
        except Exception as e:
            print(f"✗ 下载/写入失败: {type(e).__name__}: {e}")

# 4. 检查目标目录是否存在
target_dir = Path("af_output/monomers")
print(f"\n目标目录存在: {target_dir.exists()}")
if target_dir.exists():
    print(f"目录可写: {target_dir.is_dir()}")
