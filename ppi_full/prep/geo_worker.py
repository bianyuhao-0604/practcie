"""并行几何描述子（20 核）。N5：子进程异常捕获入账不中断。
iface 默认占位（仅管线打通）；论文级 R4/R5 换 --masif-dir 指向 MaSIF-site 官方输出。"""
import argparse, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_surf import parse_cif, descriptors, iface_heuristic
from config import MAX_LEN, Paths

def work(pid, cif_dir, feat_dir, masif_dir):
    out = os.path.join(feat_dir, f"surf_{pid}.npy")
    if os.path.exists(out):
        return (pid, "skip")
    cif = os.path.join(cif_dir, f"AF-{pid}-F1-model_v4.cif")
    coords, res, plddt = parse_cif(cif)
    L = min(len(res), MAX_LEN)
    D = descriptors(coords[:L], res[:L], plddt[:L])
    if masif_dir:
        iface = np.load(os.path.join(masif_dir, f"{pid}.iface.npy"))[:L]
    else:
        iface = iface_heuristic(coords[:L], D, plddt[:L])
    np.save(out, D.astype(np.float16))
    np.save(os.path.join(feat_dir, f"iface_{pid}.npy"), iface.astype(np.float16))
    np.save(os.path.join(feat_dir, f"plddt_{pid}.npy"), plddt[:L].astype(np.float16))
    return (pid, "ok")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--cif-dir", default="./workdir/afdb")
    ap.add_argument("--feat-dir", default=None)
    ap.add_argument("--masif-dir", default="")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    feat_dir = a.feat_dir or Paths().feat_dir
    os.makedirs(feat_dir, exist_ok=True)
    pids = [l.strip() for l in open(a.pids) if l.strip()]
    done = fails = 0
    fail_list = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, p, a.cif_dir, feat_dir, a.masif_dir): p for p in pids}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                pid, st = fut.result()
                if st == "ok":
                    done += 1
            except Exception as e:                        # N5：失败记账，进程池不中断
                fails += 1
                fail_list.append(f"{futs[fut]}\t{e}")
            if i % 500 == 0:
                print(f"{i}/{len(pids)} ok={done} fail={fails}")
    with open(os.path.join(feat_dir, "_geo_failed.txt"), "w") as f:
        f.write("\n".join(fail_list))
    print(f"DONE ok={done} fail={fails} → _geo_failed.txt")

if __name__ == "__main__":
    main()
