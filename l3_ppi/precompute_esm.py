"""Pre-compute ESM embeddings for all proteins in a dataset and cache to disk.

Usage:
    python precompute_esm.py --dataset shs148k
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from config import EncoderCfg, DATA_ROOT, CACHE_ROOT
from dataset import load_dataset
from encoder import ESMEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shs148k")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    proteins, edges, ids = load_dataset(args.dataset, DATA_ROOT)
    print(f"{len(proteins)} proteins, {len(edges)} edges", flush=True)

    enc = ESMEncoder(EncoderCfg).to(device)
    out_path = CACHE_ROOT / f"esm_emb_{args.dataset}.pt"

    cache = {}
    all_ids = list(proteins.keys())
    t0 = time.time()
    for i in range(0, len(all_ids), args.batch_size):
        sub = all_ids[i:i + args.batch_size]
        seqs = [proteins[p] for p in sub]
        h = enc.forward_str(seqs, device, ids=sub)  # populates enc._cache
        for p in sub:
            cache[p] = enc._cache[p].cpu()
        if (i // args.batch_size) % 10 == 0:
            print(f"  {i+len(sub)}/{len(all_ids)} ({time.time()-t0:.0f}s)", flush=True)

    torch.save(cache, out_path)
    print(f"saved {len(cache)} embeddings -> {out_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
