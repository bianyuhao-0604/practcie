"""Mine L3 interaction patterns from a real PPI network for surrogate training.

For every edge (u, v) in the training PPI graph we count length-3 paths
(u - x - y - v) and extract the local ``pattern graph`` = the ego subgraph
induced by {u, v} plus their length-2 neighbours along L3 paths.  The
resulting graph is labelled positive (it is a real interaction with an L3
signature) or negative (a non-edge with few/no L3 paths).  These become
the pre-training samples for ``L3PPISurrogate``.

Saved artefact (pickled list of tuples):
    (x: (N,D) float, edge_index: (2,E) long, edge_weight: (E,) float, label: 0/1)

Usage:
    python mine_patterns.py --dataset yeast --out patterns.pt
"""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(__file__))

import torch
from collections import defaultdict
from config import DATA_ROOT
from dataset import load_dataset, compute_l3_features


def build_pattern(node_feat, u, v, adj, max_branches=8):
    """Return (x, edge_index, edge_weight) for the L3 pattern around (u,v).

    Branch nodes = length-2 neighbours of u that connect to v through one
    more node (i.e. endpoints of L3 paths u-x-y-v).  We keep at most
    ``max_branches`` such branches; each branch forms the 3-edge path
    u - x - y - v, which we encode as the prompt topology
    u - p_i - p0 - v (see model.py).
    """
    # find L3 intermediate pairs (x, y) with u-x, x-y, y-v
    branches = []
    seen_y = set()
    for x in adj.get(u, ()):
        if x == v:
            continue
        for y in adj.get(x, ()):
            if y == u or y == x or y == v:
                continue
            if v in adj.get(y, ()) and y not in seen_y:
                seen_y.add(y)
                branches.append((x, y))
            if len(branches) >= max_branches:
                break
        if len(branches) >= max_branches:
            break

    K = len(branches)
    if K == 0:
        return None
    D = len(node_feat[0])
    # node order: u, v, p0, p1..pK (= branch nodes x,y collapsed to one virtual node)
    x_vec = torch.zeros(K + 3, D)
    x_vec[0] = torch.tensor(node_feat[u], dtype=torch.float)
    x_vec[1] = torch.tensor(node_feat[v], dtype=torch.float)
    for i, (bx, by) in enumerate(branches):
        feat = (torch.tensor(node_feat[bx], dtype=torch.float) +
                torch.tensor(node_feat[by], dtype=torch.float)) / 2.0
        x_vec[i + 3] = feat
    row, col = [], []
    for i in range(1, K + 1):
        row += [0, i + 2, 2]; col += [i + 2, 2, 1]
    return x_vec, torch.tensor([row, col], dtype=torch.long), torch.ones(3 * K)


def mine(name, out, max_per=2000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    proteins, edges, ids = load_dataset(name, DATA_ROOT)
    adj, id2idx, _ = compute_l3_features(edges)
    idx2id = {v: k for k, v in id2idx.items()}
    D = 64  # placeholder feature dim; replaced by encoder dim at training time
    # NOTE: index range is len(id2idx) (all edge endpoints), not len(ids)
    # (proteins only) -- some edge endpoints may lack a sequence.
    node_feat = {i: [rng.random() for _ in range(D)] for i in range(len(id2idx))}

    pos, neg = [], []
    # positives: sample edges with >=1 L3 path
    for a, b, _ in edges:
        if len(pos) >= max_per:
            break
        ai, bi = id2idx.get(a, -1), id2idx.get(b, -1)
        if ai < 0 or bi < 0:
            continue
        g = build_pattern(node_feat, ai, bi, adj)
        if g is None:
            continue
        pos.append(g + (1.0,))

    # negatives: random non-edges (likely few L3 paths)
    pos_set = {tuple(sorted((id2idx[a], id2idx[b]))) for a, b, _ in edges}
    n = len(ids); tries = 0
    while len(neg) < len(pos):
        tries += 1
        u, v = rng.integers(0, n, 2).tolist()
        if u == v or tuple(sorted((u, v))) in pos_set:
            continue
        g = build_pattern(node_feat, u, v, adj)
        if g is None:
            continue
        neg.append(g + (0.0,))
        if tries > max_per * 50:
            break

    data = pos + neg
    rng.shuffle(data)
    torch.save(data, out)
    n_pos = sum(1 for *_, l in data if l == 1.0)
    print(f"[{name}] mined {len(data)} patterns ({n_pos} pos / {len(data) - n_pos} neg) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yeast")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-per", type=int, default=2000)
    args = ap.parse_args()
    out = args.out or str((DATA_ROOT.parent / "processed" / f"patterns_{args.dataset}.pt"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mine(args.dataset, out, args.max_per)


if __name__ == "__main__":
    main()
