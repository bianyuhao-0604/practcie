"""Dataset layer for L3-PPI.

Supports three PPI collections:
* ``yeast``   -- Yu et al. yeast PPI, used in Scenario 1 (binary prediction).
* ``shs27k``  -- Shen et al. SHS27k, used in Scenario 2 (type prediction);
                 we treat it as binary for the base experiment.
* ``string``  -- STRING human PPI network (huge, subsample-friendly).

Expected on-disk layout under ``data/raw/<name>/``::

    proteins.fasta        # id<tab>sequence   OR  >id\\nseq
    interactions.txt      # id_a<tab>id_b[<tab>type]

If a dataset is not present, a small synthetic graph is generated so that the
whole pipeline (data -> pre-train -> fine-tune -> test) remains runnable
end-to-end for development / CI.  Set ``L3PPI_ALLOW_SYNTH=1`` to enable.
"""
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DataCfg, RunCfg, CACHE_ROOT


# ---------------------------------------------------------------------- #
# I/O
# ---------------------------------------------------------------------- #
def _read_fasta(path):
    seqs = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            else:
                seqs[cur].append(line)
    return {k: "".join(v).upper() for k, v in seqs.items()}


def _read_pairs(path):
    """Return list of (a, b, type).  type=None when no 3rd column."""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            typ = parts[2] if len(parts) > 2 else None
            out.append((a, b, typ))
    return out


def load_dataset(name, root):
    """Return (proteins: dict[id->seq], edges: list[(a,b,type)], node_list)."""
    root = Path(root) / name
    synth = os.environ.get("L3PPI_ALLOW_SYNTH", "1") == "1"
    if not root.exists() or not any(root.iterdir()):
        if synth:
            return _synthetic(name)
        raise FileNotFoundError(
            f"No raw data for '{name}' under {root}. Place proteins.fasta and "
            "interactions.txt there, or set L3PPI_ALLOW_SYNTH=1 for a demo graph."
        )

    proteins, edges = {}, []
    for fn in root.iterdir():
        fn = fn.name
        if "fasta" in fn or fn.endswith(".fa"):
            proteins = _read_fasta(root / fn)
        elif "interact" in fn or fn.endswith(".txt") or "edge" in fn:
            edges = _read_pairs(root / fn)
    if not proteins:
        # build protein id set from edge endpoints
        ids = sorted({u for a, b, _ in edges for u in (a, b)})
        proteins = {i: _random_seq(200) for i in ids}
    return proteins, edges, list(proteins.keys())


def _random_seq(n):
    import random
    return "".join(random.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(n))


def _synthetic(name):
    """A small random PPI graph that empirically obeys the L3 tendency."""
    rng = np.random.default_rng(0)
    n = 300
    ids = [f"P{i:03d}" for i in range(n)]
    proteins = {i: _random_seq(150) for i in ids}
    # plant communities so L3 signal exists
    comm = rng.integers(0, 6, n)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if comm[i] == comm[j] and rng.random() < 0.12:
                edges.append((ids[i], ids[j], None))
            elif rng.random() < 0.004:
                edges.append((ids[i], ids[j], None))
    print(f"[dataset] synthetic '{name}': {n} proteins, {len(edges)} edges")
    return proteins, edges, ids


# ---------------------------------------------------------------------- #
# L3 path statistics (used for the L3-rule validation figure & surrogate)
# ---------------------------------------------------------------------- #
def count_l3_paths(adj, u, v):
    """Number of length-3 paths between u and v in graph ``adj`` (set neighbours).

    A length-3 path u - x - y - v requires x != y and x, y not in {u,v}.
    We count ordered intermediate pairs (x, y) with edges u-x, x-y, y-v.
    """
    if u == v:
        return 0
    total = 0
    for x in adj.get(u, ()):
        if x == v:
            continue
        for y in adj.get(x, ()):
            if y == u or y == x:
                continue
            if v in adj.get(y, ()):
                total += 1
    return total


def compute_l3_features(edges):
    """Return (adj_dict, id2idx, idx2id).  adj stored as set of int indices."""
    ids = sorted({a for a, b, _ in edges for a in (a,)} | {b for a, b, _ in edges})
    id2idx = {p: i for i, p in enumerate(ids)}
    adj = defaultdict(set)
    for a, b, _ in edges:
        ai, bi = id2idx[a], id2idx[b]
        adj[ai].add(bi)
        adj[bi].add(ai)
    return dict(adj), id2idx, ids


# ---------------------------------------------------------------------- #
# Splits
# ---------------------------------------------------------------------- #
def edge_split(edges, cfg: DataCfg, seed=42):
    """Split edges (positive interactions) into train/val/test by ``cfg.split``.

    * random -- random shuffle.
    * bfs / dfs -- graph-aware via BFS/DFS tree ordering (mimics cold-split).
    Negative pairs are generated by random non-edge sampling (same # as pos,
    balanced) at batch-construction time, controlled by ``balance_neg``.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(edges))
    if cfg.split == "random":
        rng.shuffle(idx)
    else:
        idx = _order_index(edges, cfg.split, seed)
    n = len(edges)
    n_val = int(n * cfg.val_ratio)
    n_test = int(n * 0.2) if cfg.split == "random" else int(n * (1 - cfg.train_ratio - cfg.val_ratio))
    n_test = max(n_test, 1)
    test = [edges[i] for i in idx[:n_test]]
    val = [edges[i] for i in idx[n_test:n_test + n_val]]
    train = [edges[i] for i in idx[n_test + n_val:]]
    return train, val, test


def _order_index(edges, mode, seed):
    """BFS/DFS ordering of edges so early edges form the training core."""
    adj = defaultdict(list)
    for i, (a, b, _) in enumerate(edges):
        adj[a].append(i); adj[b].append(i)
    nodes = list(adj.keys())
    rng = np.random.default_rng(seed)
    start = nodes[0]
    seen, order = set(), []
    stack = [start]
    while stack:
        cur = stack.pop(0) if mode == "bfs" else stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for ei in adj[cur]:
            if ei not in order:
                order.append(ei)
        for nb in adj[cur]:
            if nb not in seen:
                stack.append(nb)
    rest = [i for i in range(len(edges)) if i not in order]
    rng.shuffle(rest)
    return order + rest


# ---------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------- #
class PPIDataset(Dataset):
    """PPI pair dataset.  Each item: (id_a, id_b, label, l3_count).

    ``label`` = 1 for an interacting pair, 0 for a negative pair.
    ``l3_count`` is pre-computed #L3 paths in the *full* PPI graph (used as
    an auxiliary feature / for the L3-rule analysis).  Negatives are sampled
    once at construction unless ``online_neg=True``.
    """

    AA = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, proteins, edges_pos, id2idx, adj, cfg: DataCfg,
                 split="train", online_neg=False, neg_pos_ratio=1.0, seed=0):
        self.proteins = proteins
        self.ids = list(proteins.keys())
        self.id2idx = id2idx
        self.adj = adj
        self.cfg = cfg
        self.split = split
        self.online_neg = online_neg
        self.rng = np.random.default_rng(seed)
        self.max_len = 512

        pos = [(a, b) for a, b, _ in edges_pos]
        self.pos = pos
        # pre-generate negatives (non-edges)
        allset = set(self.ids)
        pos_set = {tuple(sorted((a, b))) for a, b in pos}
        self._pos_set = pos_set
        self.neg = self._sample_neg(int(len(pos) * neg_pos_ratio))

        # cache L3 counts for positives (expensive, do once)
        self._l3 = {}
        if split == "train":
            pass  # computed on demand / can be skipped
        self._precompute_l3(pos)

    def _precompute_l3(self, pos_pairs):
        for a, b in pos_pairs:
            ai, bi = self.id2idx.get(a, -1), self.id2idx.get(b, -1)
            if ai < 0 or bi < 0:
                self._l3[(a, b)] = 0
            else:
                self._l3[(a, b)] = count_l3_paths(self.adj, ai, bi)

    def _sample_neg(self, n):
        ids = self.ids
        out = []
        tries = 0
        while len(out) < n and tries < n * 30:
            tries += 1
            a, b = self.rng.choice(ids), self.rng.choice(ids)
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in self._pos_set:
                continue
            out.append((a, b))
        return out

    def __len__(self):
        return len(self.pos) + len(self.neg)

    def encode(self, seq):
        """Integer encode amino-acid sequence -> (L,) LongTensor, padded."""
        seq = seq[:self.max_len]
        toks = [self.AA.index(c) if c in self.AA else 0 for c in seq]
        x = torch.zeros(self.max_len, dtype=torch.long)
        x[:len(toks)] = torch.tensor(toks, dtype=torch.long)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:len(toks)] = True
        return x, mask

    def __getitem__(self, i):
        if i < len(self.pos):
            a, b = self.pos[i]
            label = 1
            l3 = self._l3.get((a, b), 0)
        else:
            a, b = self.neg[i - len(self.pos)]
            label = 0
            l3 = 0
        xa, ma = self.encode(self.proteins[a])
        xb, mb = self.encode(self.proteins[b])
        return dict(a=xa, a_mask=ma, b=xb, b_mask=mb,
                    id_a=a, id_b=b,
                    seq_a=self.proteins[a], seq_b=self.proteins[b],
                    label=torch.tensor(label, dtype=torch.float),
                    l3=torch.tensor(float(l3), dtype=torch.float))


def collate(batch):
    return dict(
        a=torch.stack([b["a"] for b in batch]),
        a_mask=torch.stack([b["a_mask"] for b in batch]),
        b=torch.stack([b["b"] for b in batch]),
        b_mask=torch.stack([b["b_mask"] for b in batch]),
        id_a=[b["id_a"] for b in batch],
        id_b=[b["id_b"] for b in batch],
        seq_a=[b["seq_a"] for b in batch],
        seq_b=[b["seq_b"] for b in batch],
        label=torch.stack([b["label"] for b in batch]),
        l3=torch.stack([b["l3"] for b in batch]),
    )


def build_loaders(cfg=None, run=None):
    cfg = cfg or DataCfg()
    run = run or RunCfg()
    from config import DATA_ROOT
    proteins, edges, _ = load_dataset(cfg.name, DATA_ROOT)
    adj, id2idx, _ = compute_l3_features(edges)
    train_e, val_e, test_e = edge_split(edges, cfg)
    print(f"[dataset] {cfg.name}: {len(proteins)} proteins, {len(edges)} PPIs | "
          f"train {len(train_e)} val {len(val_e)} test {len(test_e)}")

    train = PPIDataset(proteins, train_e, id2idx, adj, cfg, "train",
                       seed=hash((cfg.name, "train")) & 0xffff)
    val = PPIDataset(proteins, val_e, id2idx, adj, cfg, "val", seed=1)
    test = PPIDataset(proteins, test_e, id2idx, adj, cfg, "test", seed=2)

    mk = lambda ds, bs, shuf: DataLoader(ds, batch_size=bs, shuffle=shuf,
                                         collate_fn=collate, num_workers=run.num_workers)
    return dict(
        train=mk(train, 32, True),
        val=mk(val, 32, False),
        test=mk(test, 32, False),
        train_ds=train, val_ds=val, test_ds=test,
        meta=dict(n_proteins=len(proteins), n_edges=len(edges),
                  n_train=len(train_e), n_val=len(val_e), n_test=len(test_e)),
    )


def l3_rule_analysis(cfg=None, li_max=7, save=None):
    """Reproduce the L3-rule correlation analysis (Pearson + mutual info)."""
    cfg = cfg or DataCfg()
    from config import DATA_ROOT
    proteins, edges, _ = load_dataset(cfg.name, DATA_ROOT)
    adj, id2idx, ids = compute_l3_features(edges)
    idx2id = {v: k for k, v in id2idx.items()}
    n = len(ids)

    def count_li(adj, u, v, i):
        # iterative BFS over paths of length i; count simple paths
        if u == v:
            return 0
        # dp[l][node] = number of simple paths from u of length l ending at node
        from collections import defaultdict
        prev = defaultdict(int)
        prev[u] = 1
        for step in range(i):
            cur = defaultdict(int)
            for node, ways in prev.items():
                for nb in adj.get(node, ()):
                    if step < i - 1 and nb == v:
                        continue
                    cur[nb] += ways
            prev = cur
        return prev.get(v, 0)

    rng = np.random.default_rng(0)
    pos_pairs = [(id2idx[a], id2idx[b]) for a, b, _ in edges]
    if len(pos_pairs) > 4000:
        pos_pairs = rng.choice(pos_pairs, 4000, replace=False).tolist()
    # negative = random non-edges
    pos_set = {tuple(sorted((a, b))) for a, b, _ in edges}
    neg_pairs = []
    tries = 0
    while len(neg_pairs) < len(pos_pairs) and tries < len(pos_pairs) * 30:
        tries += 1
        u, v = rng.integers(0, n, 2).tolist()
        if u == v:
            continue
        if tuple(sorted((u, v))) in pos_set:
            continue
        neg_pairs.append((u, v))
    all_pairs = pos_pairs + neg_pairs
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))

    from scipy.stats import pearsonr, mutual_info_score
    rows = []
    for li in range(2, li_max + 1):
        feats = []
        for u, v in all_pairs:
            feats.append(count_li(adj, u, v, li))
        feats = np.array(feats, dtype=float)
        # discretize for MI
        disc = np.digitize(feats, np.quantile(feats[feats > 0], np.linspace(0, 1, 11)) if (feats > 0).any() else [0])
        r, _ = pearsonr(feats, y)
        mi = mutual_info_score(y, disc)
        rows.append(dict(li=li, pearson=r, mutual_info=mi, mean_pos=feats[:len(pos_pairs)].mean(),
                         mean_neg=feats[len(pos_pairs):].mean()))
        print(f"[L{li}] pearson={r:+.4f}  MI={mi:.4f}  mean(pos)={rows[-1]['mean_pos']:.2f}  mean(neg)={rows[-1]['mean_neg']:.2f}")
    if save:
        import pandas as pd
        pd.DataFrame(rows).to_csv(save, index=False)
        print(f"[L3-rule] saved -> {save}")
    return rows
