"""External / de novo prediction with a trained L3-PPI model.

Given either:
* a FASTA file of protein sequences (every pair is scored), or
* a TSV of pre-defined pairs (id_a <tab> id_b),

produce PPI likelihoods.  This is the "external predictor" interface --
the model is model-agnostic and uses ESM-2 embeddings when available.

Usage:
    # score all pairs in a FASTA
    python predict.py --fasta queries.fasta --out predictions.tsv

    # score a specific pair list
    python predict.py --pairs pairs.tsv --out predictions.tsv

    # restrict to a candidate list (FASTA) vs a target list
    python predict.py --fasta queries.fasta --targets targets.fasta --out preds.tsv

Input FASTA format:  >id  \\n  MKT...  (ids must be unique).
Pairs TSV format:    id_a <tab> id_b   (one per line).
"""
import argparse, os, sys, csv
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import Dataset, DataLoader
from config import (EncoderCfg, PromptCfg, FineTuneCfg, RunCfg, CKPT_ROOT)
from encoder import build_encoder
from model import L3PPI, PPIDataset
from utils import set_seed, pick_device


def read_fasta(path):
    seqs = OrderedDict()
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs.setdefault(cur, [])
            elif cur:
                seqs[cur].append(line)
    return {k: "".join(v).upper() for k, v in seqs.items()}


def read_pairs(path):
    out = []
    with open(path) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2:
                out.append((p[0], p[1]))
    return out


class QueryDataset(Dataset):
    AA = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, pairs, proteins, max_len=512):
        self.pairs = pairs
        self.proteins = proteins
        self.max_len = max_len

    def encode(self, seq):
        seq = seq[:self.max_len]
        toks = [self.AA.index(c) if c in self.AA else 0 for c in seq]
        x = torch.zeros(self.max_len, dtype=torch.long)
        x[:len(toks)] = torch.tensor(toks, dtype=torch.long)
        m = torch.zeros(self.max_len, dtype=torch.bool); m[:len(toks)] = True
        return x, m

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        xa, ma = self.encode(self.proteins.get(a, ""))
        xb, mb = self.encode(self.proteins.get(b, ""))
        return dict(a=xa, a_mask=ma, b=xb, b_mask=mb, id_a=a, id_b=b)


def collate(batch):
    return dict(
        a=torch.stack([b["a"] for b in batch]),
        a_mask=torch.stack([b["a_mask"] for b in batch]),
        b=torch.stack([b["b"] for b in batch]),
        b_mask=torch.stack([b["b_mask"] for b in batch]),
        id_a=[b["id_a"] for b in batch],
        id_b=[b["id_b"] for b in batch],
    )


def all_pairs(query_ids, target_ids=None):
    targets = target_ids if target_ids is not None else query_ids
    out = []
    for a in query_ids:
        for b in targets:
            if a != b:
                out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--targets", default=None, help="optional second FASTA (bait list)")
    ap.add_argument("--pairs", default=None, help="pre-defined pair list TSV")
    ap.add_argument("--out", default="predictions.tsv")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=0, help="only write top-K per query")
    args = ap.parse_args()

    ec, pc, fc, run = EncoderCfg, PromptCfg, FineTuneCfg, RunCfg
    if args.encoder: ec.use_esm = (args.encoder == "esm")
    if args.K: pc.K = args.K
    set_seed(run.seed)
    device = pick_device(run.device)

    # ---- assemble protein dict ----
    proteins = {}
    if args.fasta:
        proteins.update(read_fasta(args.fasta))
    targets = None
    if args.targets:
        targets = read_fasta(args.targets)
        proteins.update(targets)

    if not proteins:
        print("ERROR: supply --fasta (and optionally --targets) or --pairs.")
        return

    # ---- pairs to score ----
    if args.pairs:
        pairs = read_pairs(args.pairs)
    else:
        qids = list(read_fasta(args.fasta).keys())
        tids = list(targets.keys()) if targets else None
        pairs = all_pairs(qids, tids)

    ds = QueryDataset(pairs, proteins)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

    # ---- model ----
    encoder, _ = build_encoder(ec, device)
    surrogate = L3PPISurrogate(pc.node_dim, pc).to(device)
    ckpt_s = CKPT_ROOT / fc.save_name
    if ckpt_s.exists():
        surrogate.load_state_dict(torch.load(ckpt_s, map_location=device))
    model = L3PPI(encoder, surrogate, pc).to(device)
    ckpt_f = CKPT_ROOT / fc.save_name
    if ckpt_f.exists():
        model.load_state_dict(torch.load(ckpt_f, map_location=device))
    model.eval()

    # ---- inference ----
    results = []
    with torch.no_grad():
        for batch in loader:
            prob = model.predict(batch, tau=pc.tau_end)
            for i in range(len(batch["id_a"])):
                results.append((batch["id_a"][i], batch["id_b"][i], float(prob[i])))

    # optionally keep only top-K per query
    if args.top_k > 0:
        from collections import defaultdict
        grouped = defaultdict(list)
        for a, b, p in results:
            grouped[a].append((b, p))
        results = []
        for a, lst in grouped.items():
            for b, p in sorted(lst, key=lambda t: -t[1])[:args.top_k]:
                results.append((a, b, p))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["protein_a", "protein_b", "ppi_probability"])
        for a, b, p in sorted(results, key=lambda t: -t[2]):
            w.writerow([a, b, f"{p:.4f}"])
    print(f"wrote {len(results)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
