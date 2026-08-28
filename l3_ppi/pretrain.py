"""Stage 1: pre-train the L3 pattern recognition surrogate.

The surrogate (a GIN) learns to distinguish valid L3 pattern graphs from
random ones.  We train on synthetic patterns by default
(``SyntheticPatternDataset``); to use patterns mined from the real PPI
network, point ``--patterns`` at a pickled list of (x, edge_index,
edge_weight, label) tuples (see ``mine_patterns.py``).

Usage:
    python pretrain.py                 # synthetic patterns, CPU-friendly
    python pretrain.py --real patterns.pt
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from config import PreTrainCfg, PromptCfg, CKPT_ROOT, LOG_ROOT, RunCfg
from model import L3PPISurrogate, SyntheticPatternDataset, pattern_collate
from utils import set_seed, Logger, num_params, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="path to mined patterns .pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--gnn", default=None, help="gin | gcn")
    ap.add_argument("--name", default=None, help="save name override")
    args = ap.parse_args()

    cfg = PreTrainCfg(); pcfg = PromptCfg(); run = RunCfg()
    epochs = args.epochs or cfg.epochs
    bs = args.batch_size or cfg.batch_size
    log = Logger(LOG_ROOT / "pretrain.log")
    log.header("L3-PPI  Stage-1  Surrogate Pre-training")
    set_seed(run.seed)
    device = pick_device(run.device)
    log.log(f"device={device}  epochs={epochs}  bs={bs}")

    surrogate = L3PPISurrogate(pcfg.node_dim, pcfg, gnn_name=args.gnn or cfg.gnn).to(device)
    opt = torch.optim.Adam(surrogate.parameters(), lr=args.lr or cfg.lr,
                           weight_decay=cfg.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()

    if args.real:
        log.log(f"loading REAL patterns from {args.real}")
        data = torch.load(args.real)
        dataset = data  # expect list of tuples
        def collate(items):
            xs, eis, ews, labels, batch = [], [], [], [], []
            off = 0
            for x, ei, ew, lab in items:
                N = x.size(0); xs.append(x); ews.append(ew); eis.append(ei + off)
                batch.append(torch.full((N,), len(labels), dtype=torch.long))
                labels.append(torch.as_tensor(lab, dtype=torch.float)); off += N
            return (torch.cat(xs), torch.cat(eis, dim=1), torch.cat(ews), torch.cat(batch)), torch.stack(labels)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate)
    else:
        loader = DataLoader(SyntheticPatternDataset(pcfg.K, pcfg.node_dim),
                            batch_size=bs, shuffle=True, collate_fn=pattern_collate)

    log.log(f"surrogate params (total): {num_params(surrogate)}")
    best = 1e9
    for ep in range(1, epochs + 1):
        surrogate.train()
        tot, n = 0.0, 0
        for (x, ei, ew, bat), lab in loader:
            x, ei, ew, bat, lab = [t.to(device) for t in (x, ei, ew, bat, lab)]
            opt.zero_grad()
            logit = surrogate(x, ei, ew, bat)
            loss = bce(logit, lab.float())
            loss.backward()
            opt.step()
            tot += loss.item() * lab.size(0); n += lab.size(0)
        train_loss = tot / max(n, 1)

        # quick eval on the same loader (synthetic == train==test distribution)
        surrogate.eval()
        with torch.no_grad():
            corr = 0; nn_ = 0
            for (x, ei, ew, bat), lab in loader:
                x, ei, ew, bat, lab = [t.to(device) for t in (x, ei, ew, bat, lab)]
                pred = (torch.sigmoid(surrogate(x, ei, ew, bat)) >= 0.5).float()
                corr += (pred == lab).sum().item(); nn_ += lab.size(0)
        acc = corr / max(nn_, 1)
        log.log(f"[{ep:3d}/{epochs}] loss={train_loss:.4f}  acc={acc:.4f}")
        if train_loss < best:
            best = train_loss
            torch.save(surrogate.state_dict(),
                       CKPT_ROOT / (args.name or cfg.save_name))
    log.log(f"saved best surrogate -> {CKPT_ROOT / (args.name or cfg.save_name)}")


if __name__ == "__main__":
    main()
