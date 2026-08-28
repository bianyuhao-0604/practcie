"""Stage 2: fine-tune the L3-PPI prompt on a PPI dataset.

Strategy: freeze the pre-trained surrogate AND the protein encoder; train
only the prompt nodes and the path gate.  This is the paper's prompt-tuning
setup.  ℒ = BCE + λ·ℒ_PN (Eq. 4), with the Gumbel temperature τ annealed
from ``tau_start`` to ``tau_end``.

Usage:
    python finetune.py                              # yeast + CNN baseline
    python finetune.py --encoder esm --dataset shs27k
    python finetune.py --no-pretrain                # ablate: train surrogate from scratch
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from config import (DataCfg, EncoderCfg, PromptCfg, FineTuneCfg, PreTrainCfg,
                    RunCfg, CKPT_ROOT, LOG_ROOT)
from dataset import build_loaders, collate
from encoder import build_encoder
from model import L3PPI, finetune_loss,L3PPISurrogate
from utils import set_seed, Logger, num_params, pick_device, binary_metrics


def anneal_tau(epoch, cfg: FineTuneCfg):
    if cfg.tau_anneal_epochs <= 0:
        return cfg.tau_start
    r = min(1.0, epoch / cfg.tau_anneal_epochs)
    return cfg.tau_start * (1 - r) + cfg.tau_end * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="yeast | shs27k | string")
    ap.add_argument("--encoder", default=None, help="esm | cnn")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--reg-weight", type=float, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--no-pretrain", action="store_true", help="ablate surrogate pre-training")
    ap.add_argument("--split", default=None, help="random | bfs | dfs")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    # apply CLI overrides onto config singletons
    dc, ec, pc, fc, run = DataCfg, EncoderCfg, PromptCfg, FineTuneCfg, RunCfg
    if args.dataset: dc.name = args.dataset
    if args.encoder: ec.use_esm = (args.encoder == "esm")
    if args.split: dc.split = args.split
    if args.K: pc.K = args.K
    if args.gamma: pc.gamma = args.gamma
    if args.reg_weight is not None: fc.reg_weight = args.reg_weight
    epochs = args.epochs or fc.epochs
    bs = args.batch_size or fc.batch_size

    log = Logger(LOG_ROOT / f"finetune_{dc.name}.log")
    log.header(f"L3-PPI  Stage-2  Fine-tuning  ({dc.name}, {dc.split} split)")
    set_seed(run.seed)
    device = pick_device(run.device)
    log.log(f"device={device}")

    # ---- data ----
    loaders = build_loaders(dc, run)
    log.log(f"meta={loaders['meta']}")

    # ---- models ----
    encoder, is_esm = build_encoder(ec, device)
    if is_esm:
        # load pre-computed ESM embeddings if available (see precompute_esm.py)
        from config import CACHE_ROOT
        emb_path = CACHE_ROOT / f"esm_emb_{dc.name}.pt"
        if not encoder.load_cache(emb_path):
            log.log(f"WARNING: no pre-computed ESM cache at {emb_path}; "
                    "ESM will run per-batch (slow). Run precompute_esm.py first.")
    surrogate = L3PPISurrogate(pc.node_dim, pc).to(device)

    if not args.no_pretrain:
        # surrogate weights come ONLY from Stage-1 pretraining; l3ppi.pt is the
        # whole-model checkpoint of a previous fine-tune and must not be loaded
        # into the bare surrogate (key prefix mismatch).
        ckpt = CKPT_ROOT / PreTrainCfg.save_name
        if ckpt.exists():
            surrogate.load_state_dict(torch.load(ckpt, map_location=device))
            log.log(f"loaded pre-trained surrogate: {ckpt}")
        else:
            log.log("WARNING: no surrogate checkpoint found; training from scratch. "
                    "Run `python pretrain.py` first for the full method.")

    model = L3PPI(encoder, surrogate, pc).to(device)

    # ---- freeze everything except prompt + gate ----
    for p in model.surrogate.parameters():
        p.requires_grad = False
    if is_esm and ec.freeze:
        for p in model.encoder.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    log.log(f"trainable params: {num_params(model)}  (prompt+gate only)")

    opt = torch.optim.Adam([
        {"params": [model.prompt_nodes], "lr": args.lr or fc.lr},
        {"params": model.gate.parameters(), "lr": fc.gate_lr},
    ], weight_decay=fc.weight_decay)

    def run_epoch(loader, train=True):
        model.train(train)
        tot = tot_bce = tot_pn = 0.0
        all_logits, all_lab = [], []
        for step, batch in enumerate(loader):
            label = batch["label"].to(device)
            out = model(batch, tau=anneal_tau(epoch, fc) if train else fc.tau_end)
            losses = finetune_loss(out, label, pc)
            if train:
                opt.zero_grad(); losses["total"].backward(); opt.step()
            tot += losses["total"].item() * label.size(0)
            tot_bce += losses["bce"].item() * label.size(0)
            tot_pn += losses["pn"].item() * label.size(0)
            all_logits.append(out["logit"].detach()); all_lab.append(label.detach())
        logits = torch.cat(all_logits); lab = torch.cat(all_lab)
        m = binary_metrics(logits, lab)
        n = lab.size(0)
        return dict(total=tot / n, bce=tot_bce / n, pn=tot_pn / n, **m)

    if args.eval_only:
        res = run_epoch(loaders["test"], train=False)
        log.log("EVAL-ONLY  " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()))
        return

    best = -1.0
    bad = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr = run_epoch(loaders["train"], train=True)
        va = run_epoch(loaders["val"], train=False)
        log.log(f"[{epoch:3d}/{epochs}] τ={anneal_tau(epoch, fc):.2f} "
                f"train: loss={tr['total']:.4f} auroc={tr['auroc']:.4f} | "
                f"val: loss={va['total']:.4f} auroc={va['auroc']:.4f} f1={va['f1']:.4f}")
        if va["auroc"] > best:
            best = va["auroc"]; bad = 0
            torch.save(model.state_dict(), CKPT_ROOT / fc.save_name)
            log.log(f"  ↳ saved (best val AUROC={best:.4f})")
        else:
            bad += 1
            if bad >= fc.early_stop:
                log.log(f"early stop @ {epoch}"); break
    log.log(f"training time: {time.time() - t0:.0f}s")

    # ---- final test with best checkpoint ----
    model.load_state_dict(torch.load(CKPT_ROOT / fc.save_name, map_location=device))
    te = run_epoch(loaders["test"], train=False)
    log.log("TEST  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in te.items()))
    log.log(f"best val AUROC={best:.4f}")


if __name__ == "__main__":
    main()
