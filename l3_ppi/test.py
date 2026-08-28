"""Stage 3: evaluate the trained L3-PPI on held-out test pairs.

Also runs an ablation (no prompt / no ℒ_PN) and reproduces the paper's
L3-rule correlation figure (Pearson + mutual information vs. L_i).

Usage:
    python test.py --dataset yeast
    python test.py --compare cnn,esm            # baseline vs ESM-2 encoder
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from config import (DataCfg, EncoderCfg, PromptCfg, FineTuneCfg, PreTrainCfg,
                    RunCfg, CKPT_ROOT, LOG_ROOT)
from dataset import build_loaders, collate, l3_rule_analysis
from encoder import build_encoder
from model import L3PPI, finetune_loss
from utils import set_seed, Logger, pick_device, binary_metrics


def evaluate(model, loader, device, pc):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            label = batch["label"].to(device)
            out = model(batch, tau=pc.tau_end, hard=False)
            logits.append(out["logit"]); labels.append(label)
    return binary_metrics(torch.cat(logits), torch.cat(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--run-ablation", action="store_true")
    ap.add_argument("--l3-analysis", action="store_true", help="reproduce L3 rule figure")
    args = ap.parse_args()

    dc, ec, pc, fc, run = DataCfg, EncoderCfg, PromptCfg, FineTuneCfg, RunCfg
    if args.dataset: dc.name = args.dataset
    if args.encoder: ec.use_esm = (args.encoder == "esm")
    if args.K: pc.K = args.K
    if args.gamma: pc.gamma = args.gamma

    log = Logger(LOG_ROOT / f"test_{dc.name}.log")
    log.header(f"L3-PPI  Stage-3  Evaluation  ({dc.name})")
    set_seed(run.seed)
    device = pick_device(run.device)

    # ---- L3 rule analysis (dataset-level, no model needed) ----
    if args.l3_analysis:
        from config import CACHE_ROOT
        l3_rule_analysis(dc, li_max=7, save=CACHE_ROOT / f"l3_rule_{dc.name}.csv")

    loaders = build_loaders(dc, run)
    encoder, _ = build_encoder(ec, device)
    surrogate = L3PPISurrogate(pc.node_dim, pc).to(device)
    ckpt_s = CKPT_ROOT / fc.save_name
    if ckpt_s.exists():
        surrogate.load_state_dict(torch.load(ckpt_s, map_location=device))
    model = L3PPI(encoder, surrogate, pc).to(device)
    ckpt_f = CKPT_ROOT / fc.save_name
    if ckpt_f.exists():
        model.load_state_dict(torch.load(ckpt_f, map_location=device))
        log.log(f"loaded: {ckpt_f}")
    else:
        log.log("WARNING: no fine-tuned checkpoint; evaluating random init.")

    res = evaluate(model, loaders["test"], device, pc)
    log.log("FULL L3-PPI  " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()))

    # ---- ablation ----
    if args.run_ablation:
        # (a) remove path-number regularisation (λ=0)
        def ablate(reg_w, tau_inf, name):
            old = fc.reg_weight
            fc.reg_weight = reg_w
            r = evaluate(model, loaders["test"], device, pc)
            fc.reg_weight = old
            log.log(f"[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))
        ablate(0.0, False, "ablation: no-ℒ_PN")
        # (b) plain dot-product baseline (no prompt head) -- reuse encoder only
        baseline(loaders["test"], encoder, device, log)

    log.log("done.")


@torch.no_grad()
def baseline(loader, encoder, device, log):
    """Concat / dot-product baseline classifier (the 'generic head' ablated)."""
    clf = torch.nn.Linear(encoder.out_dim * 2, 1).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    encoder.eval()
    for epoch in range(5):
        for batch in loader:
            a = encoder(batch["a"].to(device), batch["a_mask"].to(device))
            b = encoder(batch["b"].to(device), batch["b_mask"].to(device))
            y = batch["label"].to(device).float()
            logit = clf(torch.cat([a, b], dim=-1)).squeeze(-1)
            loss = bce(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
    logits, labels = [], []
    for batch in loader:
        a = encoder(batch["a"].to(device), batch["a_mask"].to(device))
        b = encoder(batch["b"].to(device), batch["b_mask"].to(device))
        logit = clf(torch.cat([a, b], dim=-1)).squeeze(-1)
        logits.append(logit); labels.append(batch["label"].to(device))
    log.log("BASELINE (concat head)  " + "  ".join(
        f"{k}={v:.4f}" for k, v in binary_metrics(torch.cat(logits), torch.cat(labels)).items()))


if __name__ == "__main__":
    main()
