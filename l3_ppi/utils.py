"""Utility helpers: reproducibility, device selection, metrics, logging."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def num_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #
@torch.no_grad()
def binary_metrics(logits, target, threshold=0.5):
    """logits: raw (B,) ; target: (B,) long/bool. returns dict."""
    prob = torch.sigmoid(logits).cpu()
    pred = (prob >= threshold).long()
    t = target.long().cpu()
    tp = ((pred == 1) & (t == 1)).sum().item()
    fp = ((pred == 1) & (t == 0)).sum().item()
    tn = ((pred == 0) & (t == 0)).sum().item()
    fn = ((pred == 0) & (t == 1)).sum().item()
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    # AUROC / AUPRC
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        auroc = roc_auc_score(t.numpy(), prob.numpy())
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(t.numpy(), prob.numpy())
    except Exception:
        auprc = float("nan")
    return dict(acc=acc, precision=prec, recall=rec, f1=f1,
                auroc=auroc, auprc=auprc, tp=tp, fp=fp, tn=tn, fn=fn)


class Logger:
    def __init__(self, path=None):
        self.path = path
        self.lines = []

    def log(self, msg):
        print(msg, flush=True)
        self.lines.append(msg)
        if self.path:
            with open(self.path, "a",encoding='utf-8') as f:
                f.write(msg + "\n")

    def header(self, s):
        self.log("=" * 60)
        self.log(s)
        self.log("=" * 60)
