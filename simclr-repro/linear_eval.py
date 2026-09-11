# linear_eval.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

from model import SimCLR
from augmentations import CIFAR_MEAN, CIFAR_STD


@torch.no_grad()
def extract_features(model: SimCLR, loader: DataLoader, device: str):
    """抽投影头之前的 h —— 论文 Table 3 显示 h 比下游用 z 好得多。"""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        _, h = model(x, return_h=True)
        feats.append(h.float().cpu()); labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def linear_eval(model: SimCLR, args, device: str) -> float:
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_ds = datasets.CIFAR10(args.data, train=True,  download=True, transform=eval_tf)
    test_ds  = datasets.CIFAR10(args.data, train=False, download=True, transform=eval_tf)
    tl = DataLoader(train_ds, batch_size=512, num_workers=8, pin_memory=True)
    el = DataLoader(test_ds,  batch_size=512, num_workers=8, pin_memory=True)

    X_tr, y_tr = extract_features(model, tl, device)
    X_te, y_te = extract_features(model, el, device)
    # 论文线性评估用 ℓ2 正则的逻辑回归；这里用带 wd 的线性层 + Adam 等价实现
    X_tr = F.normalize(X_tr, dim=1); X_te = F.normalize(X_te, dim=1)

    clf = nn.Linear(X_tr.size(1), 10).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.eval_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.eval_epochs)

    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)

    for ep in range(args.eval_epochs):
        clf.train()
        perm = torch.randperm(X_tr.size(0), device=device)
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            loss = F.cross_entropy(clf(X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    clf.eval()
    with torch.no_grad():
        pred = clf(X_te).argmax(1)
    return (pred == y_te).float().mean().item() * 100


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./datasets")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_epochs", type=int, default=100)
    p.add_argument("--eval_lr", type=float, default=1e-3)
    args = p.parse_args()

    device = "cuda:0"
    model = SimCLR("resnet18", "cifar10").to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    # 关键：冻结 encoder（这里直接整模型 eval + no_grad，且不更新其参数）
    for prm in model.parameters():
        prm.requires_grad = False

    acc = linear_eval(model, args, device)
    print(f"\nLinear evaluation Top-1: {acc:.2f}%")
