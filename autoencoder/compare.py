"""
对比脚本：需先运行 classic_autoencoder.py 与 conv_autoencoder.py
运行: python compare.py
输出: results_compare/ 下的并排重构图、合并损失曲线、PSNR 对比（含平移 4px）、metrics.json
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classic_autoencoder import ClassicAutoencoder
from conv_autoencoder import ConvAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results_compare"


def get_test_loader():
    test = datasets.CIFAR10("data", train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(test, batch_size=128, shuffle=False, num_workers=2)


@torch.no_grad()
def evaluate(model, loader, shift=0):
    """shift>0 时先对图像做循环平移再重构，测位置鲁棒性"""
    model.eval()
    mses, psnrs = [], []
    for x, _ in loader:
        x = x.to(DEVICE)
        if shift:
            x = torch.roll(x, shifts=(shift,shift), dims=(2, 3))
        out = model(x)
        mse = ((out - x) ** 2).mean(dim=(1, 2, 3))
        mses.append(mse)
        psnrs.append(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))
    return torch.cat(mses).mean().item(), torch.cat(psnrs).mean().item()


@torch.no_grad()
def save_side_by_side(models, test_loader, path, n=10):
    x = next(iter(test_loader))[0][:n]       # shuffle=False，两个模型看同一组图
    fig, axes = plt.subplots(1 + len(models), n, figsize=(n * 1.5, (1 + len(models)) * 1.5))
    for i in range(n):
        axes[0, i].imshow(x[i].permute(1, 2, 0))
    for r, (name, m) in enumerate(models.items(), start=1):
        recon = m(x.to(DEVICE)).cpu()
        for i in range(n):
            axes[r, i].imshow(recon[i].permute(1, 2, 0))
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=8)
    for r, name in enumerate(models, start=1):
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle("Original vs Classic AE vs CNN AE", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_loss_curves(hist_paths, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, p in hist_paths.items():
        with open(p) as f:
            h = json.load(f)
        ax.plot(h["train"], label=f"{name} (train)")
        ax.plot(h["test"], "--", label=f"{name} (test)")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Training / test loss")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_psnr_bars(metrics, path):
    names = list(metrics.keys())
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - w/2, [metrics[n]["test_psnr"] for n in names], w, label="original")
    b2 = ax.bar(x + w/2, [metrics[n]["psnr_shift4"] for n in names], w, label="shifted by 4 px")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("PSNR (dB)"); ax.legend()
    ax.set_title("Reconstruction quality & translation robustness")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    test_loader = get_test_loader()

    models = {
        "Classic AE": ClassicAutoencoder(),
        "CNN AE": ConvAutoencoder(),
    }
    models["Classic AE"].load_state_dict(torch.load("results_classic/classic_ae.pth", map_location=DEVICE))
    models["CNN AE"].load_state_dict(torch.load("results_conv/conv_ae.pth", map_location=DEVICE))
    for m in models.values():
        m.to(DEVICE).eval()

    metrics = {}
    for name, m in models.items():
        mse, psnr = evaluate(m, test_loader)
        _, psnr_shift = evaluate(m, test_loader, shift=4)
        metrics[name] = {"params": sum(p.numel() for p in m.parameters()),
                         "test_mse": round(mse, 5),
                         "test_psnr": round(psnr, 2),
                         "psnr_shift4": round(psnr_shift, 2),
                         "psnr_drop": round(psnr - psnr_shift, 2)}
        print(name, metrics[name])

    save_side_by_side(models, test_loader, f"{OUT_DIR}/side_by_side.png")
    save_psnr_bars(metrics, f"{OUT_DIR}/psnr_robustness.png")
    try:
        save_loss_curves({"Classic AE": "results_classic/history.json",
                          "CNN AE": "results_conv/history.json"},
                         f"{OUT_DIR}/loss_curves.png")
    except FileNotFoundError as e:
        print(f"跳过损失曲线（缺 history.json，请给 classic 版打补丁后重跑）: {e}")

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n对比结果已保存至 {OUT_DIR}/")


if __name__ == "__main__":
    main()
