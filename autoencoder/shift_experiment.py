"""
扩展平移实验：在 {2, 4, 6, 8, 12, 16} 像素位移下测两模型 PSNR，
验证 H2 的细化版——Classic AE 的表观稳健是否会在更大位移下崩塌
前置: 已训练好两个模型的 .pth
运行: python shift_experiment.py
输出: results_compare/shift_curve.png + shift_metrics.json
"""
import json
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classic_autoencoder import ClassicAutoencoder
from conv_autoencoder import ConvAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results_compare"
SHIFTS = [0, 2, 4, 6, 8, 12, 16]


def get_test_loader():
    test = datasets.CIFAR10("data", train=False, download=True,
                            transform=transforms.ToTensor())
    return DataLoader(test, batch_size=128, shuffle=False, num_workers=2)


@torch.no_grad()
def evaluate(model, loader, shift):
    model.eval()
    mses, psnrs = [], []
    for x, _ in loader:
        x = x.to(DEVICE)
        if shift > 0:
            x = torch.roll(x, shifts=(shift, shift), dims=(2, 3))
        out = model(x)
        mse = ((out - x) ** 2).mean(dim=(1, 2, 3))  # 占位，下一行替换
        mses.append(mse)
        psnrs.append(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))
    return torch.cat(mses).mean().item(), torch.cat(psnrs).mean().item()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    loader = get_test_loader()
    models = {"Classic AE": ClassicAutoencoder(), "CNN AE": ConvAutoencoder()}
    models["Classic AE"].load_state_dict(
        torch.load("results_classic/classic_ae.pth", map_location=DEVICE))
    models["CNN AE"].load_state_dict(
        torch.load("results_conv/conv_ae.pth", map_location=DEVICE))
    for m in models.values():
        m.to(DEVICE).eval()

    results = {n: {} for n in models}
    for name, m in models.items():
        print(f"评估 {name} ...")
        for s in SHIFTS:
            mse, psnr = evaluate(m, loader, s)
            results[name][s] = round(psnr, 2)
            print(f"  shift={s:2d}px  PSNR={psnr:.2f} dB")

    with open(f"{OUT_DIR}/shift_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in results.items():
        ax.plot(SHIFTS, [r[s] for s in SHIFTS], marker="o", label=name)
    ax.set_xlabel("translation (pixels, wrap-around)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs. translation magnitude")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(f"{OUT_DIR}/shift_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {OUT_DIR}/shift_curve.png 与 shift_metrics.json")


if __name__ == "__main__":
    main()
