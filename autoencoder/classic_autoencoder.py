"""
经典（全连接）自编码器 · CIFAR-10
运行: python classic_autoencoder.py
输出: results_classic/ 下的重构对比图、损失曲线、模型权重与指标
"""
import os
import time
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------- 超参数 ----------------
SEED       = 42
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 1e-3
LATENT_DIM = 128          # 瓶颈维度（与后续 CNN AE 对比时保持一致）
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR    = "results_classic"

torch.manual_seed(SEED)


# ---------------- 数据 ----------------
def get_loaders():
    transform = transforms.ToTensor()  # [0,255] -> [0,1]
    train = datasets.CIFAR10("data", train=True,  download=True, transform=transform)
    test  = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2),
            DataLoader(test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2))


# ---------------- 模型 ----------------
class ClassicAutoencoder(nn.Module):
    """对称全连接结构: 3072 -> 1024 -> 256 -> 128(latent) -> 256 -> 1024 -> 3072"""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),                        # (B,3,32,32) -> (B,3072)
            nn.Linear(3072, 1024), nn.ReLU(True),
            nn.Linear(1024, 256),  nn.ReLU(True),
            nn.Linear(256, latent_dim), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(True),
            nn.Linear(256, 1024),      nn.ReLU(True),
            nn.Linear(1024, 3072),     nn.Sigmoid(),   # 输出压回 [0,1]
        )

    def forward(self, x):
        recon = self.decoder(self.encoder(x))
        return recon.view(x.size(0), 3, 32, 32)   # 还原成图像形状


# ---------------- 评估 ----------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_mse, total_psnr, n = 0.0, 0.0, 0
    for x, _ in loader:
        x = x.to(DEVICE)
        recon = model(x)
        total_mse += nn.functional.mse_loss(recon, x).item() * x.size(0)
        # 逐图 PSNR = 10 * log10(1 / MSE)，像素范围为 [0,1]
        per_img_mse = ((recon - x) ** 2).mean(dim=(1, 2, 3))
        psnr = 10 * torch.log10(1.0 / per_img_mse.clamp(min=1e-10))
        total_psnr += psnr.sum().item()
        n += x.size(0)
    return total_mse / n, total_psnr / n


# ---------------- 训练 ----------------
def train(model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train": [], "test": []}
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, _ in train_loader:      # 注意：标签不用，输入即监督信号
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), x)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_mse = running / len(train_loader.dataset)

        test_mse, test_psnr = evaluate(model, test_loader)
        history["train"].append(train_mse)
        history["test"].append(test_mse)
        print(f"epoch {epoch:3d}/{EPOCHS} | train_mse={train_mse:.4f} | "
              f"test_mse={test_mse:.4f} | test_psnr={test_psnr:.2f} dB")

    print(f"训练完成, 总用时 {time.time() - t0:.0f}s")
    return history


# ---------------- 可视化 ----------------
@torch.no_grad()
def save_reconstruction_grid(model, test_loader, path, n=10):
    model.eval()
    x, _ = next(iter(test_loader))
    x = x[:n].to(DEVICE)
    recon = model(x).cpu()
    x = x.cpu()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.5))
    for i in range(n):
        axes[0, i].imshow(x[i].permute(1, 2, 0))
        axes[1, i].imshow(recon[i].permute(1, 2, 0))
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("original",   fontsize=9)
    axes[1, 0].set_ylabel("reconstructed", fontsize=9)
    fig.suptitle("Classic Autoencoder on CIFAR-10", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(history, path):
    with open(f"{OUT_DIR}/history.json", "w") as f:
         json.dump(history, f)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["train"], label="train MSE")
    ax.plot(history["test"], "--", label="test MSE")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.set_title("Classic AE training curve")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------- 主流程 ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_loader, test_loader = get_loaders()

    model = ClassicAutoencoder(LATENT_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,} | 设备: {DEVICE} | 瓶颈维度: {LATENT_DIM} "
          f"(压缩比 3072:{LATENT_DIM} = {3072 // LATENT_DIM}:1)")

    history = train(model, train_loader, test_loader)

    test_mse, test_psnr = evaluate(model, test_loader)
    print(f"\n最终测试集: MSE={test_mse:.4f}, PSNR={test_psnr:.2f} dB")

    save_reconstruction_grid(model, test_loader, f"{OUT_DIR}/reconstruction.png")
    save_loss_curve(history, f"{OUT_DIR}/loss_curve.png")
    torch.save(model.state_dict(), f"{OUT_DIR}/classic_ae.pth")
    print(f"图表与模型权重已保存至 {OUT_DIR}/")


if __name__ == "__main__":
    main()
