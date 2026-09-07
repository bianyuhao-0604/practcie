"""
隐空间 t-SNE 可视化：验证 H3（CNN AE 的 latent 按语义类别聚类，Classic AE 则否）
前置: results_classic/classic_ae.pth 与 results_conv/conv_ae.pth
运行: python latent_tsne.py
输出: results_compare/latent_tsne.png （附 tsne_cache.npz 缓存）
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classic_autoencoder import ClassicAutoencoder
from conv_autoencoder import ConvAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results_compare"
N_SAMPLES = 2000      # t-SNE 样本数
PERPLEXITY = 30


def get_test_loader():
    test = datasets.CIFAR10("data", train=False, download=True,
                            transform=transforms.ToTensor())
    return DataLoader(test, batch_size=128, shuffle=False, num_workers=2)


@torch.no_grad()
def collect_latents(model, loader, n):
    model.eval()
    zs, ys = [], []
    total = 0
    for x, y in loader:
        zs.append(model.encoder(x.to(DEVICE)).cpu())
        ys.append(y)
        total += zs[-1].size(0)
        if total >= n:
            break
    return torch.cat(zs)[:n].numpy(), torch.cat(ys)[:n].numpy()


def run_tsne(z):
    return TSNE(n_components=2, perplexity=PERPLEXITY, init="pca",
                learning_rate="auto", random_state=42).fit_transform(z)


def plot_tsne(z2, labels, name, ax):
    """绘制散点图，并返回 scatter 句柄（colorbar 需要它）"""
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=labels,
                    cmap="tab10", s=4, alpha=0.7)
    ax.set_title(f"{name}: latent space (t-SNE)")
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = f"{OUT_DIR}/tsne_cache.npz"

    if os.path.exists(cache):        # 命中缓存：跳过昂贵的 t-SNE 计算
        print("检测到缓存，直接重绘")
        data = np.load(cache)
        embeddings = {"Classic AE": data["classic"], "CNN AE": data["cnn"]}
        labels = data["labels"]
    else:
        loader = get_test_loader()
        models = {"Classic AE": ClassicAutoencoder(),
                  "CNN AE": ConvAutoencoder()}
        models["Classic AE"].load_state_dict(
            torch.load("results_classic/classic_ae.pth", map_location=DEVICE))
        models["CNN AE"].load_state_dict(
            torch.load("results_conv/conv_ae.pth", map_location=DEVICE))
        for m in models.values():
            m.to(DEVICE).eval()

        embeddings, labels = {}, None
        for name, m in models.items():
            print(f"提取 {name} 的隐向量...")
            z, labels = collect_latents(m, loader, N_SAMPLES)
            print(f"  t-SNE 降维中（{N_SAMPLES} 个样本）...")
            embeddings[name] = run_tsne(z)
        np.savez(cache, classic=embeddings["Classic AE"],
                 cnn=embeddings["CNN AE"], labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sc = None
    for ax, (name, z2) in zip(axes, embeddings.items()):
        sc = plot_tsne(z2, labels, name, ax)

    fig.colorbar(sc, ax=list(axes), ticks=range(10),
                 label="class (0=plane 1=car 2=bird 3=cat 4=deer "
                       "5=dog 6=frog 7=horse 8=ship 9=truck)")
    fig.savefig(f"{OUT_DIR}/latent_tsne.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {OUT_DIR}/latent_tsne.png")


if __name__ == "__main__":
    main()
