"""
full_evaluation.py — Classic AE vs CNN AE 全面性能对比
覆盖: 重建质量(MSE/PSNR/SSIM/LPIPS) + 误差热力图 + 分类别PSNR
     + 效率(参数量/GMACs/延迟) + 隐空间(插值/维度利用率)
     + 下游任务(线性探针分类 + 混淆矩阵)
前置: results_classic/classic_ae.pth 与 results_conv/conv_ae.pth
运行: python full_evaluation.py
输出: results_full/ 下的 summary.md、metrics.json 与 6 张图
"""
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classic_autoencoder import ClassicAutoencoder
from conv_autoencoder import ConvAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results_full"
CLASSES = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
IN_MAIN = (__name__ == "__main__")   # DataLoader worker 也会 import 本模块,
                                     # 仅主进程打印警告, 避免刷屏

# ---- 可选依赖: 缺失时降级为 None 并跳过, 不阻塞其余指标 ----
try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
except ImportError:
    ssim_fn = None
    if IN_MAIN:
        print("警告: 缺 torchmetrics, 跳过 SSIM (pip install torchmetrics)")

try:
    import lpips
    _lpips = lpips.LPIPS(net="alex").to(DEVICE).eval()
    for p in _lpips.parameters():
        p.requires_grad = False
except Exception as e:                      # 含首次下载权重失败
    _lpips = None
    if IN_MAIN:
        print(f"警告: LPIPS 不可用({e}), 跳过 (pip install lpips)")

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None
    if IN_MAIN:
        print("警告: 缺 ptflops, 跳过 FLOPs (pip install ptflops)")


def get_loaders():
    tf = transforms.ToTensor()
    train = datasets.CIFAR10("data", train=True, download=True, transform=tf)
    test = datasets.CIFAR10("data", train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=128, shuffle=False, num_workers=2),
            DataLoader(test, batch_size=128, shuffle=False, num_workers=2))


# ============ 1. 重建质量: MSE / PSNR / SSIM / LPIPS + 分类别 PSNR ============
@torch.no_grad()
def reconstruction_metrics(model, loader):
    model.eval()
    sums = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    cls_psnr = torch.zeros(10)
    cls_cnt = torch.zeros(10)
    n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        mse_per = ((out - x) ** 2).mean(dim=(1, 2, 3))
        psnr_per = 10 * torch.log10(1.0 / mse_per.clamp(min=1e-10))
        b = x.size(0)
        sums["mse"] += mse_per.sum().item()
        sums["psnr"] += psnr_per.sum().item()
        if ssim_fn is not None:                        # 批均值 × 批大小还原为和
            sums["ssim"] += ssim_fn(out, x, data_range=1.0).item() * b
        if _lpips is not None:                         # LPIPS 要求输入 [-1,1]
            sums["lpips"] += _lpips(out * 2 - 1, x * 2 - 1).view(-1).sum().item()
        for c in range(10):                            # 分类别 PSNR
            m = (y == c)
            cls_psnr[c] += psnr_per[m].sum().cpu()
            cls_cnt[c] += m.sum().cpu()
        n += b
    metrics = {k: v / n for k, v in sums.items()}
    metrics["per_class_psnr"] = (cls_psnr / cls_cnt).tolist()
    return metrics


# ============ 2. 误差热力图 ============
@torch.no_grad()
def save_error_heatmap(model, x, path, name, n=8):
    model.eval()
    recon = model(x[:n].to(DEVICE)).cpu()
    orig = x[:n]
    err = (recon - orig).abs().mean(dim=1)             # 通道均值 → (n,H,W)
    vmax = err.max()
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4.8))
    for i in range(n):
        axes[0, i].imshow(orig[i].permute(1, 2, 0))
        axes[1, i].imshow(recon[i].permute(1, 2, 0))
        im = axes[2, i].imshow(err[i], cmap="hot", vmin=0, vmax=vmax)
    for r, lab in enumerate(["original", "recon", "|error|"]):
        axes[r, 0].set_ylabel(lab, fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[2, :].tolist(), shrink=0.9, pad=0.02)
    fig.suptitle(f"{name}: reconstruction error heatmap", y=1.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============ 3. 分类别 PSNR 柱状图 ============
def save_per_class_chart(per_class, path):
    x = np.arange(10); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w/2, per_class["Classic AE"], w, label="Classic AE")
    b2 = ax.bar(x + w/2, per_class["CNN AE"], w, label="CNN AE")
    ax.bar_label(b1, fmt="%.1f", fontsize=7); ax.bar_label(b2, fmt="%.1f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_ylabel("PSNR (dB)"); ax.legend()
    ax.set_title("Per-class reconstruction quality")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============ 4. 效率: FLOPs / 延迟 (参数量在 main 里直接数) ============
@torch.no_grad()
def efficiency_stats(model, loader):
    out = {"gmacs": None, "ms_per_img": None}
    if get_model_complexity_info is not None:
        model = model.to("cpu")                        # ptflops 需 CPU 输入
        macs, _ = get_model_complexity_info(
            model, (3, 32, 32), print_per_layer_stat=False,
            as_strings=False, verbose=False)
        model = model.to(DEVICE)
        if macs:
            out["gmacs"] = round(macs / 1e9, 4)
    model.eval()                                       # 推理延迟
    x = next(iter(loader))[0][:256].to(DEVICE)
    with torch.no_grad():
        for _ in range(20):
            model(x)                                   # 预热
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    out["ms_per_img"] = round((time.time() - t0) / 100 / x.size(0) * 1000, 4)
    return out


# ============ 5. 隐空间插值 (cat → ship) ============
@torch.no_grad()
def save_interpolation(models, loader, path, c0=3, c1=8,
                       alphas=(0, .25, .5, .75, 1.0)):
    x0 = x1 = None                                     # 找两类的第一张测试图
    for xb, yb in loader:
        if x0 is None and (yb == c0).any():
            x0 = xb[(yb == c0).nonzero()[0].item()]
        if x1 is None and (yb == c1).any():
            x1 = xb[(yb == c1).nonzero()[0].item()]
        if x0 is not None and x1 is not None:
            break
    x0, x1 = x0[None].to(DEVICE), x1[None].to(DEVICE)

    fig, axes = plt.subplots(len(models), len(alphas),
                             figsize=(len(alphas) * 1.6, len(models) * 1.8))
    for r, (name, m) in enumerate(models.items()):
        m.eval()
        z0, z1 = m.encoder(x0), m.encoder(x1)
        for c_idx, a in enumerate(alphas):
            out = m.decoder(a * z1 + (1 - a) * z0).cpu()
            if out.dim() == 2:                         # Classic AE: (B,3072) 展平输出
                out = out.view(-1, 3, 32, 32)          #   补上 forward 里那步 reshape
            img = out[0]
            ax = axes[r, c_idx]
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"α={a}", fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle("Latent interpolation: cat (α=0) → ship (α=1)", y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============ 6. 维度利用率 ============
@torch.no_grad()
def dimension_stats(model, loader, n=2000):
    model.eval()
    zs = []
    total = 0
    for x, _ in loader:
        zs.append(model.encoder(x.to(DEVICE)).cpu())
        total += zs[-1].size(0)
        if total >= n:
            break
    z = torch.cat(zs)[:n]
    std = z.std(dim=0)
    pr = (std.sum() ** 2 / (std ** 2).sum()).item()    # participation ratio
    eff = int((std > 0.01 * std.max()).sum())
    return {"std": std.numpy(), "participation_ratio": round(pr, 1),
            "effective_dims": eff}


def save_dim_chart(dim_stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, d) in zip(axes, dim_stats.items()):
        ax.bar(range(128), np.sort(d["std"])[::-1])
        ax.set_title(f"{name}: PR={d['participation_ratio']}, "
                     f"eff_dims={d['effective_dims']}/128")
        ax.set_xlabel("dimension (sorted by std)"); ax.set_ylabel("std")
    fig.suptitle("Latent dimension utilization", y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============ 7. 线性探针下游分类 ============
@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        zs.append(model.encoder(x.to(DEVICE)).cpu())
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def train_probe(model, train_loader, test_loader, epochs=30):
    Xtr, ytr = extract_features(model, train_loader)   # 冻结编码器, 只训线性层
    Xte, yte = extract_features(model, test_loader)
    torch.manual_seed(42)
    clf = nn.Linear(128, 10)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(Xtr.size(0))
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            loss = F.cross_entropy(clf(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = clf(Xte).argmax(1)
        acc = (pred == yte).float().mean().item()
        cm = confusion_matrix(yte.numpy(), pred.numpy())
    return acc, cm


def save_confusion(cms, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, (acc, cm)) in zip(axes, cms.items()):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, fontsize=7)
        ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_title(f"{name}: linear probe acc={acc:.4f}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============ 8. 汇总表 ============
def write_summary(results, path_md, path_json):
    rows = [
        ("MSE ↓",              "mse",        "{:.5f}"),
        ("PSNR (dB) ↑",        "psnr",       "{:.2f}"),
        ("SSIM ↑",             "ssim",       "{:.4f}"),
        ("LPIPS ↓",            "lpips",      "{:.4f}"),
        ("参数量",              "params",     "{:,}"),
        ("GMACs",              "gmacs",      "{:.4f}"),
        ("延迟 ms/张",          "ms_per_img", "{:.4f}"),
        ("线性探针准确率 ↑",     "probe_acc",  "{:.4f}"),
        ("有效维度 PR/128",     "pr",         "{:.1f}"),
    ]
    c, k = results["Classic AE"], results["CNN AE"]
    lines = ["| 指标 | Classic AE | CNN AE | Δ (CNN−Classic) |",
             "|---|---|---|---|"]
    for label, key, fmt in rows:
        v1, v2 = c.get(key), k.get(key)
        s1 = fmt.format(v1) if v1 is not None else "—"
        s2 = fmt.format(v2) if v2 is not None else "—"
        d = fmt.format(v2 - v1) if (v1 is not None and v2 is not None) else "—"
        lines.append(f"| {label} | {s1} | {s2} | {d} |")
    table = "\n".join(lines)
    with open(path_md, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n" + table)


# ============ 主流程 ============
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_loader, test_loader = get_loaders()

    models = {"Classic AE": ClassicAutoencoder(), "CNN AE": ConvAutoencoder()}
    models["Classic AE"].load_state_dict(
        torch.load("results_classic/classic_ae.pth", map_location=DEVICE))
    models["CNN AE"].load_state_dict(
        torch.load("results_conv/conv_ae.pth", map_location=DEVICE))
    for m in models.values():
        m.to(DEVICE).eval()

    results, per_class, dim_stats, cms = {}, {}, {}, {}
    heat_batch = next(iter(test_loader))[0][:8]

    for name, m in models.items():
        print(f"\n>>> 评估 {name} ...")
        rec = reconstruction_metrics(m, test_loader)
        eff = efficiency_stats(m, test_loader)
        dim = dimension_stats(m, test_loader)
        acc, cm = train_probe(m, train_loader, test_loader)

        results[name] = {
            "mse": round(rec["mse"], 5), "psnr": round(rec["psnr"], 2),
            "ssim": round(rec["ssim"], 4) if ssim_fn else None,
            "lpips": round(rec["lpips"], 4) if _lpips else None,
            "params": sum(p.numel() for p in m.parameters()),
            "gmacs": eff["gmacs"], "ms_per_img": eff["ms_per_img"],
            "probe_acc": round(acc, 4),
            "pr": dim["participation_ratio"],
            "effective_dims": dim["effective_dims"],
            "per_class_psnr": [round(v, 2) for v in rec["per_class_psnr"]],
        }
        per_class[name] = rec["per_class_psnr"]
        dim_stats[name] = dim
        cms[name] = (acc, cm)
        save_error_heatmap(m, heat_batch, f"{OUT_DIR}/error_heatmap_"
                           f"{'classic' if 'Classic' in name else 'cnn'}.png", name)
        print(f"  {results[name]}")

    save_per_class_chart(per_class, f"{OUT_DIR}/per_class_psnr.png")
    save_interpolation(models, test_loader, f"{OUT_DIR}/latent_interp.png")
    save_dim_chart(dim_stats, f"{OUT_DIR}/dim_utilization.png")
    save_confusion(cms, f"{OUT_DIR}/probe_confusion.png")
    write_summary(results, f"{OUT_DIR}/summary.md", f"{OUT_DIR}/metrics.json")
    print(f"\n全部结果已保存至 {OUT_DIR}/")


if __name__ == "__main__":
    main()
