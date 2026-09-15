import torch

class EmbedAugment:
    """嵌入级组合增强（对应 SimCLR 的 cutout/color-drop + blur/brightness）"""
    def __init__(self, dim_mask_p=0.10, noise_std=0.02, scale_range=(0.9, 1.1)):
        self.p, self.ns, self.sr = dim_mask_p, noise_std, scale_range

    def __call__(self, X):                       # X: (B, D) 已在 GPU
        mask = (torch.rand(X.shape, device=X.device) > self.p).float()
        X1 = X * mask                                          # ① 维度掩码
        std = X1.std(dim=1, keepdim=True).clamp(min=1e-6)
        X1 = X1 + torch.randn_like(X1) * self.ns * std         # ② 相对高斯噪声
        s = torch.empty(X.size(0), 1, device=X.device).uniform_(*self.sr)
        return X1 * s                                          # ③ 幅度缩放
