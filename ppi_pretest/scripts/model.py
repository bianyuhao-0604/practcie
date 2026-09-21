# model.py — 逐残基双塔: esm2 / prostt5 各自独立掩码注意力池化 + 全局向量塔 + 对称交互头
import torch
import torch.nn as nn


class ResidueTower(nn.Module):
    """逐残基投影 + 掩码注意力池化 → (B, d_fuse)。mask 屏蔽 padding, 只聚合真实 token。"""

    def __init__(self, d_in, d_fuse=128, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(p),
            nn.Linear(512, d_fuse), nn.GELU(),
        )
        self.attn = nn.Linear(d_fuse, 1)

    def forward(self, x, mask):                      # x [B,T,D], mask [B,T] bool
        h = self.proj(x)                            # [B,T,d_fuse]
        s = self.attn(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)                 # [B,T]
        return torch.einsum("bt,btd->bd", a, h)     # [B,d_fuse]


class VecTower(nn.Module):
    """蛋白级向量塔 (text/genome)"""

    def __init__(self, d_in, d_fuse=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(p),
            nn.Linear(512, d_fuse), nn.GELU(), nn.Dropout(p),
        )

    def forward(self, x):
        return self.net(x)


def _interact(za, zb):
    return torch.cat([(za - zb).abs(), za * zb, za + zb], dim=-1)


class PPIModel(nn.Module):
    """tower_seq(esm2) [+ tower_str(prostt5)] 池化拼接 [+ tower_glob(text+genome)] → 对称交互头
    两残基塔使用各自的 lengths 掩码, 不要求 token 级对齐
    (prostt5 分词器附加 <AA2fold>/</s>, 长度天然比 esm2 多)。"""

    def __init__(self, flags, d_seq=480, d_str=1024, d_txt=384, d_gen=64,
                 d_fuse=128, p=0.2):
        super().__init__()
        self.flags = flags
        self.use_struct = bool(flags["use_struct"])
        self.tower_seq = ResidueTower(d_seq, d_fuse, p)
        self.tower_str = ResidueTower(d_str, d_fuse, p) if self.use_struct else None
        d_glob = (d_txt if flags["use_text"] else 0) + \
                 (d_gen if flags["use_genome"] else 0)
        self.use_global = d_glob > 0
        self.tower_glob = VecTower(d_glob, d_fuse, p) if self.use_global else None
        d_rep = d_fuse * (2 if self.use_struct else 1) + \
                (d_fuse if self.use_global else 0)
        self.head = nn.Sequential(
            nn.Linear(3 * d_rep, 256), nn.GELU(), nn.Dropout(p),
            nn.Linear(256, 1),
        )

    def forward(self, xsA, xsB, msA, msB, xpA=None, xpB=None, mpA=None, mpB=None,
                ga=None, gb=None):
        za = self.tower_seq(xsA, msA)
        zb = self.tower_seq(xsB, msB)
        if self.tower_str is not None:
            za = torch.cat([za, self.tower_str(xpA, mpA)], dim=-1)
            zb = torch.cat([zb, self.tower_str(xpB, mpB)], dim=-1)
        feat = _interact(za, zb)
        if self.use_global:
            feat = torch.cat([feat, _interact(self.tower_glob(ga),
                                              self.tower_glob(gb))], dim=-1)
        return self.head(feat).squeeze(-1)
