import torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm

def _lin(i, o, sn=True):
    return spectral_norm(nn.Linear(i, o)) if sn else nn.Linear(i, o)

class Adapter(nn.Module):
    """可训练适配器：对比损失真正训练的对象，训练后其输出即下游新嵌入 h'"""
    def __init__(self, dim=1280, hidden=1280, p_drop=0.1, spec_norm=True):
        super().__init__()
        self.net = nn.Sequential(
            _lin(dim, hidden, spec_norm), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(p_drop),
            _lin(hidden, dim, spec_norm), nn.LayerNorm(dim))
    def forward(self, h): return self.net(h)

class ProjectionHead(nn.Module):
    """仅对比训练期使用，训练后丢弃；输出已 L2 归一化"""
    def __init__(self, dim=1280, hidden=1280, out=128, spec_norm=True):
        super().__init__()
        self.net = nn.Sequential(_lin(dim, hidden, spec_norm), nn.ReLU(inplace=True),
                                 _lin(hidden, out, spec_norm))
    def forward(self, h): return F.normalize(self.net(h), dim=-1)
