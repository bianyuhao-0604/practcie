"""Graph neural network building blocks used by the surrogate and the prompt graph."""
import torch
import torch.nn as nn
from config import PromptCfg


def gin_mlp(in_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
    )


class GIN(nn.Module):
    """Graph Isomorphism Network (Xu et al., 2019) over (X, edge_index, edge_weight)."""

    def __init__(self, in_dim, hidden, num_layers, dropout=0.1, pool="sum"):
        super().__init__()
        self.pool = pool
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GINConv(in_dim, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # x: (N, in) ; edge_index: (2, E) ; edge_weight: (E,)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = self.dropout(x)
        return x


class GINConv(nn.Module):
    """Message passing one layer with MLP( (1+eps)*x + sum msg )."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.mlp = gin_mlp(in_dim, hidden)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        agg = torch.zeros_like(x)
        if edge_weight is None:
            agg = agg.index_add(0, col, x[row])
        else:
            w = edge_weight.unsqueeze(-1)
            agg = agg.index_add(0, col, x[row] * w)
        out = (1 + self.eps) * x + agg
        return self.mlp(out)


class GCN(nn.Module):
    """Lightweight GCN for ablation."""

    def __init__(self, in_dim, hidden, num_layers, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Linear(in_dim, hidden) if i == 0 else nn.Linear(hidden, hidden)
            for i in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        deg = torch.zeros(x.size(0), device=x.device)
        deg = deg.index_add(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
        deg = deg.clamp(min=1).pow(-0.5)
        for i, conv in enumerate(self.convs):
            row, col = edge_index
            msg = x[row] * (deg[row] * deg[col]).unsqueeze(-1)
            x = msg.new_zeros((x.size(0), msg.size(-1))).index_add(0, col, msg)
            x = self.bns[i](conv(x))
            x = torch.relu(x)
            x = self.dropout(x)
        return x


def global_pool(x, batch, mode="sum"):
    """``x`` (N, D), ``batch`` (N,) -> (B, D).  Falls back to first-N split if batch=None."""
    if batch is None:
        return x  # caller did per-graph packing already
    out = []
    for b in torch.unique(batch):
        xi = x[batch == b]
        out.append(xi.sum(0) if mode == "sum" else xi.mean(0))
    return torch.stack(out)


def build_gnn(name, in_dim, cfg: PromptCfg):
    if name == "gin":
        return GIN(in_dim, cfg.gin_hidden, cfg.gin_num_layers, cfg.gin_dropout)
    if name == "gcn":
        return GCN(in_dim, cfg.gin_hidden, cfg.gin_num_layers, cfg.gin_dropout)
    raise ValueError(name)
