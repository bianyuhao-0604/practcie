"""L3-PPI core model.

Implements the three-stage framework from the paper:

1. **Surrogate pre-training** -- a GIN that learns to recognise valid L3
   pattern graphs (Sec. 4.1).  Trained on synthetic L3 patterns
   (graph-level binary classification).  ``L3PPISurrogate``.
2. **Prompt pattern construction** -- for a query protein pair (u, v) we
   build a star-shaped prompt of ``K`` candidate L3 paths sharing a central
   virtual node ``v0`` (Sec. 4.2).  Each candidate path is the 4-node,
   3-edge graph  u — p_i — p0 — v  (the "L3 path" between u and v via
   two virtual nodes).  All K paths plus the query nodes form the pattern
   graph of  K + 3 nodes and 3K + 1 edges.
3. **Gating + path-number regularisation** -- a gate scores every path;
   Gumbel-Softmax produces a soft/hard activation.  ℒ_PN (Eq. 4) pushes
   positives toward more active paths and negatives toward fewer.
   The activated prompt is read by the (frozen) surrogate -> final PPI
   likelihood.

The prompt + gate is **model-agnostic**: it consumes protein embeddings
from any encoder (here ESM-2 / CNN) and attaches as a plug-and-play head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import PromptCfg


# ---------------------------------------------------------------------- #
# 1. Surrogate: L3 pattern recognition (pre-trained, then frozen)
# ---------------------------------------------------------------------- #
class L3PPISurrogate(nn.Module):
    """GIN + graph-level classifier.  Scores an L3 *pattern graph*."""

    def __init__(self, node_dim, cfg: PromptCfg, gnn_name="gin"):
        super().__init__()
        from gnn import build_gnn, global_pool
        self._build_gnn, self._pool = build_gnn, global_pool
        self.gnn = build_gnn(gnn_name, node_dim, cfg)
        self.head = nn.Sequential(
            nn.Linear(self.gnn.out_dim, self.gnn.out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.gin_dropout),
            nn.Linear(self.gnn.out_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        h = self.gnn(x, edge_index, edge_weight, batch)
        g = self._pool(h, batch, mode="sum")
        return self.head(g).squeeze(-1)          # (B,) raw logits

    @torch.no_grad()
    def predict(self, x, edge_index, edge_weight, batch):
        return torch.sigmoid(self.forward(x, edge_index, edge_weight, batch))


# ---------------------------------------------------------------------- #
# 2. Gate for one candidate L3 path
# ---------------------------------------------------------------------- #
class PathGate(nn.Module):
    """Scores one candidate L3 path -> (B,) raw activation in (-inf, +inf)."""

    def __init__(self, node_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 4, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, 1),
        )

    def forward(self, z_u, z_v, p_0, p_i):
        cat = torch.cat([z_u, z_v, p_0, p_i], dim=-1)
        return self.net(cat).squeeze(-1)


# ---------------------------------------------------------------------- #
# 3. Full L3-PPI predictor
# ---------------------------------------------------------------------- #
class L3PPI(nn.Module):
    """Prompt head on top of a protein encoder.

    Parameters
    ----------
    encoder : nn.Module
        Maps (tokens, mask) -> (B, D).  See ``encoder.py``.
    surrogate : L3PPISurrogate
        Pre-trained surrogate (frozen during fine-tuning).
    cfg : PromptCfg
    """

    def __init__(self, encoder, surrogate, cfg: PromptCfg = None):
        super().__init__()
        self.cfg = cfg or PromptCfg()
        self.encoder = encoder
        self.surrogate = surrogate
        D = self.cfg.node_dim
        self.K = self.cfg.K
        # ESM encoder exposes forward_str(seqs, device); CNN uses forward(tokens, mask)
        self.is_esm = hasattr(encoder, "forward_str")

        # learnable prompt nodes: p0 (central) + p1..pK (branch nodes)
        self.prompt_nodes = nn.Parameter(torch.randn(self.K + 1, D) * 0.1)
        self.gate = PathGate(D)

    # ---------------- prompt graph assembly ----------------
    def _pattern(self, z_u, z_v, per_path):
        """Assemble the (batched) L3 pattern graph with gated edge weights.

        Node layout per example:  [u, v, p0, p1, ..., pK]   (K+3 nodes)
        Edges per branch i in 1..K:  (u, p_i), (p_i, p0), (p0, v)  -> 3K
        Plus an optional direct query edge (u, v) -> 1  (3K+1 total)

        ``per_path`` (B, K): activation of every branch (Gumbel / sigmoid).
        Each branch's 3 edges are weighted by its activation.

        Returns ``x (N,D), edge_index (2,E), edge_weight (E,), batch (N,)``.
        """
        B, D = z_u.shape
        K = self.K
        dev = z_u.device
        p = self.prompt_nodes                       # (K+1, D)
        N = K + 3

        nodes = torch.zeros(B, N, D, device=dev)
        nodes[:, 0] = z_u
        nodes[:, 1] = z_v
        nodes[:, 2] = p[0]
        nodes[:, 3:] = p[1:].unsqueeze(0).expand(B, -1, -1)

        # local edge list (per example): 3K edges
        local_row, local_col = [], []
        for i in range(1, K + 1):
            local_row += [0, i + 2, 2]      # u,  p_i,  p0
            local_col += [i + 2, 2, 1]       # p_i, p0,  v
        local_row = torch.tensor(local_row, device=dev)
        local_col = torch.tensor(local_col, device=dev)
        E_per_ex = local_row.numel()

        offset = (torch.arange(B, device=dev) * N).unsqueeze(1)
        edge_index = torch.stack([
            local_row.unsqueeze(0) + offset,
            local_col.unsqueeze(0) + offset,
        ], dim=0).reshape(2, B * E_per_ex)

        # edge weight: each branch i's 3 edges get per_path[:, i-1]
        # order in flattened edges: branch 1 (3), branch 2 (3), ...
        branch_of_edge = (torch.arange(E_per_ex, device=dev) // 3)   # (3K,) in 0..K-1
        act = per_path[:, branch_of_edge]          # (B, 3K)
        edge_weight = act.reshape(-1)              # (B*3K,)

        x = nodes.reshape(B * N, D)
        batch = torch.arange(B, device=dev).unsqueeze(1).expand(B, N).reshape(-1)
        return x, edge_index, edge_weight, batch

    # ---------------- forward ----------------
    def forward(self, batch, tau=None, hard=None, return_debug=False):
        dev = next(self.parameters()).device
        # ESM needs raw sequence strings; CNN uses integer tokens + mask.
        if self.is_esm:
            z_u = self.encoder.forward_str(batch["seq_a"], dev, ids=batch["id_a"])
            z_v = self.encoder.forward_str(batch["seq_b"], dev, ids=batch["id_b"])
        else:
            xa, ma = batch["a"].to(dev), batch["a_mask"].to(dev)
            xb, mb = batch["b"].to(dev), batch["b_mask"].to(dev)
            z_u = self.encoder(xa, ma)
            z_v = self.encoder(xb, mb)
        D = z_u.size(-1)

        # gate each branch
        p = self.prompt_nodes
        p0 = p[0].expand_as(z_u)
        pi = p[1:]                                          # (K, D)
        logits = torch.stack(
            [self.gate(z_u, z_v, p0, pi[i].expand_as(z_u)) for i in range(self.K)],
            dim=1)                                          # (B, K)

        # Gumbel-Softmax relaxation (Eq. 3).  hard at train/eval? see paper:
        # during tuning sample soft->hard; at inference use hard selections.
        t = tau if tau is not None else self.cfg.tau
        use_hard = self.training if hard is None else hard
        if self.training or use_hard:
            paths = F.gumbel_softmax(logits, tau=t, hard=use_hard, dim=1)
        else:
            paths = torch.sigmoid(logits)

        x, edge_index, w, bat = self._pattern(z_u, z_v, paths)

        # surrogate graph-level score
        score = self.surrogate(x, edge_index, w, bat)

        out = dict(logit=score, prob=torch.sigmoid(score),
                   paths=paths, gate_logits=logits)
        if return_debug:
            out["active"] = paths.sum(1)                    # (B,)
        return out

    @torch.no_grad()
    def predict(self, batch, tau=None):
        self.eval()
        return self.forward(batch, tau=tau, hard=False)["prob"]


# ---------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------- #
def pn_loss(active_per_sample, label, K, gamma):
    """ℒ_PN (paper Eq. 4).

    active_per_sample (B,) = sum_i path_i   (after Gumbel/soft)
    label               (B,) in {0,1}
        y=1 : max(0,  K*(1 - 1/γ) - Σ p_i )
        y=0 : max(0,  Σ p_i - K/γ )
    """
    term = K * (1.0 - 1.0 / gamma)
    l_pos = torch.relu(term - active_per_sample)
    l_neg = torch.relu(active_per_sample - K / gamma)
    pos = (label == 1)
    return (l_pos * pos + l_neg * (~pos)).mean()


def finetune_loss(out, label, cfg: PromptCfg):
    """BCE (PPI classification) + λ · ℒ_PN (path-number regularisation)."""
    try:
        from config import FineTuneCfg
        reg_w = FineTuneCfg.reg_weight
    except Exception:
        reg_w = 0.5
    active = out["paths"].sum(1)                            # (B,)
    bce = F.binary_cross_entropy_with_logits(out["logit"], label.float())
    pn = pn_loss(active, label, cfg.K, cfg.gamma)
    return dict(total=bce + reg_w * pn, bce=bce, pn=pn)


# ---------------------------------------------------------------------- #
# Synthetic L3 pattern graph generator for surrogate pre-training
# ---------------------------------------------------------------------- #
def sample_pattern_graphs(batch_size, K, node_dim, pos_ratio=0.5, device="cpu"):
    """Return a batch of random L3 pattern graphs (toy surrogate target).

    Each graph has the same topology as the prompt (K+3 nodes, 3K edges)
    but with random node features and a binary label: patterns carrying a
    high-multiplicity 'L3 signal' (here: a distinguished central motif
    repeated ``n_signal`` times) are positive.  This is a self-contained
    stand-in for patterns mined from the real PPI network -- replace with
    mined patterns by overriding ``SurrogateDataset`` for full fidelity.
    """
    n_pos = int(batch_size * pos_ratio)
    labels = torch.zeros(batch_size, device=device)
    graphs = []
    for b in range(batch_size):
        is_pos = b < n_pos
        n_signal = (K if is_pos else max(0, K // 4))
        x = torch.randn(K + 3, node_dim, device=device) * 0.3
        # positive signal: first n_signal branches share a correlated motif
        if is_pos and n_signal > 0:
            motif = torch.randn(node_dim, device=device)
            x[3:3 + n_signal] += motif.unsqueeze(0)
        row, col = [], []
        for i in range(1, K + 1):
            row += [0, i + 2, 2]; col += [i + 2, 2, 1]
        ei = torch.tensor([row, col], device=device).long()
        ew = torch.ones(ei.size(1), device=device)
        graphs.append((x, ei, ew))
        labels[b] = 1.0 if is_pos else 0.0
    # pack into a single big graph with batch vector
    xs, eis, ws, batch = [], [], [], []
    off = 0
    for b, (x, ei, ew) in enumerate(graphs):
        N = x.size(0)
        xs.append(x); ws.append(ew)
        eis.append(ei + off)
        batch.append(torch.full((N,), b, device=device, dtype=torch.long))
        off += N
    return (torch.cat(xs), torch.cat(eis, dim=1), torch.cat(ws),
            torch.cat(batch)), labels


class SyntheticPatternDataset(torch.utils.data.Dataset):
    """On-the-fly synthetic L3 pattern graphs (used when no mined patterns)."""

    def __init__(self, n=4096, K=8, node_dim=64, seed=0):
        self.n, self.K, self.node_dim = n, K, node_dim
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # one graph per item (variable size -> collate below)
        is_pos = (i % 2 == 0)
        K, D = self.K, self.node_dim
        n_signal = K if is_pos else max(0, K // 4)
        x = torch.randn(K + 3, D) * 0.3
        if n_signal > 0:
            x[3:3 + n_signal] += torch.randn(D)
        row, col = [], []
        for i in range(1, K + 1):
            row += [0, i + 2, 2]; col += [i + 2, 2, 1]
        return dict(x=x, edge_index=torch.tensor([row, col], dtype=torch.long),
                    edge_weight=torch.ones(3 * K), label=torch.tensor(1.0 if is_pos else 0.0))


def pattern_collate(items):
    xs, eis, ews, labels, batch = [], [], [], [], []
    off = 0
    for it in items:
        N = it["x"].size(0)
        xs.append(it["x"]); ews.append(it["edge_weight"])
        eis.append(it["edge_index"] + off)
        batch.append(torch.full((N,), len(labels), dtype=torch.long))
        labels.append(it["label"])
        off += N
    return (torch.cat(xs), torch.cat(eis, dim=1), torch.cat(ews),
            torch.cat(batch)), torch.stack(labels)
