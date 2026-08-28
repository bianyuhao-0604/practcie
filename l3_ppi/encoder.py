"""Protein sequence encoder.

Priority:
1. If ``EncoderCfg.use_esm`` and ``fair-esm`` is installed -> ESM-2 embeddings.
2. Otherwise -> a small multi-scale 1D-CNN over amino-acid one-hot (the
   "learned fallback" used when the project runs offline / CPU-only).

Both expose ``forward(tokens, mask) -> (B, D)`` so the rest of the model is
agnostic to the backbone.
"""
import os
import torch
import torch.nn as nn
from config import EncoderCfg, CACHE_ROOT


class CNNEncoder(nn.Module):
    """Multi-scale 1D CNN over one-hot amino-acid tokens."""

    def __init__(self, cfg: EncoderCfg):
        super().__init__()
        self.vocab = cfg.aa_vocab
        self.embed = nn.Embedding(cfg.aa_vocab + 1, 32, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(32, cfg.cnn_channels, k, padding=k // 2),
                nn.BatchNorm1d(cfg.cnn_channels),
                nn.ReLU(inplace=True),
            ) for k in cfg.cnn_kernels
        ])
        self.proj = nn.Linear(cfg.cnn_channels * len(cfg.cnn_kernels), cfg.node_dim)
        self.out_dim = cfg.node_dim

    def forward(self, tokens, mask):
        # tokens: (B, L)  mask: (B, L)
        x = self.embed(tokens).transpose(1, 2)          # (B, 32, L)
        feats = [conv(x) for conv in self.convs]        # each (B, C, L)
        x = torch.cat(feats, dim=1).transpose(1, 2)     # (B, L, K*C)
        x = self.proj(x)                                # (B, L, D)
        mask = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled


class ESMEncoder(nn.Module):
    """Wrapper around facebookresearch/esm (ESM-2). Lazy import."""

    def __init__(self, cfg: EncoderCfg):
        super().__init__()
        import esm
        model_name = cfg.esm_model
        self._esm, self._alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self._batch_converter = self._alphabet.get_batch_converter()
        hidden = self._esm.embed_dim
        if cfg.freeze:
            for p in self._esm.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(hidden, cfg.node_dim) if hidden != cfg.node_dim else nn.Identity()
        self.out_dim = cfg.node_dim
        self.pool = cfg.esm_pool
        self._device = None
        self._cache = {}   # id -> embedding tensor (after proj), for frozen ESM

    def load_cache(self, path):
        """Load pre-computed embeddings from disk into the in-memory cache."""
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu")
            self._cache = {k: v for k, v in data.items()}
            print(f"[encoder] loaded {len(self._cache)} pre-computed ESM embeddings from {path}", flush=True)
            return True
        return False

    def to(self, device):
        super().to(device)
        self._device = device
        self._esm.to(device)
        return self

    def forward(self, tokens, mask, seqs=None):
        # tokens/mask from AA encoder are ignored; we re-convert from raw seqs
        raise NotImplementedError("use ESMEncoder.forward_str(batch_seqs, device, ids)")

    def forward_str(self, seqs, device, ids=None):
        """Embed a batch of sequences.  When ESM is frozen and ``ids`` are given,
        results are cached per id so repeated proteins are encoded only once."""
        # serve as many as possible from cache; collect indices needing compute
        out = [None] * len(seqs)
        miss_idx, miss_seq = [], []
        n_hit = 0
        if ids is not None:
            for i, (pid, s) in enumerate(zip(ids, seqs)):
                cached = self._cache.get(pid)
                if cached is not None:
                    out[i] = cached.to(device); n_hit += 1
                else:
                    miss_idx.append(i); miss_seq.append((pid, s))
        else:
            miss_idx = list(range(len(seqs)))
            miss_seq = [(None, s) for s in seqs]

        if miss_seq:
            data = [("p", s[:512]) for _, s in miss_seq]  # cap at 512aa for speed
            _, _, batch_tokens = self._batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.set_grad_enabled(not self._esm.training or any(
                    p.requires_grad for p in self._esm.parameters())):
                results = self._esm(batch_tokens, repr_layers=[self._esm.num_layers], return_contacts=False)
            h = results["representations"][self._esm.num_layers]   # (B, L+2, H)
            h = h[:, 1:-1, :]                                      # drop <cls>/<eos>
            if self.pool == "cls":
                h = h[:, 0, :]
            else:
                mask = (batch_tokens[:, 1:-1] != self._alphabet.padding_idx).float()
                # keepdim=True so (B,L,1)*(B,L,1)->(B,L,H) and sum/(B,1) broadcasts right
                h = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            h = self.proj(h)
            for k, (pid, _) in enumerate(miss_seq):
                out[miss_idx[k]] = h[k]
                if pid is not None:
                    self._cache[pid] = h[k].detach()
        return torch.stack(out, dim=0)


def build_encoder(cfg: EncoderCfg, device=None):
    """Return (encoder, is_esm).  Falls back to CNN automatically."""
    if cfg.use_esm:
        try:
            import esm  # noqa
            print("[encoder] using ESM-2:", cfg.esm_model, "(freeze=%s)" % cfg.freeze)
            return ESMEncoder(cfg).to(device), True
        except Exception as e:
            print(f"[encoder] ESM-2 unavailable ({e}); falling back to CNN baseline.")
    enc = CNNEncoder(cfg)
    if device is not None:
        enc = enc.to(device)
    print(f"[encoder] CNN fallback  out_dim={enc.out_dim}")
    return enc, False
