import torch, numpy as np
from torch.utils.data import DataLoader
from config import *
from simclr_data import SimCLRData
from simclr_augment import EmbedAugment
from simclr_model import Adapter, ProjectionHead
from simclr_loss import hybrid_loss
from train_simclr import IdxDS, seed_all

seed_all(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = SimCLRData(PREP_NPZ, SPLIT_DIR); data.X = data.X.to(dev)
dl = DataLoader(IdxDS(data.N), batch_size=BATCH, shuffle=True, drop_last=True)
aug = EmbedAugment(DIM_MASK_P, NOISE_STD, SCALE_RANGE)
adapter = Adapter(dim=1280, hidden=1280, spec_norm=SPEC_NORM).to(dev)
proj = ProjectionHead(dim=1280, out=PROJ_DIM, spec_norm=SPEC_NORM).to(dev)
params = list(adapter.parameters()) + list(proj.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

for bi, idx in enumerate(dl):
    idx = idx.to(dev); h = data.X[idx]
    z1, z2 = proj(adapter(aug(h))), proj(adapter(aug(h)))
    M = data.pos_mask(idx, dev)
    orphan = int((M.sum(1) == 0).sum()) if M.dim() == 2 else -1
    print(f"batch {bi}: M{tuple(M.shape)} 正样本对={int(M.sum())} "
          f"孤儿锚点={orphan}")
    loss, li, ls = hybrid_loss(z1, z2, M, TEMP, ALPHA)
    print(f"  loss={loss.item()} inst={li} sup={ls} "
          f"requires_grad={loss.requires_grad}")
    opt.zero_grad(); loss.backward()
    gnan = sum(1 for p in params if p.grad is not None and torch.isnan(p.grad).any())
    print(f"  梯度含nan的参数: {gnan}/{len(params)}")
    if bi >= 2: break
