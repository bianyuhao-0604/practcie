"""R0 contract smoke (CPU, seconds): store->collate->Reim2D->smooth->loss->backward->_predict."""
import os, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from config import Paths, build_model_cfg
from data import FeatureStore, PairDataset, load_pairs, make_collate
from engine import _predict, smooth, set_seed
from model import PPIModel

set_seed(42)
paths = Paths()
store = FeatureStore(paths.feat_dir)
print("store ok: n =", store.n)
mcfg = build_model_cfg("R0", None)
model = PPIModel(mcfg)
print("model ok:", model.report_params())

csv = os.path.join(paths.data_dir, "val.csv")
if not os.path.exists(csv):
    csv = os.path.join(paths.data_dir, "train.csv")
df = load_pairs(csv)
Ls = np.asarray(store.lengths)
ia = np.array([store.pid2idx[p] for p in df.pid_a])
ib = np.array([store.pid2idx[p] for p in df.pid_b])
sub = df[(Ls[ia] < 300) & (Ls[ib] < 300)].head(8).reset_index(drop=True)
print(f"smoke pairs: {len(sub)} (short proteins, CPU friendly)")

ds = PairDataset(sub, store)
coll = make_collate(store)
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False,
                                     collate_fn=coll, num_workers=0)
batch = next(iter(loader))
y = batch["y"]
y_sm = smooth(y, 0.05)
print("y:", tuple(y.shape), " y_sm:", tuple(y_sm.shape))

out1 = model(batch, y=y_sm, mix_pair=None)
t1 = out1["y_mix"] if out1["y_mix"] is not None else y_sm
print("logits:", tuple(out1["logits"].shape), " t1:", tuple(t1.shape))
loss = F.binary_cross_entropy_with_logits(out1["logits"].float().squeeze(-1), t1)
print("train loss:", float(loss))

perm = torch.randperm(y.shape[0])
out2 = model(batch, y=y_sm, mix_pair=(perm, 0.7))
t2 = out2["y_mix"] if out2["y_mix"] is not None else y_sm
loss2 = F.binary_cross_entropy_with_logits(out2["logits"].float().squeeze(-1), t2)
print("mixup loss:", float(loss2), " y_mix:", tuple(t2.shape))

loss.backward()
print("backward ok, sum|grad| =", float(sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)))

vp, vy = _predict(model, loader, "cpu")
vp, vy = np.asarray(vp).ravel(), np.asarray(vy).ravel()
print("_predict: vp", vp.shape, " vy", vy.shape)
from sklearn.metrics import average_precision_score
print("smoke AUPRC:", float(average_precision_score(vy.astype(int), vp)))
print("SMOKE PASS - R0 contract green, cleared for launch")
