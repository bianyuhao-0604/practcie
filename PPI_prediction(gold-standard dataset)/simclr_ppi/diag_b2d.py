import torch, numpy as np, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
D = 1280; torch.manual_seed(0)

class B2D(nn.Module):
    def __init__(self, D=1280, h3=64):
        super().__init__()
        h, h2 = D//4, D//16
        self.fc1=nn.Linear(D,h); self.fc2=nn.Linear(h,h2); self.fc3=nn.Linear(h2,h3)
        self.conv=nn.Conv2d(1,h3,1)
    def forward(self, v1, v2, tag=""):
        a = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(v1)))))).squeeze(1)
        b = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(v2)))))).squeeze(1)
        mat = torch.einsum('bi,bj->bij', a, b).unsqueeze(1)
        logit = self.conv(mat).amax(dim=(1,2,3)); out = torch.sigmoid(logit)
        print(f"[{tag}] a全零={(a.abs().sum(1)==0).float().mean():.1%} "
              f"a活跃率={(a>0).float().mean():.1%} | mat max={mat.max():.1f} | "
              f"logit mean={logit.mean():.2f} std={logit.std():.3f} | 输出std={out.std():.4f}")

c = list(Path('..').rglob('embeddings_prep.npz')); assert c, "没找到npz,手动改路径"
X = torch.tensor(np.load(c[0])['X'], dtype=torch.float32)
print(f"X: mean={X.mean():.2f} |X|max={X.abs().max():.1f} 逐维std[{X.std(0).min():.1f},{X.std(0).max():.1f}]")
i = torch.randperm(len(X)); v1, v2 = X[i[:256]].unsqueeze(1), X[i[256:512]].unsqueeze(1)
m = B2D(); ln = nn.LayerNorm(D)
m(v1, v2, "raw输入"); m(ln(v1), ln(v2), "LN输入 ")
