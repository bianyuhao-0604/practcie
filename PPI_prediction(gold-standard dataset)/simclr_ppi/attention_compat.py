# attention_compat.py
# Reim codebase 兼容层:不复制原权重结构,只做必要修补(均注明原因)。
# 修补 1: Attention.scale 在 __init__ 绑定 device,不被 .to() 搬运 → fix_scale
# 修补 2: kernel=2 + padding='same' 新版 PyTorch 禁用 → 广播退化下改 k=1(等价)
# 修补 3: nn.MultiheadAttention 默认 seq-first → batch_first=True
# 修补 4: create_square_mask 全零嵌入行防护(并入修补 6 的向量化实现)
# 修补 5: TUnA 的 VanillaRFFLayer 后验头与统一 Adam+BCE 反传协议不兼容:
#          其拟合机制是 update_precision 在线协方差/精度累积(含 train() 覆写),
#          需 Reim 原训练循环触发;反传下后验均值恒 0 → 输出恒 sigmoid(0)=0.5
#          (实测:三种子 aupr=0.4958/0.5047/0.4992, ep=1 早停)。
#          → 替换为可反传 nn.Linear(64,1) 头,接口/输出形状/sigmoid 不变。
#          [论文记录点:协议偏差,方法节如实注明]
# 修补 6: create_square_mask 原为逐样本 Python 循环 + .item()(GPU 同步),
#          每 epoch ~66 万次同步,单种子 341–1039s → 向量化重写。
#          [沙箱验证:L=1 实际场景与原版逐元素相等]
import torch, torch.nn as nn, torch.nn.functional as F
from models.attention import EncoderLayer, CrossEncoderLayer, TUnA as _TUnA

def fix_scale(module: nn.Module, dev):
    """必须在 model.to(dev) 之后调用。重绑 Attention.scale 到正确设备。"""
    for m in module.modules():
        if hasattr(m, "scale") and isinstance(m.scale, torch.Tensor) \
           and hasattr(m, "hid_dim") and hasattr(m, "n_heads"):
            m.scale = torch.sqrt(
                torch.FloatTensor([m.hid_dim // m.n_heads])).to(dev)

# ---- 修补 4 + 6:向量化 create_square_mask ----
# 与原版唯一差异:原版仅在"整批 mask 全零"时返回全 1;此版按样本防护
# (全零嵌入行 → 该样本 mask 全 1)。真实数据(嵌入非零)下与原版逐元素相等。
def _fast_square_mask(self, x):
    device = x.device
    N, seq_len, _ = x.size()                                   # x: (N, L, D)
    lens = (x.sum(dim=-1) != 0).long().sum(dim=1)              # (N,) 每样本有效长度
    keep = torch.arange(seq_len, device=device).unsqueeze(0) < lens.unsqueeze(1)  # (N, L)
    mask = (keep.unsqueeze(2) & keep.unsqueeze(1)).to(x.dtype)    # (N, L, L)
    empty = lens == 0                                          # 修补 4:全零嵌入行
    if empty.any():
        mask = torch.where(empty.view(-1, 1, 1), torch.ones_like(mask), mask)
    return mask.unsqueeze(1)                                   # (N, 1, L, L)

_TUnA.create_square_mask = _fast_square_mask

class _Chain(nn.Module):
    """原 SelfAtt/CrossAtt 共享 fc1→fc2→fc3 降维链 (D→D/4→D/16→h3)"""
    def __init__(self, D=1280, h3=64):
        super().__init__()
        h, h2 = D // 4, D // 16
        self.fc1 = nn.Linear(D, h); self.fc2 = nn.Linear(h, h2)
        self.fc3 = nn.Linear(h2, h3); self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x))))))

class AttB(nn.Module):
    """广播 L=1 的 batched Self/CrossAttention 交互头。输入 (B,1,D)。"""
    def __init__(self, D=1280, num_heads=8, h3=64, cross=False,
                 dropout=0.2, ff_dim=256):
        super().__init__()
        self.chain = _Chain(D, h3)
        self.core = (CrossEncoderLayer(h3, num_heads, ff_dim, dropout) if cross
                     else EncoderLayer(h3, num_heads, ff_dim, dropout))
        self.cross = cross
        self.conv = nn.Conv2d(h3, 1, kernel_size=1)      # 修补 2
        self.sigmoid = nn.Sigmoid()
    def forward(self, v1, v2):                           # (B,1,D)
        x1, x2 = self.chain(v1), self.chain(v2)
        if self.cross:
            x1 = self.core(x1, x2, None); x2 = self.core(x2, x1, None)
        else:
            x1 = self.core(x1, None);     x2 = self.core(x2, None)
        mat = torch.einsum('bik,bjk->bijk', x1, x2).permute(0, 3, 1, 2)
        return self.sigmoid(self.conv(mat).amax(dim=(1, 2, 3)))   # (B,)

class RichouxPP(nn.Module):
    """AttentionRichoux 修正版(修补 3)。self-attn → mean → 双塔 fc → 拼接。"""
    def __init__(self, D=1280, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, num_heads, batch_first=True)
        self.fc1 = nn.Linear(D, 20); self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(D, 20); self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(40, 20); self.fc6 = nn.Linear(20, 1)
    def forward(self, v1, v2):                           # (B,1,D)
        s1, _ = self.attn(v1, v1, v1); s2, _ = self.attn(v2, v2, v2)
        x1 = F.relu(self.fc2(F.relu(self.fc1(s1.mean(1)))))
        x2 = F.relu(self.fc4(F.relu(self.fc3(s2.mean(1)))))
        return torch.sigmoid(
            self.fc6(F.relu(self.fc5(torch.cat([x1, x2], 1))))).view(-1)

def build_tuna(D=1280, num_heads=8, hid=64, rffs=1028):
    # 修补 5:rffs 参数保留仅为签名兼容,不再使用。
    # 替换 pred_layer 后,forward 里 self.pred_layer(features) 的调用点、
    # 输出形状 (B,1)、外层 sigmoid 均与原版一致,无需改 TUnA.forward。
    core = _TUnA(embed_dim=D, num_heads=num_heads, hid_dim=hid, rffs=rffs)
    core.pred_layer = nn.Linear(hid, 1)      # 原: VanillaRFFLayer(hid, rffs, 1)
    return core
