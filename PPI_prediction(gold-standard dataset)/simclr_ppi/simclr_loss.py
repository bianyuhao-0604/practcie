"""SimCLR/SupCon 混合损失(数值安全版)
- inst: NT-Xent, 正样本 = 同一蛋白的另一视图
- sup : SupCon, 正样本 = M 标记的 PPI 伙伴, 批内无伙伴的孤儿锚点剔除
掩码统一用有限大负数 -1e9, 杜绝 -inf*0=NaN。"""
import torch
import torch.nn.functional as F

NEG_INF = -1e9  # 有限: 任何乘法/求和都不会产生 NaN

def hybrid_loss(z1, z2, M, temp=0.2, alpha=1.0):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                  # (2B, d)
    logits = z @ z.t() / temp                       # (2B, 2B)
    eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    logits_nm = logits.masked_fill(eye, NEG_INF)    # 去自身

    # ---- inst: NT-Xent ----
    lab = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    inst = F.cross_entropy(logits_nm, lab)

    # ---- sup: SupCon, 孤儿锚点剔除 ----
    pos = M.float()
    n_pos = pos.sum(dim=1)
    valid = n_pos > 0
    lse = torch.logsumexp(logits_nm, dim=1, keepdim=True)
    log_prob = logits_nm - lse
    row = -(log_prob * pos).sum(dim=1) / n_pos.clamp(min=1.0)
    sup = row[valid].mean() if valid.any() else \
          torch.zeros((), device=z.device)

    loss = inst + alpha * sup
    return loss, inst.item(), sup.item()
