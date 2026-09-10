# -*- coding: utf-8 -*-
"""
baselines_paper.py — PPI基线复现 (完整版 v3.4)
协议: 训练=Intra1(sub=0.5) | 验证=Intra0 | 测试=Intra2 | b=16批内逐样本前向/反向
      Adam lr=1e-3 | 梯度裁剪1.0 | 最多30轮 | 早停: 验证AUROC连续8轮不提升
v3.4: DScriptLike分桶前向(边长取整到64的倍数) — 修复显存分配器"形状风暴":
      每对全新张量形状→缓存永不命中→WDDM下每对50-150ms的cudaMalloc开销(2.7对/s元凶)。
      等价性已沙箱验证: eval逐位一致 | train梯度<5e-6 | 3步SGD轨迹差1.5e-10。
环境变量: PP_NOPAD=1回退v3.2原路径 | PP_BUCKET=64分桶粒度 | PP_EPOCHS=30轮数上限
          PP_MAXLEN=2000长度过滤(防超长对OOM) | PP_H5=嵌入h5绝对路径
          PP_HEARTBEAT=250心跳批次间隔 | PP_VALEVERY=4000验证进度间隔
用法: python baselines_paper.py <dscript|richoux> [种子...]   (默认种子42)
依赖: torch(+CUDA) h5py numpy scikit-learn
"""
import os, sys, time, glob, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

# ==================== 配置 ====================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "dataset"

TRAIN_POS = DATA_DIR / "Intra1_pos_rr.txt"; TRAIN_NEG = DATA_DIR / "Intra1_neg_rr.txt"
VAL_POS   = DATA_DIR / "Intra0_pos_rr.txt"; VAL_NEG   = DATA_DIR / "Intra0_neg_rr.txt"
TEST_POS  = DATA_DIR / "Intra2_pos_rr.txt"; TEST_NEG  = DATA_DIR / "Intra2_neg_rr.txt"
RESULTS_DIR = ROOT / "results"
os.environ.setdefault("PP_H5", str(ROOT / "Embeddings" / "embeddings_per_tok.h5"))


EMBED_DIM = 1280          # ESM-2嵌入维度; 与h5不符会在预载时报错并显示实际值
BATCH  = 16
LR     = 1e-3
EPOCHS = int(os.environ.get("PP_EPOCHS", "30"))
ES     = 8                # 早停耐心(轮)
SUB    = 0.5              # 训练集子采样比例
MAX_LEN = int(os.environ.get("PP_MAXLEN", "2000"))
HEARTBEAT_EVERY    = int(os.environ.get("PP_HEARTBEAT", "250"))
VAL_PROGRESS_EVERY = int(os.environ.get("PP_VALEVERY", "4000"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False    # v3.3: 必须关闭! 逐样本(L1,L2)形状不重复,
                                          # benchmark缓存永不命中, 每对触发全算法搜索

EMB = {}                  # 预载: 蛋白名 -> fp16 CPU张量
_LEN_CACHE = {}           # 未预载时的惰性长度缓存(bench2用)
_h5 = None

# ==================== 数据层 ====================
# 注: 若启动时报路径错误, 或启动后的"数据指纹"与旧版不一致, 把你旧文件的数据读入段
#     原样搬来替换本段即可, 模型/训练层不动。
class _DF:
    def __init__(s, name1, name2, interaction):
        s.name1, s.name2, s.interaction = name1, name2, interaction
    def __len__(s):
        return len(s.interaction)

def _resolve_h5():
    global _h5
    if _h5 is not None:
        return _h5
    p = os.environ.get("PP_H5")
    if not p:
        cands = []
        for d in (DATA_DIR, ROOT, HERE):
            cands += glob.glob(str(d / "*.h5"))
        cands = sorted(set(cands))
        if len(cands) != 1:
            sys.exit(f"[错误] 嵌入h5定位失败(找到{len(cands)}个候选): {cands}\n"
                     f"  解决: 设 $env:PP_H5='h5完整路径' 后重跑, 或把唯一的h5放进 {DATA_DIR}")
        p = cands[0]
    print(f"[嵌入] {p}", flush=True)
    _h5 = h5py.File(p, "r")
    return _h5

def protein_len(name):
    if name in EMB:
        return EMB[name].shape[0]
    if name not in _LEN_CACHE:
        _LEN_CACHE[name] = _resolve_h5()[name].shape[0]
    return _LEN_CACHE[name]

def load_split(pos_file, neg_file):
    name1, name2, inter = [], [], []
    for fn, lab in ((pos_file, 1), (neg_file, 0)):
        if not fn.exists():
            sys.exit(f"[错误] 数据文件不存在: {fn}\n  解决: 修改顶部 TRAIN/VAL/TEST 路径常量为实际位置")
        with open(fn, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name1.append(parts[0]); name2.append(parts[1]); inter.append(lab)
    print(f"[{Path(pos_file).stem}+{Path(neg_file).stem}] n={len(name1)} pos={sum(inter)}", flush=True)
    return _DF(name1, name2, inter)

def preload(names):
    t0 = time.time()
    f = _resolve_h5()
    need = sorted(set(names))
    missing = [n for n in need if n not in f]
    if missing:
        sys.exit(f"[错误] {len(missing)}/{len(need)}个蛋白不在h5中(如 {missing[:3]})\n"
                 f"  h5键示例: {list(f.keys())[:3]} — 检查txt与h5的蛋白ID格式是否一致")
    for n in need:
        arr = np.asarray(f[n][...], dtype=np.float16)
        if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
            sys.exit(f"[错误] 蛋白{n}嵌入形状={arr.shape}, 与EMBED_DIM={EMBED_DIM}不符 — 修改顶部EMBED_DIM")
        EMB[n] = torch.from_numpy(arr)
    gb = sum(t.numel() for t in EMB.values()) * 2 / 1024 ** 3
    print(f"[预载完成] {len(EMB)}个蛋白 {gb:.1f}GB(fp16) 用时{time.time()-t0:.0f}s — 训练期零h5读", flush=True)

def get_emb(name):
    return EMB[name]

# ==================== 模型层 ====================
# ---------- 模型1: D-SCRIPT-ESM-2 (v3.4分桶前向, 沙箱已验证与v3.2逐位等价) ----------
# 机理: (L1,L2)取整到64的倍数 → 大张量形状在有限组合内循环 → 显存缓存高命中,
#       消除每对一次的cudaMalloc/cudaFree风暴(WDDM下每次50-150ms)。
# 等价性: 零填充恒等(1x1卷积点态 / 7x7卷积零边=自带padding / BN统计只计实区),
#         eval逐位一致, train梯度差<5e-6, SGD轨迹差1.5e-10。
class DScriptLike(nn.Module):
    def __init__(s, embed_dim, d=100, w=7, h=50, x0=0.5, k=20, pool_size=9,
                 do_pool=False, do_w=True, theta_init=1, lambda_init=0, gamma_init=0):
        super().__init__()
        s.embed_dim = embed_dim; s.x0 = x0; s.do_w, s.do_pool = do_w, do_pool
        s.k = nn.Parameter(torch.FloatTensor([float(k)]))
        s.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)
        s.xx = nn.Parameter(torch.arange(50000), requires_grad=False)  # 覆盖超长蛋白
        s.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))
        if do_w:
            s.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            s.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))
        s.clip()
        s.fc1 = nn.Linear(embed_dim, d); s.relu1, s.dropout1 = nn.ReLU(), nn.Dropout(0.5)
        s.conv2 = nn.Conv2d(2 * d, h, 1); s.relu2 = nn.ReLU(); s.norm1 = nn.BatchNorm2d(h)
        s.conv = nn.Conv2d(h, 1, w, padding=w // 2); s.norm2 = nn.BatchNorm2d(1); s.relu3 = nn.ReLU()

    def _proj(s, x):
        x = x.to(torch.float32)
        if x.dim() == 3:
            x = x.view(-1, s.embed_dim)
        return s.dropout1(s.relu1(s.fc1(x)))

    def _masked_bn(s, bn, x, L1, L2, mask):
        """分桶图上的等价BN: 统计量只计实区(与nn.BatchNorm2d逐位一致), 假区强制归零"""
        N = L1 * L2
        mom = 0.1 if bn.momentum is None else bn.momentum
        if bn.training:
            mean = (x * mask).sum(dim=(2, 3), keepdim=True) / N
            var = ((x - mean) * mask).pow(2).sum(dim=(2, 3), keepdim=True) / N
            with torch.no_grad():
                unb = var * (N / (N - 1)) if N > 1 else var * 0.0
                bn.running_mean.mul_(1 - mom).add_(mom * mean.reshape(-1))
                bn.running_var.mul_(1 - mom).add_(mom * unb.reshape(-1))
        else:
            mean = bn.running_mean.view(1, -1, 1, 1)
            var = bn.running_var.view(1, -1, 1, 1)
        y = (x - mean) / torch.sqrt(var + bn.eps)
        return (y * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)) * mask

    def forward(s, x1, x2):
        if os.environ.get("PP_NOPAD", "0") == "1":
            return s.forward_orig(x1, x2)
        bkt = int(os.environ.get("PP_BUCKET", "64"))
        L1, L2 = x1.shape[-2], x2.shape[-2]
        P1 = ((L1 + bkt - 1) // bkt) * bkt
        P2 = ((L2 + bkt - 1) // bkt) * bkt
        z1 = F.pad(s._proj(x1), (0, 0, 0, P1 - L1))          # 实区=原值, 假区=0
        z2 = F.pad(s._proj(x2), (0, 0, 0, P2 - L2))
        a = z1.transpose(0, 1).unsqueeze(2).unsqueeze(0)      # (1, d, P1, 1)
        b = z2.transpose(0, 1).unsqueeze(1).unsqueeze(0)      # (1, d, 1, P2)
        m = torch.cat([(a - b).abs(), a * b], dim=1)          # (1, 2d, P1, P2) 分桶形状
        mask = z1.new_zeros((1, 1, P1, P2)); mask[:, :, :L1, :L2] = 1.0
        h1 = s.relu2(s._masked_bn(s.norm1, s.conv2(m), L1, L2, mask))
        Cp = s.relu3(s._masked_bn(s.norm2, s.conv(h1), L1, L2, mask))
        if s.do_w:
            u1 = -torch.square((s.xx[:L1] + 1 - (L1 + 1) / 2) / (-(L1 + 1) / 2))
            u2 = -torch.square((s.xx[:L2] + 1 - (L2 + 1) / 2) / (-(L2 + 1) / 2))
            Wr = (1 - s.theta) * torch.exp(s.lambda_ * u1).unsqueeze(1) * torch.exp(s.lambda_ * u2) + s.theta
            W = F.pad(Wr, (0, P2 - L2, 0, P1 - L1)).unsqueeze(0).unsqueeze(0)
            yhat = Cp * W
        else:
            yhat = Cp
        if s.do_pool:
            ps = s.maxPool.kernel_size
            yhat = s.maxPool(yhat)[:, :, : -(-L1 // ps), : -(-L2 // ps)]
            mu, sigma = torch.mean(yhat), torch.var(yhat)
            Q = torch.relu(yhat - mu - s.gamma * sigma)
        else:
            N = L1 * L2
            mu = yhat.sum() / N
            sigma = (((yhat - mu) * mask).pow(2).sum()) / (N - 1)
            Q = torch.relu(yhat - mu - s.gamma * sigma) * mask
        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        return torch.clamp(1 / (1 + torch.exp(-s.k * (phat - s.x0))), 0, 1), Cp

    def forward_orig(s, x1, x2):
        """v3.2原路径 (PP_NOPAD=1回退; bench2的A/B基准)"""
        x1 = x1.to(torch.float32).unsqueeze(0).contiguous().view(1, -1, s.embed_dim)
        x2 = x2.to(torch.float32).unsqueeze(0).contiguous().view(1, -1, s.embed_dim)
        x1 = s.dropout1(s.relu1(s.fc1(x1))); x2 = s.dropout1(s.relu1(s.fc1(x2)))
        a = x1.transpose(1, 2).unsqueeze(3)
        b = x2.transpose(1, 2).unsqueeze(2)
        m = torch.cat([(a - b).abs(), a * b], dim=1)
        m = s.relu2(s.norm1(s.conv2(m)))
        C = s.relu3(s.norm2(s.conv(m)))
        if s.do_w:
            N, M = C.shape[2:]
            u1 = -torch.square((s.xx[:N] + 1 - (N + 1) / 2) / (-(N + 1) / 2))
            u2 = -torch.square((s.xx[:M] + 1 - (M + 1) / 2) / (-(M + 1) / 2))
            W = (1 - s.theta) * torch.exp(s.lambda_ * u1).unsqueeze(1) * torch.exp(s.lambda_ * u2) + s.theta
            yhat = C * W
        else:
            yhat = C
        if s.do_pool:
            yhat = s.maxPool(yhat)
        mu, sigma = torch.mean(yhat), torch.var(yhat)
        Q = torch.relu(yhat - mu - s.gamma * sigma)
        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        return torch.clamp(1 / (1 + torch.exp(-s.k * (phat - s.x0))), 0, 1), C

    def clip(s):
        if s.do_w:
            s.theta.data.clamp_(min=0, max=1); s.lambda_.data.clamp_(min=0)
        s.gamma.data.clamp_(min=0)

# ---------- 模型2: Richoux (LSTM, 嵌入重实现) ----------
# 注意: 42号种子的richoux结果已在旧版完成并存档; 若此实现与你旧版结构有出入,
# 补跑其他种子前先回来对齐, 不要把不同结构的行写进同一张表。
class RichouxLike(nn.Module):
    def __init__(s, embed_dim, proj=50, hidden=50):
        super().__init__()
        s.proj = nn.Linear(embed_dim, proj)
        s.lstm = nn.LSTM(proj, hidden, num_layers=1, batch_first=True, bidirectional=True)
        s.head = nn.Sequential(nn.Linear(2 * hidden, 2 * hidden), nn.ReLU(),
                               nn.Dropout(0.5), nn.Linear(2 * hidden, 1))
    def forward(s, x1, x2):
        z = torch.cat([s.proj(x1.to(torch.float32)),
                       s.proj(x2.to(torch.float32))], dim=0).unsqueeze(0)
        out, _ = s.lstm(z)
        p = torch.sigmoid(s.head(out[:, -1])).view(1)
        return p, None

MODELS = {"dscript": DScriptLike, "richoux": RichouxLike}

# ==================== 训练层 ====================
def forward_loss_dscript(model, crit, b):
    """v3.4: 逐样本forward+backward; 标签整批一次上载; loss在GPU累积(差~1e-7)"""
    tot = torch.zeros((), device=DEVICE)
    ysd = torch.tensor(b["interaction"], dtype=torch.float32, device=DEVICE)
    for i in range(len(b["interaction"])):
        p, _ = model(get_emb(b["name1"][i]).to(DEVICE),
                     get_emb(b["name2"][i]).to(DEVICE))
        li = crit(p.view(1), ysd[i:i + 1])
        li.backward()
        tot = tot + li.detach()
    return tot

def _batches(pairs, n):
    for i in range(0, len(pairs), n):
        chunk = pairs[i:i + n]
        yield {"name1": [p[0] for p in chunk],
               "name2": [p[1] for p in chunk],
               "interaction": [p[2] for p in chunk]}

def train_one_epoch(model, opt, crit, pairs, seed, ep):
    model.train()
    order = list(pairs)
    random.Random(10000 * seed + ep).shuffle(order)
    nbat = (len(order) + BATCH - 1) // BATCH
    t0 = time.time(); tloss = 0.0; nb = 0
    for b in _batches(order, BATCH):
        tot = forward_loss_dscript(model, crit, b)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if hasattr(model, "clip"):
            model.clip()
        tloss += float(tot); nb += 1
        done = min(nb * BATCH, len(order))
        if nb % HEARTBEAT_EVERY == 0:
            rate = done / (time.time() - t0)
            eta = (len(order) - done) / rate / 60
            print(f"    [心跳] batch {nb}/{nbat} | {rate:.1f}对/s | 训练段剩余~{eta:.0f}min", flush=True)
    return tloss / max(nb, 1), (time.time() - t0) / 60

def evaluate(model, crit, pairs, tag="验证"):
    model.eval()
    ys = torch.tensor([p[2] for p in pairs], dtype=torch.float32, device=DEVICE)
    ps = []
    tot = torch.zeros((), device=DEVICE)
    t0 = time.time()
    with torch.no_grad():
        for i, (n1, n2, _) in enumerate(pairs):
            p, _ = model(get_emb(n1).to(DEVICE), get_emb(n2).to(DEVICE))
            tot = tot + crit(p.view(1), ys[i:i + 1])
            ps.append(p)
            if (i + 1) % VAL_PROGRESS_EVERY == 0:
                print(f"    [{tag}] {i + 1}/{len(pairs)} | {(i + 1) / (time.time() - t0):.1f}对/s", flush=True)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    except ImportError:
        sys.exit("[错误] 缺 scikit-learn: pip install scikit-learn")
    pn = torch.cat(ps).float().cpu().numpy()
    ynp = ys.cpu().numpy()
    return {"loss": float(tot) / len(pairs),
            "auroc": float(roc_auc_score(ynp, pn)),
            "aupr": float(average_precision_score(ynp, pn)),
            "f1": float(f1_score(ynp, (pn >= 0.5).astype(int))),
            "mins": (time.time() - t0) / 60}

def train_model(model_name, seed, tr, va, te):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = RESULTS_DIR / f"{model_name}_seed{seed}_best.pt"

    # 子采样 + 长度过滤
    idx = list(range(len(tr)))
    random.Random(seed).shuffle(idx)
    keep = idx[: int(len(tr) * SUB)]
    def to_pairs(df, ids=None):
        ids = ids if ids is not None else range(len(df))
        return [(df.name1[i], df.name2[i], df.interaction[i]) for i in ids]
    def filt(pairs):
        return [p for p in pairs
                if protein_len(p[0]) <= MAX_LEN and protein_len(p[1]) <= MAX_LEN]
    trp = to_pairs(tr, keep); vap = to_pairs(va); tep = to_pairs(te)
    n0 = (len(trp), len(vap), len(tep))
    trp, vap, tep = filt(trp), filt(vap), filt(tep)
    Lmax = max(max(protein_len(p[0]), protein_len(p[1])) for p in trp) if trp else 0
    print(f"[长度] 训练对最大L={Lmax} | MAX_LEN={MAX_LEN} 丢弃: 训练{n0[0]-len(trp)} "
          f"验证{n0[1]-len(vap)} 测试{n0[2]-len(tep)} | 训练{len(trp)}对 验证{len(vap)}对 测试{len(tep)}对", flush=True)
    if Lmax > 1500:
        print("[警告] 最大L>1500有显存OOM风险 — 可Ctrl+C后设 $env:PP_MAXLEN='1200' 重启(丢弃超长对,协议记一笔)", flush=True)

    model = MODELS[model_name](EMBED_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCELoss()

    print("\n" + "=" * 64)
    print(f"[{model_name} seed {seed}] lr={LR:g} ep={EPOCHS} es={ES} sub={SUB:g} "
          f"b={BATCH} device={DEVICE.type} | {time.strftime('%H:%M:%S')}")
    print("=" * 64, flush=True)

    best = {"auroc": -1.0, "epoch": -1, "state": None, "val": None}
    es_left, ep_run, t_start = ES, 0, time.time()
    try:
        for ep in range(1, EPOCHS + 1):
            ep_run = ep
            print(f"\n[epoch {ep}/{EPOCHS}] 训练中...", flush=True)
            tl, tmin = train_one_epoch(model, opt, crit, trp, seed, ep)
            vm = evaluate(model, crit, vap)
            if vm["auroc"] > best["auroc"]:
                best = {"auroc": vm["auroc"], "epoch": ep, "val": vm,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                es_left = ES; mark = "(新最优)"
            else:
                es_left -= 1; mark = ""
            print(f"[epoch {ep}/{EPOCHS}] loss={tl:.4f}({tmin:.0f}min) | 验证: loss={vm['loss']:.4f} "
                  f"AUROC={vm['auroc']:.4f} AUPR={vm['aupr']:.4f} F1={vm['f1']:.4f}({vm['mins']:.0f}min) | "
                  f"best=epoch{best['epoch']}(AUROC {best['auroc']:.4f}) es剩{es_left} {mark}", flush=True)
            if best["state"] is not None:
                torch.save(best["state"], ckpt)
            if es_left <= 0:
                print(f"[早停] 验证AUROC已连续{ES}轮不提升, 停止训练", flush=True)
                break
    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C — 最优模型已存盘(若出现过); 本次不写结果行; 重启将从头训练", flush=True)
        return None

    if best["state"] is None:
        print("[异常] 没有任何验证记录, 不写结果", flush=True)
        return None
    model.load_state_dict(best["state"])
    tm = evaluate(model, crit, tep, tag="测试")
    wallh = (time.time() - t_start) / 3600
    bv = best["val"]
    print(f"\n[完成] {model_name} seed={seed} | best epoch {best['epoch']} | "
          f"val AUROC={bv['auroc']:.4f} AUPR={bv['aupr']:.4f} F1={bv['f1']:.4f} | "
          f"test AUROC={tm['auroc']:.4f} AUPR={tm['aupr']:.4f} F1={tm['f1']:.4f} | "
          f"总用时{wallh:.1f}h | 模型: {ckpt}", flush=True)
    return {"model": model_name, "seed": seed, "best_epoch": best["epoch"], "epochs_run": ep_run,
            "val_auroc": round(bv["auroc"], 4), "val_aupr": round(bv["aupr"], 4), "val_f1": round(bv["f1"], 4),
            "test_auroc": round(tm["auroc"], 4), "test_aupr": round(tm["aupr"], 4), "test_f1": round(tm["f1"], 4),
            "wall_h": round(wallh, 1), "version": "v3.4"}

def append_result(row):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    header = "model,seed,best_epoch,epochs_run,val_auroc,val_aupr,val_f1,test_auroc,test_aupr,test_f1,wall_h,version"
    f = RESULTS_DIR / "paper_results.csv"
    if f.exists():
        first = f.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        if first and first[0].strip() != header:
            f = RESULTS_DIR / "paper_results_v34.csv"
            print(f"[注意] 已有结果文件表头不同, 本次改写入 {f.name}", flush=True)
    new = not f.exists()
    with open(f, "a", encoding="utf-8") as fh:
        if new:
            fh.write(header + "\n")
        fh.write(",".join(str(row[k]) for k in
                ["model", "seed", "best_epoch", "epochs_run", "val_auroc", "val_aupr", "val_f1",
                 "test_auroc", "test_aupr", "test_f1", "wall_h", "version"]) + "\n")
    print(f"[结果] 已追加 {f}", flush=True)

def main():
    if len(sys.argv) < 2 or sys.argv[1].lower() not in MODELS:
        print(f"用法: python baselines_paper.py <{'|'.join(MODELS)}> [种子...]  (默认种子42)")
        return
    model_name = sys.argv[1].lower()
    seeds = [int(a) for a in sys.argv[2:]] or [42]
    print("加载数据...", flush=True)
    tr = load_split(TRAIN_POS, TRAIN_NEG)
    va = load_split(VAL_POS, VAL_NEG)
    te = load_split(TEST_POS, TEST_NEG)
    preload(set(tr.name1) | set(tr.name2) | set(va.name1) | set(va.name2)
            | set(te.name1) | set(te.name2))
    for s in seeds:
        row = train_model(model_name, s, tr, va, te)
        if row:
            append_result(row)

if __name__ == "__main__":
    main()
