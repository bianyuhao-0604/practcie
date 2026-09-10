# -*- coding: utf-8 -*-
# bench_gpu.py — GPU微基准: 测DScriptLike纯GPU上每对真实耗时, 隔离数据/循环/其他应用干扰
# 前提: 已Ctrl+C停训练, 已关闭浏览器/千问/WPS等GPU应用
import sys, time, importlib.util
from pathlib import Path
import torch, torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False          # 与v3.3训练设置一致

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bp", HERE / "baselines_paper.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)                     # 只加载类定义, main()有护栏不会执行

dev = bp.DEVICE
assert dev.type == "cuda", "未检测到CUDA"
free_b, total_b = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)} | 总显存 {total_b/1024**3:.1f}GB | 当前空闲 {free_b/1024**3:.1f}GB")
if free_b/1024**3 < 2.0:
    print("!! 空闲显存<2GB — 有应用占着或上次进程残留, 关掉它们(或重启机器)后重测, 否则数字失真")

model = bp.DScriptLike(bp.EMBED_DIM).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCELoss()
y1 = torch.tensor([1.0], device=dev)
results = {}
for L in (200, 400, 700, 1000):
    try:
        x1 = torch.randn(L, bp.EMBED_DIM, device=dev)
        x2 = torch.randn(L, bp.EMBED_DIM, device=dev)
        for _ in range(3):                       # 预热
            opt.zero_grad(); p, _ = model(x1, x2)
            crit(p.view(1), y1).backward(); opt.step()
        torch.cuda.synchronize(); t0 = time.time(); N = 20
        for _ in range(N):
            opt.zero_grad(); p, _ = model(x1, x2)
            crit(p.view(1), y1).backward(); opt.step()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / N * 1000
        results[L] = ms
        print(f"L={L:4d}x{L:<4d}: {ms:7.1f} ms/对 ({1000/ms:5.1f} 对/s) | 峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.2f}GB", flush=True)
        del x1, x2; torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"L={L}: CUDA OOM — 该长度显存不够(训练中MAX_LEN过滤后仍会碰到, 需处理)", flush=True)
        torch.cuda.empty_cache()

if 400 in results:
    ms = results[400]
    ep_min = 46860 * ms / 1000 / 60 * 1.45      # 训练46860对 + 验证段折算
    print(f"\n结论锚点: L=400(数据集均值L≈439, 由9.8GB/9296蛋白反推) = {ms:.0f}ms/对")
    print(f"         → ≈{ep_min:.0f}min/epoch → 15个epoch≈{ep_min*15/60:.1f}h")
    if ms < 25:   print("→ GPU本身够快! 之前慢在环境(其他应用抢GPU) — 保持应用关闭, 直接重启训练, 今晚可收官")
    elif ms < 60: print("→ GPU中等: 全程1-2天, 挂机可行或考虑云GPU")
    else:         print("→ GPU算力不足: 本地不可行 — 走云GPU(全程约¥10-20)或协议折衷, 带bench输出回来定方案")
