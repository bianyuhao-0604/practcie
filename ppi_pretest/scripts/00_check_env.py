import sys, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from configs import *

print("=" * 60)
print("[0] Python:", sys.version.split()[0])
print("[1] torch:", torch.__version__)
ok = torch.cuda.is_available()
print("[2] CUDA available:", ok)
if not ok:
    print("!! 检测不到 GPU。请确认 torch_final 内安装的是 CUDA 版 torch：")
    print("   python -m pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"[3] GPU: {p.name}, VRAM {p.total_memory/1e9:.1f} GB")
bf16 = torch.cuda.is_bf16_supported()
print("[4] bf16 supported:", bf16)
# 显存试写
x = torch.zeros(1024, 1024, device="cuda"); del x; torch.cuda.empty_cache()
print("[5] 显存读写正常")
print("环境自检通过。")
