import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"; FEATURES = ROOT / "features"
CKPT = ROOT / "checkpoints"; RESULTS = ROOT / "results"; LOGS = ROOT / "logs"
for d in (DATA, FEATURES, CKPT, RESULTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

# ---- 数据 ----
FIGSHARE_URL = "https://figshare.com/ndownloader/articles/21591618"
MAX_LEN = 256
TRAIN_N, VAL_N, TEST_N = 5000, 1000, 1000
SUBSET_SEED = 42
ROW_TRAIN, ROW_VAL, ROW_TEST = 163192, 59260, 52048   # 行数自动映射依据<span data-allow-html class='source-item source-aggregated' data-group-key='source-group-3' data-url='https://figshare&#46;com/articles/dataset/PPI&#95;prediction&#95;from&#95;sequence&#95;gold&#95;standard&#95;dataset/21591618' data-id='turn1fetch0'><span data-allow-html class='source-item-num' data-group-key='source-group-3' data-id='turn1fetch0' data-url='https://figshare&#46;com/articles/dataset/PPI&#95;prediction&#95;from&#95;sequence&#95;gold&#95;standard&#95;dataset/21591618'><span class='source-item-num-name' data-allow-html>figshare.com</span><span data-allow-html class='source-item-num-count'></span></span></span>

# ---- 特征维度 ----
SEQ_DIM, STRUCT_DIM, TEXT_DIM, GENOME_DIM = 480, 1024, 384, 64
ESM2_DIM    = 480
PROSTT5_DIM = 1024

# ---- 模型（预实验轻量版）----
D_FUSE, N_HEADS, DROPOUT = 128, 4, 0.2

# ---- 训练 ----
BATCH, EPOCHS, LR, WD = 16, 20, 5e-4, 1e-5
PATIENCE = 5
LS_EPS, MIXUP_PROB, RDROP_ALPHA = 0.05, 0.5, 1.0
SEEDS = [42, 1234]
