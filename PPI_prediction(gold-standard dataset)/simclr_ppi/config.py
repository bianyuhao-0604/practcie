from pathlib import Path

BASE = Path(r"D:\pythonprojects\practice-github\PPI_prediction(gold-standard dataset)")
PKL       = BASE / "Embeddings" / "embeddings_mean.pkl"
SPLIT_DIR = BASE / "dataset"
OUT_DIR   = BASE / "simclr_out"; OUT_DIR.mkdir(exist_ok=True)

PREP_NPZ     = OUT_DIR / "embeddings_prep.npz"    # 缩放(+白化)后的全蛋白嵌入
SIMCLR_NPZ   = OUT_DIR / "embeddings_simclr.npz"  # Adapter 重塑后的全蛋白嵌入
ADAPTER_CKPT = OUT_DIR / "adapter_best.pt"
SIMCLR_LOG   = OUT_DIR / "simclr_log.csv"

WHITEN     = False    # 消融开关: 缩放后是否做 ZCA 白化
WHITEN_EPS = 1e-4

# ---- SimCLR 超参 ----
SEED, BATCH, EPOCHS = 42, 512, 100
LR, WEIGHT_DECAY    = 1e-4, 1e-6
TEMP, ALPHA         = 0.2, 1.0        # ALPHA=0 → 纯 SimCLR
DIM_MASK_P          = 0.10
NOISE_STD           = 0.02
SCALE_RANGE         = (0.9, 1.1)
PROJ_DIM, SPEC_NORM = 128, True
PROBE_EVERY         = 5
PROBE_NPAIRS        = 20000

# ---- 下游 fc2 ----
FC2_EPOCHS, FC2_BATCH, FC2_LR = 60, 512, 1e-3
FC2_PATIENCE, N_SEEDS         = 8, 3
N_PERM                        = 1000
