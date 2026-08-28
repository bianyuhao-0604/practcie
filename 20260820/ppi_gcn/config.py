"""
config.py — 全局配置
所有路径、超参数、设备选项集中管理，方便实验调参。
"""

import os
import torch

# ───────────────────── 路径配置 ─────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
PROC_DIR    = os.path.join(DATA_DIR, "processed")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR     = os.path.join(BASE_DIR, "logs")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")

for d in [DATA_DIR, PROC_DIR, CKPT_DIR, LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# 原始数据文件（需自行放入 data/ 目录）
SEQ_FILE  = os.path.join(DATA_DIR, "protein.SHS148k.sequences.dictionary.tsv")
PPI_FILE  = os.path.join(DATA_DIR, "protein.actions.SHS148k.txt")

# ───────────────────── 模型超参 ─────────────────────
NUM_CLASSES   = 7
NUM_LAYERS    = 2
HIDDEN_DIM    = 128
DROPOUT       = 0.2
RESIDUAL      = True
USE_BN        = True

# 节点编码器
ENC_INPUT_DIM = 22       # 20 氨基酸 + 1 gap + 1 unknown
ENC_HIDDEN    = 64
ENC_OUTPUT    = HIDDEN_DIM

# 边预测头
EDGE_MLP_HIDDEN = 256

# ───────────────────── 训练超参 ─────────────────────
BATCH_SIZE    = 4096
EPOCHS        = 200
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 20       # 早停耐心
GRAD_CLIP     = 1.0
SCHEDULER     = "cosine"  # cosine / step / none

# ───────────────────── 损失函数 ─────────────────────
LOSS_TYPE     = "bce"    # bce / focal
FOCAL_GAMMA   = 2.0
FOCAL_ALPHA   = 0.25

# ───────────────────── 数据划分 ─────────────────────
SPLIT_MODE    = "bfs"    # random / bfs / dfs
TRAIN_RATIO   = 0.6
VAL_RATIO     = 0.2
TEST_RATIO    = 0.2
SEED          = 42

# ───────────────────── 设备 ─────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────────────── 类别名称 ─────────────────────
CLASS_NAMES = [
    "activation",
    "binding",
    "catalysis",
    "expression",
    "inhibition",
    "ptmod",
    "reaction",
]

# ───────────────────── 负采样 ─────────────────────
NEGATIVE_RATIO = 1.0  # 负样本 : 正样本 比例
