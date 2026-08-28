# 超参数与路径配置
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_SIZE = 224
NUM_WORKERS = 4
MODEL_NAME = "resnet50"
NUM_CLASSES = 10
PRETRAINED = True
FREEZE_BACKBONE = False
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7
DEVICE = "cuda"
N_SPLITS = 5
SHUFFLE = True
RANDOM_SEED = 42