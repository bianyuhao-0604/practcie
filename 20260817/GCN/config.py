import os
from pathlib import Path
#路径配置
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
#数据配置
TASK_TYPE = 'node'#支持两种任务模式: "node" (节点分类) / "graph" (图分类)
DATASET_NAME = 'Cora'# 数据集名称
CUSTOM_GRAPH_PATH = DATA_DIR / "my_graph.pt"# 自定义数据路径
#模型配置
MODEL_TYPE = "GCN"# GCN / GAT / GraphSAGE
NUM_LATERS = 2# 图卷积层数
HIDDEN_DIM = 64# 隐藏层维度
DROPOUT = 0.5# Dropout 比率
USE_BATCH_NORM = True# 是否在层间使用 BatchNorm
#预训练与微调
PRETRAINED = True# 是否加载预训练权重
PERTRAINED_PATH = OUTPUT_DIR / "pretrained_gcn.pth"
FREEZE_ENCODER = False# 是否冻结图编码器（仅训练分类头）
#训练配置
EPOCHS = 200#训练轮数
LEARNING_RATE = 1e-2#学习率
WEIGHT_DECAY = 5e-4#权重衰减
PATIENCE = 30# Early Stopping 耐心值
DEVICE = "cuda"# cuda / cpu
#交叉验证配置
N_SPLITS = 10# 划分次数 / 折数
TRAIN_RATTO = 0.6# 节点分类: 训练集比例
VAL_RATIO = 0.2# 节点分类: 验证集比例
RANDOM_SEED = 42#随机种子