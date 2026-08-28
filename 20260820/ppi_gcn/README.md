# 🧬 SHS148k GCN/GAT — 蛋白质相互作用预测

基于 **SHS148k** 数据集的图神经网络（GCN / GAT）模型，用于预测蛋白质-蛋白质相互作用（PPI）的类型。

---

## 📁 项目结构

```
ppi_gcn/
├── config.py              # 全局配置（路径、超参、设备）
├── aa_utils.py            # 氨基酸编码（one-hot / 组成向量）+ 数据加载
├── data_process.py        # 原始数据 → 特征矩阵 + 图边 + 三种划分
├── dataset.py             # PyTorch Dataset / DataLoader 封装
├── models/
│   ├── __init__.py        # PPINetwork 完整模型
│   ├── node_encoder.py    # MLP 节点编码器 (22 → 64 → 128)
│   ├── gnn_stack.py       # GCN / GAT 多层堆叠（残差 + BN）
│   └── edge_predictor.py  # 边预测头（双线性 + 拼接 → 7 类）
├── focal_loss.py          # 多标签 Focal Loss + 类别权重
├── evaluate.py            # 评估指标（F1 / AUC / MCC / Hamming）
├── train.py               # 训练主循环（AdamW + Cosine + 早停）
├── run_eval.py            # 独立评估 + 可视化
├── verify_all.py          # 完整离线验证套件
├── run.sh                 # 一键运行脚本
└── README.md
```

---

## 🏗️ 模型架构

```
┌──────────────────────────────────────────────────────────┐
│                     PPINetwork                           │
│                                                          │
│  氨基酸组成 (22维)                                      │
│       ↓                                                  │
│  ┌─────────────────────┐                                │
│  │   NodeEncoder       │  MLP + BN + ReLU              │
│  │   22 → 64 → 128    │                                │
│  └──────────┬──────────┘                                │
│             ↓  h [N, 128]                               │
│  ┌─────────────────────┐                                │
│  │   GNNStack × 3     │  GCNConv / GATConv            │
│  │   + 残差 + BN      │  + Dropout                    │
│  └──────────┬──────────┘                                │
│             ↓  z [N, 128]  (结构感知嵌入)               │
│                                                          │
│  对每条边 (u, v):                                       │
│  ┌─────────────────────────────────┐                    │
│  │  z_u ⊕ z_v ⊕ (z_u⊙z_v) ⊕|z_u−z_v|  │  → 512      │
│  │  + Bilinear(z_u, z_v)              │  → 128       │
│  │  → MLP: 512→256→128→7             │               │
│  │  → Sigmoid                            │               │
│  └─────────────────────────────────┘                    │
│             ↓                                            │
│  7 维多标签预测 [activation, binding, catalysis,         │
│                   expression, inhibition, ptmod, reaction]│
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib
# 如果使用 GAT，确保 torch-geometric 版本 ≥ 2.0
```

### 2. 准备数据

将以下文件放入 `data/` 目录：

| 文件 | 说明 | 下载地址 |
|---|---|---|
| `protein.SHS148k.sequences.dictionary.tsv` | 蛋白质序列 | [Zenodo](https://doi.org/10.5281/zenodo.15694560) |
| `protein.actions.SHS148k.txt` | PPI 注释 | 同上 |

> 💡 也可从 HuggingFace 下载预处理版本：`Synthyra/SHS148k` 或 `GleghornLab/ppi_SHS148k_bfs_2025`

### 3. 一键运行

```bash
# 默认：GCN + BFS 划分 + 200 epochs
bash run.sh

# GAT + BFS + 300 epochs + Focal Loss
bash run.sh gat bfs 300 --loss focal

# GCN + Random 划分 + 100 epochs
bash run.sh gcn random 100
```

### 4. 分步运行

```bash
# Step 1: 预处理
python data_process.py --seq_path data/protein.SHS148k.sequences.dictionary.tsv \
                       --ppi_path data/protein.actions.SHS148k.txt \
                       --out_dir data/processed --split_mode bfs

# Step 2: 训练
python train.py --gnn_type gcn --split_mode bfs --epochs 200

# Step 3: 独立评估
python run_eval.py --ckpt checkpoints/best_gcn_bfs.pt --plot
```

---

## ⚙️ 关键配置

`config.py` 中可调整：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `HIDDEN_DIM` | 128 | GNN 隐藏维度 |
| `NUM_LAYERS` | 3 | GNN 层数 |
| `DROPOUT` | 0.2 | Dropout 率 |
| `BATCH_SIZE` | 2048 | 边批次大小 |
| `LR` | 1e-3 | 学习率 |
| `WEIGHT_DECAY` | 1e-4 | AdamW 权重衰减 |
| `LOSS_TYPE` | "bce" | "bce" 或 "focal" |
| `SPLIT_MODE` | "bfs" | "random" / "bfs" / "dfs" |
| `PATIENCE` | 20 | 早停耐心 |
| `NEGATIVE_RATIO` | 1.0 | 负样本比例 |

---

## 📊 预期性能

| 划分模式 | Micro-F1 (Val) | Macro-F1 (Val) | 说明 |
|---|---|---|---|
| Random | ~0.90–0.93 | ~0.55–0.65 | 训练/测试蛋白高度重叠，指标虚高 |
| BFS | ~0.78–0.85 | ~0.40–0.55 | 测试集为局部密集子图 |
| DFS | ~0.72–0.80 | ~0.35–0.50 | 最难的泛化场景 |

> ⚠️ **Random 结果仅供参考**，BFS/DFS 的 NS（两蛋白都未见）子集才是模型真实泛化能力的体现。

---

## 🔬 三种划分模式说明

```
                    ┌─────────────────────────────────┐
                    │       PPI 网络（图）             │
                    │                                 │
   Random 划分:     │  ●────●────●────●────●         │
   (随机切边)        │       │         │              │
                    │  ●────●────●    ●────●         │
                    │       │         │              │
                    │  ●────●────●────●              │
                    └─────────────────────────────────┘
                        ▲ Train  ▲ Val  ▲ Test
                        (边随机分到三个集合)

                    ┌─────────────────────────────────┐
                    │       PPI 网络（图）             │
                    │                                 │
   BFS 划分:        │  ┌─Train Zone──┐               │
   (BFS 排序后切)    │  │ ●─●─●─●     │               │
                    │  │  │   │      │               │
                    │  │ ●─●─●      │               │
                    │  └─────────────┘               │
                    │       ┌─Val Zone──┐            │
                    │       │ ●─●       │            │
                    │       │  │        │            │
                    │       └────────────┘           │
                    │            ┌─Test Zone─┐       │
                    │            │ ●─●─●    │       │
                    │            │  │       │       │
                    │            └──────────┘       │
                    └─────────────────────────────────┘
                        (BFS 遍历顺序决定优先级)

   DFS 划分: 类似但按深度优先遍历排序，测试集更分散
```

---

## 📈 升级路线

| 阶段 | 改动 | 预期提升 |
|---|---|---|
| **当前基线** | 氨基酸组成 + GCN + BCE | F1μ ≈ 0.82 (Random) |
| + Focal Loss | 替换损失函数 | +0.01–0.02 |
| + ESM-2 特征 | NodeEncoder 输入换 1280 维蛋白语言模型嵌入 | +0.04–0.06 |
| + 结构图边 | 从 AlphaFold2 PDB 生成 k-NN / r-ball 边，双图融合 | +0.02–0.03 |
| + 对比预训练 | 两阶段训练（参考 JmcPPI） | +0.02 |
| + GAT 注意力 | 换 `gnn_type=gat` | +0.01–0.02 |

---

## 🧪 验证

运行完整离线验证（无需 GPU / torch-geometric）：

```bash
python verify_all.py
```

会检查：
- ✅ 所有文件存在
- ✅ 各模块可独立导入
- ✅ 模型前向传播 shape 正确
- ✅ 数据预处理流水线连通
- ✅ 训练循环无报错
- ✅ 评估指标计算正确

---

## ⚠️ 注意事项

1. **标签是多标签**：SHS148k 中一条 PPI 可同时属于多个类别，务必用 sigmoid + BCE/Focal，不可用 softmax
2. **负样本需自行采样**：原始数据只有正样本，预处理脚本会自动 1:1 采样负边
3. **Random 划分会虚高**：论文中必须报告 BFS/DFS 结果
4. **类别不平衡严重**：expression 仅占 ~0.16%，建议用 Focal Loss 或类别权重
5. **序列长度差异大**：氨基酸组成编码已做长度归一化，无截断问题

---

## 📄 License

MIT License. 数据集版权归原始发布者所有。
