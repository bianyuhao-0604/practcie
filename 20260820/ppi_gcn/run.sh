#!/bin/bash
# run.sh — 一键运行脚本
# 用法:
#   bash run.sh                # 默认 GCN + BFS
#   bash run.sh gat bfs 200   # GAT + BFS + 200 epochs
#   bash run.sh gcn random 100 --loss focal  # GCN + Random + Focal Loss

set -e

cd "$(dirname "$0")"

GNN_TYPE=${1:-gcn}
SPLIT_MODE=${2:-bfs}
EPOCHS=${3:-200}
EXTRA_ARGS="${@:4}"

echo "════════════════════════════════════════════"
echo "  SHS148k PPI-GNN 训练"
echo "  GNN: ${GNN_TYPE^^}  Split: ${SPLIT_MODE^^}  Epochs: ${EPOCHS}"
echo "  Extra: ${EXTRA_ARGS}"
echo "════════════════════════════════════════════"

# Step 1: 检查数据
if [ ! -f "data/protein.SHS148k.sequences.dictionary.tsv" ] || \
   [ ! -f "data/protein.actions.SHS148k.txt" ]; then
    echo ""
    echo "⚠️  原始数据文件缺失！"
    echo "请将以下文件放入 data/ 目录："
    echo "  - protein.SHS148k.sequences.dictionary.tsv"
    echo "  - protein.actions.SHS148k.txt"
    echo ""
    echo "可从以下地址下载："
    echo "  Zenodo: https://doi.org/10.5281/zenodo.15694560"
    echo "  HuggingFace: https://huggingface.co/datasets/Synthyra/SHS148k"
    echo ""
    exit 1
fi

# Step 2: 预处理
PROC_FILE="data/processed/SHS148k_${SPLIT_MODE}.pt"
if [ ! -f "${PROC_FILE}" ]; then
    echo ""
    echo "▶ Step 1: 数据预处理 (${SPLIT_MODE})"
    python data_process.py \
        --seq_path data/protein.SHS148k.sequences.dictionary.tsv \
        --ppi_path data/protein.actions.SHS148k.txt \
        --out_dir data/processed \
        --split_mode ${SPLIT_MODE}
else
    echo "✅ 预处理文件已存在: ${PROC_FILE} (跳过)"
fi

# Step 3: 训练
echo ""
echo "▶ Step 2: 训练模型"
python train.py \
    --gnn_type ${GNN_TYPE} \
    --split_mode ${SPLIT_MODE} \
    --epochs ${EPOCHS} \
    --data_path ${PROC_FILE} \
    ${EXTRA_ARGS}

# Step 4: 评估
CKPT="checkpoints/best_${GNN_TYPE}_${SPLIT_MODE}.pt"
if [ -f "${CKPT}" ]; then
    echo ""
    echo "▶ Step 3: 独立评估 + 绘图"
    python run_eval.py --ckpt ${CKPT} --plot
fi

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ 全部完成！"
echo "  检查点: ${CKPT}"
echo "  日志  : logs/history_${GNN_TYPE}_${SPLIT_MODE}.json"
echo "  曲线  : plots/curves_${GNN_TYPE}_${SPLIT_MODE}.png"
echo "════════════════════════════════════════════"
