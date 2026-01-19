#!/bin/bash
# 多Backbone实验并行运行脚本
# 使用4个GPU (0,1,2,3) 并行运行，每个GPU上串行运行4个backbone

cd /home/bd2/DB/PEMSB

source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate ANATS

DATASETS=("PEMS03-B" "PEMS04-B" "PEMS07-B" "PEMS08-B")
BACKBONES=("transformer" "gru" "mlp")

LOG_DIR="./logs_backbone"
CKPT_DIR="./checkpoints_backbone"

mkdir -p $LOG_DIR
mkdir -p $CKPT_DIR

echo "Starting backbone experiments at $(date)"

# 每个GPU负责一个数据集，串行运行4个backbone
run_gpu() {
    gpu=$1
    dataset=$2
    for backbone in "${BACKBONES[@]}"; do
        echo "[GPU $gpu] Starting $backbone on $dataset"
        python train_backbone.py \
            --backbone $backbone \
            --dataset $dataset \
            --gpu $gpu \
            --hidden_dim 32 \
            --num_blocks 4 \
            --epochs 100 \
            --batch_size 64 \
            --save_dir $CKPT_DIR \
            > "${LOG_DIR}/${backbone}_${dataset}.log" 2>&1
        echo "[GPU $gpu] Finished $backbone on $dataset"
    done
}

# 4个GPU并行，每个GPU上串行运行
run_gpu 0 "PEMS03-B" &
run_gpu 1 "PEMS04-B" &
run_gpu 2 "PEMS07-B" &
run_gpu 3 "PEMS08-B" &

wait

echo "All experiments completed at $(date)"
echo "Results saved in $CKPT_DIR"
echo "Run 'python generate_backbone_table.py' to generate LaTeX table"
