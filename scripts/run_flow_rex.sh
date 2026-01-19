#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

SAVE_DIR="./checkpoints_flow_rex"
mkdir -p $SAVE_DIR

echo "========================================"
echo "FlowREx: Input=Flow only, Env=Speed/Occ"
echo "========================================"

# 4个数据集在3张GPU并行训练
CUDA_VISIBLE_DEVICES=1 python train_flow_rex.py \
    --dataset PEMS03-B \
    --hidden_dim 48 --num_layers 2 --num_heads 4 \
    --rex_weight 0.5 --num_envs 4 \
    --warmup_epochs 10 --patience 25 --epochs 100 \
    --batch_size 64 --lr 0.001 \
    --save_dir $SAVE_DIR \
    > logs_flow_rex_PEMS03-B.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python train_flow_rex.py \
    --dataset PEMS04-B \
    --hidden_dim 48 --num_layers 2 --num_heads 4 \
    --rex_weight 0.5 --num_envs 4 \
    --warmup_epochs 10 --patience 25 --epochs 100 \
    --batch_size 64 --lr 0.001 \
    --save_dir $SAVE_DIR \
    > logs_flow_rex_PEMS04-B.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python train_flow_rex.py \
    --dataset PEMS07-B \
    --hidden_dim 48 --num_layers 2 --num_heads 4 \
    --rex_weight 0.5 --num_envs 4 \
    --warmup_epochs 10 --patience 25 --epochs 100 \
    --batch_size 64 --lr 0.001 \
    --save_dir $SAVE_DIR \
    > logs_flow_rex_PEMS07-B.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=1 python train_flow_rex.py \
    --dataset PEMS08-B \
    --hidden_dim 48 --num_layers 2 --num_heads 4 \
    --rex_weight 0.5 --num_envs 4 \
    --warmup_epochs 10 --patience 25 --epochs 100 \
    --batch_size 64 --lr 0.001 \
    --save_dir $SAVE_DIR \
    > logs_flow_rex_PEMS08-B.log 2>&1 &
PID4=$!

echo "Started: PEMS03-B(GPU1), PEMS04-B(GPU2), PEMS07-B(GPU3), PEMS08-B(GPU1)"
echo "PIDs: $PID1, $PID2, $PID3, $PID4"
echo ""
echo "Monitor: tail -f logs_flow_rex_PEMS03-B.log"
echo ""

wait $PID1 $PID2 $PID3 $PID4

echo "========================================"
echo "All training completed!"
echo "========================================"

grep -A 5 "Test Results" logs_flow_rex_*.log
