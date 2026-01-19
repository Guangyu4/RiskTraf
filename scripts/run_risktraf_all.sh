#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

echo "========================================"
echo "LightSTGCN: All Datasets"
echo "========================================"

# 4 datasets on 3 GPUs
CUDA_VISIBLE_DEVICES=1 python train_light_stgcn.py --dataset PEMS03-B --hidden_dim 32 --num_blocks 4 --epochs 100 > logs_light_stgcn_PEMS03-B.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python train_light_stgcn.py --dataset PEMS04-B --hidden_dim 32 --num_blocks 4 --epochs 100 > logs_light_stgcn_PEMS04-B.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python train_light_stgcn.py --dataset PEMS07-B --hidden_dim 32 --num_blocks 4 --epochs 100 > logs_light_stgcn_PEMS07-B.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=1 python train_light_stgcn.py --dataset PEMS08-B --hidden_dim 32 --num_blocks 4 --epochs 100 > logs_light_stgcn_PEMS08-B.log 2>&1 &
PID4=$!

echo "Started all 4 datasets"
echo "PIDs: $PID1, $PID2, $PID3, $PID4"

wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "========================================"
echo "All Training Completed!"
echo "========================================"
echo ""

for ds in PEMS03-B PEMS04-B PEMS07-B PEMS08-B; do
    echo "=== $ds ==="
    grep -E "Test Results|Horizon|Overall" logs_light_stgcn_${ds}.log | tail -6
    echo ""
done
