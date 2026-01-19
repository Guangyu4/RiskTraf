#!/bin/bash

cd /home/bd2/DB/PEMSB

DATASETS=("PEMS03-B" "PEMS04-B" "PEMS07-B" "PEMS08-B")
MODELS=("STGCN" "DCRNN" "GWNet" "MegaCRN")

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "=========================================="
        echo "Running $model on $dataset"
        echo "=========================================="
        python train_baseline.py --model $model --dataset $dataset --epochs 20 --batch_size 64
    done
done

echo "All experiments completed!"
echo "Results saved in checkpoints_baseline/results.txt"

