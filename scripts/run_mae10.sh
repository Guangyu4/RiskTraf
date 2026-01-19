#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

models=("STGCN" "GWNet" "MegaCRN")
datasets=("PEMS03-B" "PEMS04-B" "PEMS07-B" "PEMS08-B")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "===== Training $model on $dataset ====="
        python train_mae10.py --model $model --dataset $dataset --epochs 2 --batch_size 64
    done
done

echo "===== All training complete ====="
cat ./checkpoints_mae10/results.txt
