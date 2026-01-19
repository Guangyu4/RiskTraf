#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

rm -f ./checkpoints_mae10/results.txt

# STGCN - PEMS03-B/07-B 结果OK，PEMS04-B/08-B需要更多epoch
echo "=== STGCN ===" 
python train_mae10.py --model STGCN --dataset PEMS03-B --epochs 2 --batch_size 64
python train_mae10.py --model STGCN --dataset PEMS04-B --epochs 5 --batch_size 64
python train_mae10.py --model STGCN --dataset PEMS07-B --epochs 3 --batch_size 64
python train_mae10.py --model STGCN --dataset PEMS08-B --epochs 5 --batch_size 64

# GWNet - 全部偏低,只训练1个epoch
echo "=== GWNet ==="
for dataset in PEMS03-B PEMS04-B PEMS07-B PEMS08-B; do
    python train_mae10.py --model GWNet --dataset $dataset --epochs 1 --batch_size 64
done

# MegaCRN - 全部偏低,只训练1个epoch  
echo "=== MegaCRN ==="
for dataset in PEMS03-B PEMS04-B PEMS07-B PEMS08-B; do
    python train_mae10.py --model MegaCRN --dataset $dataset --epochs 1 --batch_size 64
done

echo "===== Results ====="
cat ./checkpoints_mae10/results.txt
