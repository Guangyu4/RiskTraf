#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

SAVE_DIR="./checkpoints_rex"
mkdir -p $SAVE_DIR
RESULTS_FILE="$SAVE_DIR/results.txt"
rm -f $RESULTS_FILE

for dataset in PEMS03-B PEMS04-B PEMS07-B PEMS08-B; do
    echo "========================================"
    echo "Training V-REx on $dataset"
    echo "========================================"
    
    python train_rex.py \
        --dataset $dataset \
        --rex_weight 1.0 \
        --num_envs 3 \
        --env_split magnitude \
        --warmup_epochs 5 \
        --patience 15 \
        --epochs 100 \
        --batch_size 64 \
        --save_dir $SAVE_DIR \
        --lr 0.001 \
        --num_layers 3
    
    echo "Finished training $dataset"
done

echo "========================================"
echo "Collecting results..."
echo "========================================"

for dataset in PEMS03-B PEMS04-B PEMS07-B PEMS08-B; do
    python train_rex.py \
        --dataset $dataset \
        --eval_only \
        --save_dir $SAVE_DIR 2>&1 | tee -a eval_$dataset.log
done

echo "========================================"
echo "All experiments completed!"
echo "========================================"
