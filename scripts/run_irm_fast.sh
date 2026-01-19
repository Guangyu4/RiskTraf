#!/bin/bash

SAVE_DIR="./checkpoints_irm_fast"
mkdir -p $SAVE_DIR

RESULTS_FILE="$SAVE_DIR/results.txt"
rm -f $RESULTS_FILE

for dataset in PEMS03-B PEMS04-B PEMS07-B PEMS08-B; do
    echo "=== Training on $dataset with IRM_FAST ==="
    python train_cf.py \
        --dataset $dataset \
        --cf_strategy irm_fast \
        --cf_weight 0.1 \
        --epochs 100 \
        --batch_size 64 \
        --save_dir $SAVE_DIR \
        --lr 0.001
done

echo "=== All training completed ==="
echo "Results saved to $SAVE_DIR"
