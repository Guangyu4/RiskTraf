#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

DATASET="PEMS03-B"
LOG_DIR="./logs_param_search"
mkdir -p $LOG_DIR

# rex_weight search
for val in 0.01 0.05 0.1 0.2 0.5; do
    echo "Running rex_weight=$val"
    CUDA_VISIBLE_DEVICES=0 python train_light_stgcn.py \
        --dataset $DATASET --rex_weight $val \
        --save_dir ./ckpt_param_search/rex_weight_$val \
        > $LOG_DIR/rex_weight_${val}.log 2>&1 &
done
wait

# num_envs search
for val in 2 3 4 6 8; do
    echo "Running num_envs=$val"
    CUDA_VISIBLE_DEVICES=0 python train_light_stgcn.py \
        --dataset $DATASET --num_envs $val \
        --save_dir ./ckpt_param_search/num_envs_$val \
        > $LOG_DIR/num_envs_${val}.log 2>&1 &
done
wait

# hidden_dim search
for val in 16 24 32 48 64; do
    echo "Running hidden_dim=$val"
    CUDA_VISIBLE_DEVICES=0 python train_light_stgcn.py \
        --dataset $DATASET --hidden_dim $val \
        --save_dir ./ckpt_param_search/hidden_dim_$val \
        > $LOG_DIR/hidden_dim_${val}.log 2>&1 &
done
wait

# num_blocks search
for val in 2 3 4 5 6; do
    echo "Running num_blocks=$val"
    CUDA_VISIBLE_DEVICES=0 python train_light_stgcn.py \
        --dataset $DATASET --num_blocks $val \
        --save_dir ./ckpt_param_search/num_blocks_$val \
        > $LOG_DIR/num_blocks_${val}.log 2>&1 &
done
wait

echo "All experiments done!"
