#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

SAVE_DIR="./checkpoints_causal_rex"
mkdir -p $SAVE_DIR

echo "========================================"
echo "CausalREx Model - Parallel Training on 4 GPUs"
echo "========================================"

# 4个数据集分配到4张GPU并行训练
CUDA_VISIBLE_DEVICES=1 python train_causal_light.py \
    --dataset PEMS03-B \
    --hidden_dim 64 --num_layers 2 --num_heads 4 \
    --vrex_weight 0.5 --contrast_weight 0.1 --num_envs 3 \
    --warmup_epochs 5 --patience 20 --epochs 100 \
    --batch_size 64 --lr 0.002 \
    --save_dir $SAVE_DIR \
    --use_contrast --use_intervention \
    > logs_causal_rex_PEMS03-B.log 2>&1 &
PID1=$!
echo "Started PEMS03-B on GPU 1, PID=$PID1"

CUDA_VISIBLE_DEVICES=2 python train_causal_light.py \
    --dataset PEMS04-B \
    --hidden_dim 64 --num_layers 2 --num_heads 4 \
    --vrex_weight 0.5 --contrast_weight 0.1 --num_envs 3 \
    --warmup_epochs 5 --patience 20 --epochs 100 \
    --batch_size 64 --lr 0.002 \
    --save_dir $SAVE_DIR \
    --use_contrast --use_intervention \
    > logs_causal_rex_PEMS04-B.log 2>&1 &
PID2=$!
echo "Started PEMS04-B on GPU 2, PID=$PID2"

CUDA_VISIBLE_DEVICES=3 python train_causal_light.py \
    --dataset PEMS07-B \
    --hidden_dim 64 --num_layers 2 --num_heads 4 \
    --vrex_weight 0.5 --contrast_weight 0.1 --num_envs 3 \
    --warmup_epochs 5 --patience 20 --epochs 100 \
    --batch_size 64 --lr 0.002 \
    --save_dir $SAVE_DIR \
    --use_contrast --use_intervention \
    > logs_causal_rex_PEMS07-B.log 2>&1 &
PID3=$!
echo "Started PEMS07-B on GPU 3, PID=$PID3"

CUDA_VISIBLE_DEVICES=1 python train_causal_light.py \
    --dataset PEMS08-B \
    --hidden_dim 64 --num_layers 2 --num_heads 4 \
    --vrex_weight 0.5 --contrast_weight 0.1 --num_envs 3 \
    --warmup_epochs 5 --patience 20 --epochs 100 \
    --batch_size 64 --lr 0.002 \
    --save_dir $SAVE_DIR \
    --use_contrast --use_intervention \
    > logs_causal_rex_PEMS08-B.log 2>&1 &
PID4=$!
echo "Started PEMS08-B on GPU 1 (shared), PID=$PID4"

echo ""
echo "All 4 training processes started!"
echo "PIDs: $PID1, $PID2, $PID3, $PID4"
echo ""
echo "Monitor with:"
echo "  tail -f logs_causal_rex_PEMS03-B.log"
echo "  tail -f logs_causal_rex_PEMS04-B.log"
echo "  tail -f logs_causal_rex_PEMS07-B.log"
echo "  tail -f logs_causal_rex_PEMS08-B.log"
echo ""

# 等待所有进程完成
wait $PID1 $PID2 $PID3 $PID4

echo "========================================"
echo "All training completed!"
echo "========================================"

# 收集结果
python -c "
import re
import glob

results = []
for dataset in ['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B']:
    try:
        log_files = glob.glob(f'./logs_causal_light/{dataset}/train_*.log')
        if log_files:
            log_file = max(log_files)
            with open(log_file, 'r') as f:
                content = f.read()
            
            pattern = r'Test Results.*?Overall.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+).*?Horizon\s+3.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+).*?Horizon\s+6.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+).*?Horizon\s+12.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                g = match.groups()
                results.append(f'{dataset},CausalREx,{g[3]},{g[4]},{g[5]},{g[6]},{g[7]},{g[8]},{g[9]},{g[10]},{g[11]},{g[0]},{g[1]},{g[2]}')
    except Exception as e:
        print(f'Error processing {dataset}: {e}')

with open('./checkpoints_causal_rex/results.txt', 'w') as f:
    f.write('\n'.join(results) + '\n')

print('Results saved to checkpoints_causal_rex/results.txt')
for r in results:
    print(r)
"
