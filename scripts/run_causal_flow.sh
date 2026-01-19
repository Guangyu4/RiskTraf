#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

SAVE_DIR="./checkpoints_causal_flow"
mkdir -p $SAVE_DIR

echo "========================================"
echo "CausalFlow-REx: Flow-centric with Speed/Occ environment"
echo "========================================"

# 4个数据集在4张GPU并行训练
CUDA_VISIBLE_DEVICES=1 python train_causal_flow.py \
    --dataset PEMS03-B \
    --hidden_dim 64 --num_heads 4 \
    --rex_weight 1.0 --num_envs 3 --env_type speed_occ \
    --warmup_epochs 5 --patience 20 --epochs 100 \
    --batch_size 64 --lr 0.002 \
    --save_dir $SAVE_DIR \
    > logs_causal_flow_PEMS03-B.log 2>&1 &
PID1=$!
echo "PEMS03-B on GPU 1, PID=$PID1"

CUDA_VISIBLE_DEVICES=2 python train_causal_flow.py \
    --dataset PEMS04-B \
    --hidden_dim 64 --num_heads 4 \
    --rex_weight 1.0 --num_envs 3 --env_type speed_occ \
    --warmup_epochs 5 --patience 20 --epochs 100 \
    --batch_size 64 --lr 0.002 \
    --save_dir $SAVE_DIR \
    > logs_causal_flow_PEMS04-B.log 2>&1 &
PID2=$!
echo "PEMS04-B on GPU 2, PID=$PID2"

CUDA_VISIBLE_DEVICES=3 python train_causal_flow.py \
    --dataset PEMS07-B \
    --hidden_dim 64 --num_heads 4 \
    --rex_weight 1.0 --num_envs 3 --env_type speed_occ \
    --warmup_epochs 5 --patience 20 --epochs 100 \
    --batch_size 64 --lr 0.002 \
    --save_dir $SAVE_DIR \
    > logs_causal_flow_PEMS07-B.log 2>&1 &
PID3=$!
echo "PEMS07-B on GPU 3, PID=$PID3"

CUDA_VISIBLE_DEVICES=1 python train_causal_flow.py \
    --dataset PEMS08-B \
    --hidden_dim 64 --num_heads 4 \
    --rex_weight 1.0 --num_envs 3 --env_type speed_occ \
    --warmup_epochs 5 --patience 20 --epochs 100 \
    --batch_size 64 --lr 0.002 \
    --save_dir $SAVE_DIR \
    > logs_causal_flow_PEMS08-B.log 2>&1 &
PID4=$!
echo "PEMS08-B on GPU 1 (shared), PID=$PID4"

echo ""
echo "All 4 processes started!"
echo "Monitor: tail -f logs_causal_flow_PEMS03-B.log"
echo ""

wait $PID1 $PID2 $PID3 $PID4

echo "========================================"
echo "All training completed! Collecting results..."
echo "========================================"

# 收集结果
python -c "
import re
import glob

results = []
for dataset in ['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B']:
    try:
        log_files = glob.glob(f'./logs_causal_flow/{dataset}/train_*.log')
        if log_files:
            log_file = max(log_files)
            with open(log_file, 'r') as f:
                content = f.read()
            
            pattern = r'Test Results.*?Overall.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+).*?Horizon\s+3.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+).*?Horizon\s+6.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+).*?Horizon\s+12.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                g = match.groups()
                results.append(f'{dataset},CausalFlow,{g[3]},{g[4]},{g[5]},{g[6]},{g[7]},{g[8]},{g[9]},{g[10]},{g[11]},{g[0]},{g[1]},{g[2]}')
                print(f'{dataset}: MAE={g[0]}')
    except Exception as e:
        print(f'Error {dataset}: {e}')

if results:
    with open('./checkpoints_causal_flow/results.txt', 'w') as f:
        f.write('\n'.join(results) + '\n')
    print('\\nResults saved!')
"
