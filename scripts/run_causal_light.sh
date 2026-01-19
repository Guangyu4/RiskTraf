#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

SAVE_DIR="./checkpoints_causal_light"
mkdir -p $SAVE_DIR

echo "========================================"
echo "CausalLight Model Training"
echo "========================================"

for dataset in PEMS03-B PEMS04-B PEMS07-B PEMS08-B; do
    echo ""
    echo "========================================"
    echo "Training CausalLight on $dataset"
    echo "========================================"
    
    python train_causal_light.py \
        --dataset $dataset \
        --hidden_dim 64 \
        --num_layers 2 \
        --num_heads 4 \
        --vrex_weight 0.5 \
        --contrast_weight 0.1 \
        --num_envs 3 \
        --warmup_epochs 5 \
        --patience 20 \
        --epochs 100 \
        --batch_size 64 \
        --lr 0.002 \
        --save_dir $SAVE_DIR \
        --use_contrast \
        --use_intervention
    
    echo "Finished training $dataset"
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"

# Collect results
python -c "
import re

results = []
for dataset in ['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B']:
    try:
        import glob
        log_files = glob.glob(f'./logs_causal_light/{dataset}/train_*.log')
        if log_files:
            log_file = max(log_files)
            with open(log_file, 'r') as f:
                content = f.read()
            
            pattern = r'Test Results.*?Overall.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+).*?Horizon\s+3.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+).*?Horizon\s+6.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+).*?Horizon\s+12.*?MAE=([\d.]+).*?RMSE=([\d.]+).*?MAPE=([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                g = match.groups()
                results.append(f'{dataset},CausalLight,{g[3]},{g[4]},{g[5]},{g[6]},{g[7]},{g[8]},{g[9]},{g[10]},{g[11]},{g[0]},{g[1]},{g[2]}')
    except Exception as e:
        print(f'Error processing {dataset}: {e}')

with open('./checkpoints_causal_light/results.txt', 'w') as f:
    f.write('\n'.join(results) + '\n')

print('Results:')
for r in results:
    print(r)
"
