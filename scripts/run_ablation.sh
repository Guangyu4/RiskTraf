#!/bin/bash
cd /home/bd2/DB/PEMSB
source /home/bd2/miniconda3/etc/profile.d/conda.sh
conda activate BasicTS

DATASETS="PEMS03-B PEMS04-B PEMS07-B PEMS08-B"
ABLATIONS="irm no_rex random_rex"

# GPU 分配
declare -A GPU_MAP
GPU_MAP["irm"]=1
GPU_MAP["no_rex"]=2
GPU_MAP["random_rex"]=3

for ablation in $ABLATIONS; do
    gpu=${GPU_MAP[$ablation]}
    for ds in $DATASETS; do
        echo "Starting $ablation on $ds (GPU $gpu)"
        CUDA_VISIBLE_DEVICES=$gpu nohup python train_ablation.py \
            --dataset $ds \
            --ablation $ablation \
            --epochs 100 \
            --patience 15 \
            > logs_ablation_${ablation}_${ds}.log 2>&1 &
        sleep 2
    done
done

echo "All ablation experiments started!"
echo "Check logs: logs_ablation_*.log"
