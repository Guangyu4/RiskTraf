#!/usr/bin/env python3
import subprocess
import os
import sys

models = ['STGCN', 'DCRNN', 'GWNet', 'MegaCRN']
datasets = ['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B']

results_file = './checkpoints_baseline/results.txt'
completed = set()

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    dataset = parts[0]
                    model = parts[1]
                    completed.add((dataset, model))

print(f"已完成实验: {len(completed)}")
print(f"已完成组合: {completed}")

remaining = []
for dataset in datasets:
    for model in models:
        if (dataset, model) not in completed:
            remaining.append((dataset, model))

print(f"\n剩余实验: {len(remaining)}")
for dataset, model in remaining:
    print(f"  {dataset} - {model}")

if not remaining:
    print("\n所有实验已完成！")
    sys.exit(0)

print(f"\n开始运行剩余实验...")
print("=" * 60)

for i, (dataset, model) in enumerate(remaining, 1):
    print(f"\n[{i}/{len(remaining)}] 运行 {dataset} - {model}")
    print("=" * 60)
    
    cmd = [
        'python', 'train_baseline.py',
        '--model', model,
        '--dataset', dataset,
        '--epochs', '20',
        '--batch_size', '64'
    ]
    
    result = subprocess.run(cmd, cwd='/home/bd2/DB/PEMSB')
    
    if result.returncode != 0:
        print(f"\n错误: {dataset} - {model} 运行失败 (退出码: {result.returncode})")
        print("停止运行剩余实验")
        sys.exit(1)
    else:
        print(f"\n完成: {dataset} - {model}")

print("\n" + "=" * 60)
print("所有实验已完成！")

