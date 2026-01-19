#!/usr/bin/env python
import subprocess
import os

datasets = ['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B']
env_modes = ['speed', 'occ']

os.makedirs('logs_env_mode', exist_ok=True)
os.makedirs('checkpoints_env_mode', exist_ok=True)

for mode in env_modes:
    for ds in datasets:
        log_file = f'logs_env_mode/{mode}_{ds}.log'
        cmd = [
            'python', 'train_light_stgcn.py',
            '--dataset', ds,
            '--env_mode', mode,
            '--save_dir', f'./checkpoints_env_mode/{mode}'
        ]
        print(f'Running: {" ".join(cmd)}')
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f'Done: {log_file}')
