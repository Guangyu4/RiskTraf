#!/usr/bin/env python3
import subprocess
import os
from datetime import datetime

MODELS = ["ST-SSDL", "Dish-TS", "STEVE"]
DATASETS = ["PEMS03-B", "PEMS04-B", "PEMS07-B", "PEMS08-B"]
RESULTS_FILE = "./checkpoints_ood/results.txt"
LOG_FILE = "./run_ood_experiments.log"

def get_completed():
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    completed.add((parts[0], parts[1]))
    return completed

def run_experiment(model, dataset):
    cmd = f"python train_ood_models.py --model {model} --dataset {dataset} --epochs 50 --batch_size 64"
    print(f"[{datetime.now()}] Running: {cmd}")
    
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{datetime.now()}] Running: {cmd}\n")
        result = subprocess.run(cmd, shell=True, stdout=f, stderr=f)
        if result.returncode != 0:
            print(f"Error running {model} on {dataset}")
            f.write(f"[{datetime.now()}] Error: return code {result.returncode}\n")
            return False
    print(f"[{datetime.now()}] Finished: {model} on {dataset}")
    return True

def main():
    os.chdir('/home/bd2/DB/PEMSB')
    print("Starting OOD experiments...")
    
    completed = get_completed()
    print(f"Already completed: {len(completed)} experiments")
    
    to_run = []
    for dataset in DATASETS:
        for model in MODELS:
            if (dataset, model) not in completed:
                to_run.append((model, dataset))
    
    if not to_run:
        print("All experiments completed!")
        return
    
    print(f"Experiments to run: {len(to_run)}")
    for i, (model, dataset) in enumerate(to_run):
        print(f"\n--- Experiment {i+1}/{len(to_run)}: {model} on {dataset} ---")
        run_experiment(model, dataset)
    
    print("\nAll experiments finished!")

if __name__ == "__main__":
    main()

