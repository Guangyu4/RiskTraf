import subprocess
import os
import sys
import datetime

BASE_COMMAND = "python train_baseline_3dim.py"
RESULTS_FILE = "./checkpoints_3dim/results.txt"
LOG_FILE = "./run_3dim_experiments.log"

MODELS = ["STGCN", "GWNet", "MegaCRN"]
DATASETS = ["PEMS03-B", "PEMS04-B", "PEMS07-B", "PEMS08-B"]


def get_completed():
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    completed.add((parts[0], parts[1]))
    return completed


def run_exp(model, dataset):
    cmd = f"{BASE_COMMAND} --model {model} --dataset {dataset} --epochs 50 --batch_size 64"
    print(f"Running: {cmd}")
    with open(LOG_FILE, 'a') as log_f:
        log_f.write(f"[{datetime.datetime.now()}] Running: {cmd}\n")
        proc = subprocess.run(cmd, shell=True, stdout=log_f, stderr=log_f)
        if proc.returncode != 0:
            print(f"Error: {model} on {dataset}")
            log_f.write(f"[{datetime.datetime.now()}] Error: {model} on {dataset}, code: {proc.returncode}\n")
            return False
    print(f"Done: {model} on {dataset}")
    return True


def main():
    os.makedirs("./checkpoints_3dim", exist_ok=True)
    print("Starting 3-dim experiments...")
    completed = get_completed()
    print(f"Already completed: {len(completed)}")
    
    to_run = []
    for dataset in DATASETS:
        for model in MODELS:
            if (dataset, model) not in completed:
                to_run.append((model, dataset))
    
    if not to_run:
        print("All experiments done.")
        return
    
    print(f"Experiments to run: {len(to_run)}")
    for i, (model, dataset) in enumerate(to_run):
        print(f"\n--- {i+1}/{len(to_run)}: {model} on {dataset} ---")
        if not run_exp(model, dataset):
            print(f"Stopped due to error")
            sys.exit(1)
    print("\nAll done.")


if __name__ == "__main__":
    main()

