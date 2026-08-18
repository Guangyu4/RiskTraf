from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

from .data import DATASETS
from .models import MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paired PEMSB-3V baseline/RiskTraf matrix.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--plugins", nargs="+", default=["baseline", "risk"])
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--paired", action="store_true", help="Run baseline then plugin for each model on the same checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = list(itertools.product(args.datasets, args.models)) if args.paired else list(itertools.product(args.datasets, args.models, args.plugins))
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    device_queue: Queue[str] = Queue()
    for device in args.devices:
        device_queue.put(device)

    def result_path(dataset: str, model: str, plugin: str) -> Path:
        return Path(args.output_dir) / dataset / model.upper() / plugin / f"seed{args.seed}_h12" / "result.json"

    def is_done(dataset: str, model: str, plugin: str) -> bool:
        path = result_path(dataset, model, plugin)
        if not path.exists():
            return False
        try:
            json.loads(path.read_text())
        except json.JSONDecodeError:
            return False
        return True

    def call(cmd, log_path):
        proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path.write_text(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"Job failed: {' '.join(cmd)}\nSee {log_path}")

    def run_job(job):
        device = device_queue.get()
        log_dir = Path(args.output_dir) / "_matrix_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            if args.paired:
                dataset, model = job
                completed = []
                for plugin in args.plugins:
                    if is_done(dataset, model, plugin):
                        completed.append(result_path(dataset, model, plugin))
                        continue
                    cmd = [
                        sys.executable,
                        "-m",
                        "risktraf.trainer",
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                        "--plugin",
                        plugin,
                        "--device",
                        device,
                        "--epochs",
                        str(args.epochs),
                        "--patience",
                        str(args.patience),
                        "--seed",
                        str(args.seed),
                        "--output_dir",
                        args.output_dir,
                    ]
                    if plugin != "baseline":
                        ckpt = Path(args.output_dir) / dataset / model.upper() / "baseline" / f"seed{args.seed}_h12" / "best.pt"
                        cmd.extend(["--init_checkpoint", str(ckpt), "--skip_backbone_train", "--calibrate"])
                    if args.smoke:
                        cmd.append("--smoke")
                    log_path = log_dir / f"{dataset}_{model}_{plugin}.log"
                    call(cmd, log_path)
                    completed.append(log_path)
                return completed[-1]

            dataset, model, plugin = job
            if is_done(dataset, model, plugin):
                return result_path(dataset, model, plugin)
            cmd = [
                sys.executable,
                "-m",
                "risktraf.trainer",
                "--dataset",
                dataset,
                "--model",
                model,
                "--plugin",
                plugin,
                "--device",
                device,
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--seed",
                str(args.seed),
                "--output_dir",
                args.output_dir,
            ]
            if plugin != "baseline":
                cmd.append("--calibrate")
            if args.smoke:
                cmd.append("--smoke")
            log_path = log_dir / f"{dataset}_{model}_{plugin}.log"
            call(cmd, log_path)
            return log_path
        finally:
            device_queue.put(device)

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            try:
                print(f"completed {future.result()}", flush=True)
            except Exception as exc:
                failures.append(exc)
                print(f"failed {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} matrix jobs failed")


if __name__ == "__main__":
    main()
