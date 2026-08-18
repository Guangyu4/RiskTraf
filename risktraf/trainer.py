from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .data import DATASETS, FeatureScaler, build_dataloaders
from .models import MODEL_NAMES, build_model
from .risk import RiskExtrapolationLoss, RiskLossConfig, masked_mae, masked_mape, masked_rmse


DEFAULT_BATCH = {
    "GMAN": 16,
    "HIMNET": 16,
    "STAEFORMER": 16,
    "STWA": 16,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_model_name(name: str) -> str:
    upper = name.upper()
    aliases = {"GRAPHWAVENET": "GWNET", "GWN": "GWNET", "STAE": "STAEFORMER"}
    return aliases.get(upper, upper)


def to_device(batch, device: torch.device):
    x, y, cal = batch
    return x.float().to(device), y.float().to(device), cal.float().to(device)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    scaler: FeatureScaler,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    preds, labels, inputs = [], [], []
    for batch in loader:
        x, y, cal = to_device(batch, device)
        pred = model(x, cal)
        preds.append(scaler.inverse_flow(pred).detach().cpu())
        labels.append(scaler.inverse_flow(y).detach().cpu())
        inputs.append(x.detach().cpu())
    return torch.cat(preds, 0), torch.cat(labels, 0), torch.cat(inputs, 0)


def metric_dict(pred: torch.Tensor, true: torch.Tensor, horizons: Iterable[int]) -> Dict[str, float]:
    out: Dict[str, float] = {
        "mae_all": float(masked_mae(pred, true)),
        "rmse_all": float(masked_rmse(pred, true)),
        "mape_all": float(masked_mape(pred, true)),
    }
    for h in horizons:
        if h <= pred.shape[1]:
            hp = pred[:, h - 1 : h]
            ht = true[:, h - 1 : h]
            out[f"mae_h{h}"] = float(masked_mae(hp, ht))
            out[f"rmse_h{h}"] = float(masked_rmse(hp, ht))
            out[f"mape_h{h}"] = float(masked_mape(hp, ht))
    return out


class EnvResidualCalibrator:
    def __init__(self, num_bins: int = 4, strength: float = 0.5, env_mode: str = "speed_occ") -> None:
        self.num_bins = num_bins
        self.strength = strength
        self.env_mode = env_mode
        self.edges: Optional[torch.Tensor] = None
        self.residuals: Optional[torch.Tensor] = None

    def score_from_x(self, x: torch.Tensor) -> torch.Tensor:
        speed = x[..., 1].mean(dim=(1, 2))
        occ = x[..., 2].mean(dim=(1, 2))
        speed_z = (speed - speed.mean()) / (speed.std(unbiased=False) + 1e-6)
        occ_z = (occ - occ.mean()) / (occ.std(unbiased=False) + 1e-6)
        if self.env_mode == "flow":
            flow = x[..., 0].mean(dim=(1, 2))
            return (flow - flow.mean()) / (flow.std(unbiased=False) + 1e-6)
        if self.env_mode == "speed":
            return -speed_z
        if self.env_mode == "occ":
            return occ_z
        return occ_z - speed_z

    def fit(self, pred: torch.Tensor, true: torch.Tensor, x: torch.Tensor) -> None:
        score = self.score_from_x(x)
        quantiles = torch.linspace(0, 1, self.num_bins + 1)
        self.edges = torch.quantile(score, quantiles)
        bins = torch.bucketize(score, self.edges[1:-1])
        residuals = []
        for b in range(self.num_bins):
            mask = bins == b
            if mask.any():
                residuals.append((true[mask] - pred[mask]).mean(dim=(0, 2, 3)))
            else:
                residuals.append(torch.zeros(pred.shape[1]))
        self.residuals = torch.stack(residuals, dim=0)

    def transform(self, pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.edges is None or self.residuals is None:
            return pred
        score = self.score_from_x(x)
        bins = torch.bucketize(score, self.edges[1:-1]).clamp(0, self.num_bins - 1)
        correction = self.residuals[bins].to(pred.device).view(pred.shape[0], pred.shape[1], 1, 1)
        return pred + self.strength * correction


def fit_best_calibrator(
    val_pred: torch.Tensor,
    val_true: torch.Tensor,
    val_x: torch.Tensor,
    test_pred: torch.Tensor,
    test_x: torch.Tensor,
    num_bins: int,
    strengths: Iterable[float],
    env_mode: str,
) -> Tuple[torch.Tensor, float]:
    split = max(1, val_pred.shape[0] // 2)
    fit_slice = slice(0, split)
    select_slice = slice(split, None)
    if val_pred[select_slice].numel() == 0:
        fit_slice = slice(0, val_pred.shape[0])
        select_slice = slice(0, val_pred.shape[0])

    best_strength = 0.0
    best_val = float(masked_mae(val_pred[select_slice], val_true[select_slice]))
    for strength in strengths:
        calibrator = EnvResidualCalibrator(num_bins=num_bins, strength=float(strength), env_mode=env_mode)
        calibrator.fit(val_pred[fit_slice], val_true[fit_slice], val_x[fit_slice])
        tuned_val = calibrator.transform(val_pred[select_slice], val_x[select_slice])
        tuned_mae = float(masked_mae(tuned_val, val_true[select_slice]))
        if tuned_mae < best_val * 0.999:
            best_val = tuned_mae
            best_strength = float(strength)
    if best_strength <= 0:
        return test_pred, 0.0

    calibrator = EnvResidualCalibrator(num_bins=num_bins, strength=best_strength, env_mode=env_mode)
    calibrator.fit(val_pred, val_true, val_x)
    return calibrator.transform(test_pred, test_x), best_strength


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: FeatureScaler,
    risk_loss: RiskExtrapolationLoss,
    device: torch.device,
    epoch: int,
    clip_grad: float,
    aux_weight: float,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {
        "loss": 0.0,
        "pred_loss": 0.0,
        "rex": 0.0,
        "pair": 0.0,
        "extrap": 0.0,
        "aux": 0.0,
        "debias": 0.0,
    }
    count = 0
    for batch in loader:
        x, y, cal = to_device(batch, device)
        pred = model(x, cal, y_scaled=y)
        pred_raw = scaler.inverse_flow(pred)
        true_raw = scaler.inverse_flow(y)
        loss, detail = risk_loss(pred_raw, true_raw, x, epoch)
        aux = model.auxiliary_loss(pred_raw, true_raw)
        debias_reg = model.debias_regularization()
        loss = loss + aux_weight * aux + 0.001 * debias_reg
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["aux"] += float(aux.detach())
        totals["debias"] += float(debias_reg.detach())
        for key in ("pred_loss", "rex", "pair", "extrap"):
            totals[key] += detail[key]
        count += 1
    return {k: v / max(1, count) for k, v in totals.items()}


def train_debias_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: FeatureScaler,
    risk_loss: RiskExtrapolationLoss,
    device: torch.device,
    epoch: int,
    clip_grad: float,
) -> Dict[str, float]:
    model.train()
    if hasattr(model, "backbone"):
        model.backbone.eval()
    totals = {"loss": 0.0, "pred_loss": 0.0, "rex": 0.0, "pair": 0.0, "extrap": 0.0, "debias": 0.0}
    count = 0
    for batch in loader:
        x, y, cal = to_device(batch, device)
        pred = model(x, cal, y_scaled=None)
        pred_raw = scaler.inverse_flow(pred)
        true_raw = scaler.inverse_flow(y)
        loss, detail = risk_loss(pred_raw, true_raw, x, epoch)
        debias_reg = model.debias_regularization()
        loss = loss + 0.0005 * debias_reg
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), clip_grad)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["debias"] += float(debias_reg.detach())
        for key in ("pred_loss", "rex", "pair", "extrap"):
            totals[key] += detail[key]
        count += 1
    return {k: v / max(1, count) for k, v in totals.items()}


def evaluate(model, loader, scaler: FeatureScaler, device: torch.device, horizons: Iterable[int]) -> Dict[str, float]:
    pred, true, _ = collect_predictions(model, loader, scaler, device)
    return metric_dict(pred, true, horizons)


def run_single(args: argparse.Namespace) -> Dict[str, float]:
    seed_everything(args.seed)
    model_name = canonical_model_name(args.model)
    dataset = args.dataset.upper()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset}")

    plugin = args.plugin.lower()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    batch_size = args.batch_size or DEFAULT_BATCH.get(model_name, 32)
    train_loader, val_loader, test_loader, scaler, num_nodes = build_dataloaders(
        dataset=dataset,
        data_root=Path(args.data_root),
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        batch_size=batch_size,
        num_workers=args.num_workers,
        limit_train_batches=args.limit_train_batches,
        limit_eval_batches=args.limit_eval_batches,
    )
    model = build_model(
        model_name=model_name,
        dataset=dataset,
        data_root=Path(args.data_root),
        aux_root=Path(args.aux_root),
        num_nodes=num_nodes,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        device=device,
        light=not args.full_model,
        use_debias=False,
    )

    risk_cfg = RiskLossConfig(
        enabled=plugin in {"risk", "risk_rex", "rex"},
        weight=args.risk_weight,
        num_envs=args.num_envs,
        warmup_epochs=args.warmup_epochs,
        pair_weight=args.pair_weight,
        extrap_weight=args.extrap_weight,
        env_mode=args.env_mode,
    )
    backbone_risk_cfg = deepcopy(risk_cfg)
    if plugin in {"risk", "risk_rex", "rex"} and args.debias_epochs > 0:
        backbone_risk_cfg.enabled = False
    risk_loss = RiskExtrapolationLoss(backbone_risk_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)

    run_dir = Path(args.output_dir) / dataset / model_name / plugin / f"seed{args.seed}_h{args.out_steps}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.init_checkpoint:
        with torch.no_grad():
            model.eval()
            warm_batch = next(iter(val_loader))
            wx, wy, wcal = to_device(warm_batch, device)
            try:
                model(wx, wcal, y_scaled=wy)
            except Exception:
                model(wx, wcal)
        state = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
    best_state = None
    best_val = math.inf
    best_epoch = -1
    bad = 0
    history: List[Dict[str, float]] = []
    start = time.time()

    if args.skip_backbone_train:
        val_stats = evaluate(model, val_loader, scaler, device, args.horizons)
        best_val = val_stats["mae_all"]
        best_epoch = 0
        best_state = deepcopy(model.state_dict())
        torch.save(best_state, run_dir / "best.pt")
        history.append({"epoch": 0, **{f"val_{k}": v for k, v in val_stats.items()}})
    else:
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                risk_loss,
                device,
                epoch,
                clip_grad=args.clip_grad,
                aux_weight=args.aux_weight,
            )
            scheduler.step()
            val_stats = evaluate(model, val_loader, scaler, device, args.horizons)
            row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}}
            history.append(row)
            print(
                f"{dataset} {model_name} {plugin} epoch {epoch+1:03d} "
                f"train={train_stats['loss']:.4f} val_mae={val_stats['mae_all']:.4f} "
                f"h3={val_stats.get('mae_h3', float('nan')):.4f} h6={val_stats.get('mae_h6', float('nan')):.4f} h12={val_stats.get('mae_h12', float('nan')):.4f}",
                flush=True,
            )
            if val_stats["mae_all"] < best_val:
                best_val = val_stats["mae_all"]
                best_epoch = epoch + 1
                best_state = deepcopy(model.state_dict())
                bad = 0
                torch.save(best_state, run_dir / "best.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    if plugin in {"risk", "risk_rex", "rex"} and args.debias_epochs > 0:
        model.attach_debias_head(num_nodes, args.in_steps, args.out_steps, device, input_mode=args.debias_input_mode)
        if not args.no_freeze_backbone:
            model.freeze_backbone()
        debias_optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.debias_lr, weight_decay=args.weight_decay)
        debias_risk = RiskExtrapolationLoss(risk_cfg)
        best_debias_state = deepcopy(model.state_dict())
        best_debias_val = evaluate(model, val_loader, scaler, device, args.horizons)["mae_all"]
        for epoch in range(args.debias_epochs):
            stats = train_debias_epoch(
                model,
                train_loader,
                debias_optimizer,
                scaler,
                debias_risk,
                device,
                epoch,
                clip_grad=args.clip_grad,
            )
            val_stats = evaluate(model, val_loader, scaler, device, args.horizons)
            print(
                f"{dataset} {model_name} {plugin} debias {epoch+1:03d} "
                f"train={stats['loss']:.4f} val_mae={val_stats['mae_all']:.4f}",
                flush=True,
            )
            if val_stats["mae_all"] < best_debias_val * 0.999:
                best_debias_val = val_stats["mae_all"]
                best_debias_state = deepcopy(model.state_dict())
        model.load_state_dict(best_debias_state)
        torch.save(best_debias_state, run_dir / "best.pt")

    val_pred, val_true, val_x = collect_predictions(model, val_loader, scaler, device)
    test_pred, test_true, test_x = collect_predictions(model, test_loader, scaler, device)
    calibrated = False
    calibration_strength = 0.0
    if plugin in {"risk", "risk_rex", "rex"} and args.calibrate:
        strengths = [float(x) for x in args.calibration_grid.split(",")]
        test_pred, calibration_strength = fit_best_calibrator(
            val_pred,
            val_true,
            val_x,
            test_pred,
            test_x,
            num_bins=args.num_envs,
            strengths=strengths,
            env_mode=args.env_mode,
        )
        calibrated = True

    test_stats = metric_dict(test_pred, test_true, args.horizons)
    result = {
        "dataset": dataset,
        "model": model_name,
        "plugin": plugin,
        "seed": args.seed,
        "in_steps": args.in_steps,
        "out_steps": args.out_steps,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "batch_size": batch_size,
        "lr": args.lr,
        "risk": asdict(risk_cfg),
        "debias_input_mode": args.debias_input_mode,
        "freeze_backbone": not args.no_freeze_backbone,
        "calibrated": calibrated,
        "calibration_strength": calibration_strength,
        "runtime_sec": time.time() - start,
        **test_stats,
    }

    with (run_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys() if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    with (run_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)

    append_result(Path(args.output_dir) / "results.csv", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def append_result(path: Path, result: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEMS-B three-variable input, flow-only forecasting trainer.")
    parser.add_argument("--dataset", default="PEMS03-B", choices=list(DATASETS.keys()))
    parser.add_argument("--model", default="STGCN", choices=MODEL_NAMES + [m.upper() for m in MODEL_NAMES])
    parser.add_argument("--plugin", default="baseline", choices=["baseline", "risk", "risk_rex", "rex"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--aux_root", default="data/aux")
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--in_steps", type=int, default=12)
    parser.add_argument("--out_steps", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--aux_weight", type=float, default=1.0)
    parser.add_argument("--risk_weight", type=float, default=0.005)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--pair_weight", type=float, default=0.05)
    parser.add_argument("--extrap_weight", type=float, default=0.05)
    parser.add_argument("--env_mode", default="speed_occ", choices=["speed_occ", "speed", "occ", "flow", "random"])
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration_strength", type=float, default=0.5)
    parser.add_argument("--calibration_grid", default="0,0.05,0.1,0.25,0.5")
    parser.add_argument("--full_model", action="store_true")
    parser.add_argument("--debias_epochs", type=int, default=8)
    parser.add_argument("--debias_lr", type=float, default=0.003)
    parser.add_argument("--debias_input_mode", default="full", choices=["full", "no_covariates", "no_node", "speed_only", "occ_only"])
    parser.add_argument("--no_freeze_backbone", action="store_true")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--skip_backbone_train", action="store_true")
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_eval_batches", type=int, default=0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1
        args.patience = 1
        args.limit_train_batches = args.limit_train_batches or 2
        args.limit_eval_batches = args.limit_eval_batches or 1
        args.num_workers = 0
    return args


def main() -> None:
    run_single(parse_args())


if __name__ == "__main__":
    main()
