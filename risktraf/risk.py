from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def masked_mae(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = true.ne(null_val).float()
    if mask.mean() <= 0:
        mask = torch.ones_like(mask)
    else:
        mask = mask / (mask.mean() + 1e-8)
    loss = (pred - true).abs() * mask
    return torch.nan_to_num(loss).mean()


def masked_rmse(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = true.ne(null_val).float()
    if mask.mean() <= 0:
        mask = torch.ones_like(mask)
    else:
        mask = mask / (mask.mean() + 1e-8)
    mse = torch.nan_to_num(((pred - true) ** 2) * mask).mean()
    return torch.sqrt(mse)


def masked_mape(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = true.ne(null_val) & true.ne(0)
    mask = mask.float()
    if mask.mean() <= 0:
        mask = torch.ones_like(mask)
    else:
        mask = mask / (mask.mean() + 1e-8)
    loss = ((pred - true).abs() / true.clamp(min=1e-5).abs()) * mask
    return torch.nan_to_num(loss).mean() * 100


@dataclass
class RiskLossConfig:
    enabled: bool = False
    weight: float = 0.1
    num_envs: int = 4
    warmup_epochs: int = 5
    env_mode: str = "speed_occ"
    pair_weight: float = 0.05
    extrap_weight: float = 0.05
    quantile_eps: float = 1e-6


class RiskExtrapolationLoss:
    """REx-style environment plugin using speed and occupancy only."""

    def __init__(self, config: RiskLossConfig) -> None:
        self.config = config

    def _environment_score(self, x: torch.Tensor) -> torch.Tensor:
        speed = x[..., 1].mean(dim=(1, 2))
        occ = x[..., 2].mean(dim=(1, 2))
        speed_z = (speed - speed.mean()) / (speed.std(unbiased=False) + 1e-6)
        occ_z = (occ - occ.mean()) / (occ.std(unbiased=False) + 1e-6)
        if self.config.env_mode == "random":
            return torch.rand_like(speed_z)
        if self.config.env_mode == "flow":
            flow = x[..., 0].mean(dim=(1, 2))
            return (flow - flow.mean()) / (flow.std(unbiased=False) + 1e-6)
        if self.config.env_mode == "speed":
            return -speed_z
        if self.config.env_mode == "occ":
            return occ_z
        return occ_z - speed_z

    def _split_envs(self, score: torch.Tensor) -> List[torch.Tensor]:
        order = torch.argsort(score)
        chunks = torch.chunk(order, min(self.config.num_envs, score.numel()))
        return [chunk for chunk in chunks if chunk.numel() > 0]

    def __call__(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        x: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        base = masked_mae(pred, true)
        if not self.config.enabled or x.shape[0] < max(4, self.config.num_envs):
            return base, {"pred_loss": float(base.detach()), "rex": 0.0, "pair": 0.0, "extrap": 0.0, "weight": 0.0}

        score = self._environment_score(x)
        env_indices = self._split_envs(score)
        if len(env_indices) < 2:
            return base, {"pred_loss": float(base.detach()), "rex": 0.0, "pair": 0.0, "extrap": 0.0, "weight": 0.0}

        env_losses = torch.stack([masked_mae(pred[idx], true[idx]) for idx in env_indices])
        rex = env_losses.var(unbiased=False)
        pair = torch.stack(
            [
                F.relu(env_losses[i + 1] - env_losses[i]).pow(2)
                for i in range(len(env_losses) - 1)
            ]
        ).mean()

        low = env_losses[0]
        high = env_losses[-1]
        extrap = F.relu(high - low.detach()).pow(2)

        ramp = min(1.0, float(epoch + 1) / max(1, self.config.warmup_epochs))
        weight = self.config.weight * ramp
        penalty = rex + self.config.pair_weight * pair + self.config.extrap_weight * extrap
        total = base + weight * penalty
        return total, {
            "pred_loss": float(base.detach()),
            "rex": float(rex.detach()),
            "pair": float(pair.detach()),
            "extrap": float(extrap.detach()),
            "weight": weight,
        }
