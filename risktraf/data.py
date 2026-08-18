from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


DATASETS: Dict[str, int] = {
    "PEMS03-B": 207,
    "PEMS04-B": 300,
    "PEMS07-B": 160,
    "PEMS08-B": 240,
}


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_flow(self, data: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean[..., :1], device=data.device, dtype=data.dtype)
        std = torch.as_tensor(self.std[..., :1], device=data.device, dtype=data.dtype)
        return data * std + mean


class PEMSBDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        starts: np.ndarray,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
    ) -> None:
        self.data = data.astype(np.float32, copy=False)
        self.starts = starts.astype(np.int64, copy=False)
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        mid = start + self.in_steps
        end = mid + self.out_steps
        x = self.data[start:mid]
        y_flow = self.data[mid:end, :, :1]
        time_idx = np.arange(start, end, dtype=np.int64)
        tod = (time_idx % self.steps_per_day).astype(np.float32) / self.steps_per_day
        dow = ((time_idx // self.steps_per_day) % 7).astype(np.float32)
        cal = np.stack([tod, dow], axis=-1)
        return torch.from_numpy(x), torch.from_numpy(y_flow), torch.from_numpy(cal)


def _read_pemsb_npz(path: Path) -> np.ndarray:
    raw = np.load(path)["data"].astype(np.float32)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3-D data in {path}, got {raw.shape}")
    # Stored as (nodes, time, features). Models use (time, nodes, features).
    if raw.shape[-1] == 3 and raw.shape[0] in DATASETS.values():
        raw = raw.transpose(1, 0, 2)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw


def split_starts(
    num_timesteps: int,
    in_steps: int,
    out_steps: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = num_timesteps - in_steps - out_steps + 1
    if total <= 0:
        raise ValueError("Time series is shorter than in_steps + out_steps.")
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return np.arange(train_end), np.arange(train_end, val_end), np.arange(val_end, total)


def build_dataloaders(
    dataset: str,
    data_root: Path,
    in_steps: int,
    out_steps: int,
    batch_size: int,
    num_workers: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    limit_train_batches: Optional[int] = None,
    limit_eval_batches: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureScaler, int]:
    path = data_root / f"{dataset}.npz"
    data = _read_pemsb_npz(path)
    train_idx, val_idx, test_idx = split_starts(
        data.shape[0], in_steps, out_steps, train_ratio, val_ratio
    )

    train_until = int(train_idx[-1] + in_steps) if len(train_idx) else int(data.shape[0] * train_ratio)
    train_slice = data[:train_until]
    mean = train_slice.mean(axis=(0, 1), keepdims=True)
    std = train_slice.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    scaler = FeatureScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))
    data = scaler.transform(data).astype(np.float32)

    train_set = PEMSBDataset(data, train_idx, in_steps, out_steps)
    val_set = PEMSBDataset(data, val_idx, in_steps, out_steps)
    test_set = PEMSBDataset(data, test_idx, in_steps, out_steps)

    if limit_train_batches:
        train_set = Subset(train_set, range(min(len(train_set), limit_train_batches * batch_size)))
    if limit_eval_batches:
        val_set = Subset(val_set, range(min(len(val_set), limit_eval_batches * batch_size)))
        test_set = Subset(test_set, range(min(len(test_set), limit_eval_batches * batch_size)))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, scaler, data.shape[1]


def full_scaled_flow_for_gts(dataset: str, data_root: Path, train_ratio: float = 0.6) -> np.ndarray:
    data = _read_pemsb_npz(data_root / f"{dataset}.npz")
    num_train = int(data.shape[0] * train_ratio)
    flow = data[:num_train, :, 0]
    mean = flow.mean()
    std = flow.std() if flow.std() > 1e-6 else 1.0
    return ((flow - mean) / std).astype(np.float32)


def ensure_auxiliary_files(dataset: str, data_root: Path, aux_root: Path, embed_dim: int = 64) -> Tuple[Path, Path]:
    """Create lightweight files required by GMAN/GTS/STDN style models."""
    aux_root.mkdir(parents=True, exist_ok=True)
    adj_src = data_root / "adj_files" / f"{dataset}_adj.pkl"
    adj_dst = aux_root / f"{dataset}_adj.pkl"
    if not adj_dst.exists():
        adj_dst.write_bytes(adj_src.read_bytes())

    se_path = aux_root / f"SE_{dataset}.txt"
    rewrite_se = True
    if se_path.exists():
        try:
            first = se_path.read_text().splitlines()[0].split()
            rewrite_se = len(first) != 2 or int(first[1]) != embed_dim
        except (IndexError, ValueError):
            rewrite_se = True
    if rewrite_se:
        data = _read_pemsb_npz(data_root / f"{dataset}.npz")
        flow = data[..., 0]
        corr = np.corrcoef(flow.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        u, s, _ = np.linalg.svd(corr, full_matrices=False)
        dim = min(embed_dim, u.shape[1])
        emb = u[:, :dim] * np.sqrt(s[:dim])[None, :]
        if dim < embed_dim:
            emb = np.pad(emb, ((0, 0), (0, embed_dim - dim)))
        with se_path.open("w") as f:
            f.write(f"{emb.shape[0]} {embed_dim}\n")
            for i, row in enumerate(emb.astype(np.float32)):
                f.write(str(i) + " " + " ".join(f"{v:.8f}" for v in row) + "\n")
    return adj_dst, se_path


def iter_default_grid() -> Iterable[Tuple[str, int]]:
    for dataset in DATASETS:
        yield dataset, DATASETS[dataset]
