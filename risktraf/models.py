from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from lib.losses import GTSLoss, MegaCRNLoss

from .data import ensure_auxiliary_files, full_scaled_flow_for_gts
from .local_staeformer import STAEformer


MODEL_NAMES = [
    "STGCN",
    "DCRNN",
    "AGCRN",
    "GWNET",
    "MTGNN",
    "GMAN",
    "StemGNN",
    "STNorm",
    "GTS",
    "STID",
    "STWA",
    "MegaCRN",
    "STAEformer",
    "HimNet",
    "STDN",
]


LIGHT_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "STGCN": {"droprate": 0.3},
    "DCRNN": {"rnn_units": 32, "num_rnn_layers": 1, "max_diffusion_step": 2},
    "AGCRN": {"embed_dim": 8, "rnn_units": 32, "num_layers": 1, "cheb_k": 2},
    "GWNET": {"residual_channels": 16, "dilation_channels": 16, "skip_channels": 64, "end_channels": 128, "blocks": 2, "layers": 2},
    "MTGNN": {"conv_channels": 16, "residual_channels": 16, "skip_channels": 32, "end_channels": 64, "layers": 2, "node_dim": 20, "subgraph_size": 20},
    "GMAN": {"statt_layers": 2, "att_heads": 4, "att_dims": 8},
    "STEMGNN": {"stack_cnt": 2, "multi_layer": 3, "dropout_rate": 0.3},
    "STNORM": {"channels": 16, "blocks": 2, "layers": 2},
    "GTS": {"rnn_units": 32, "num_rnn_layers": 1, "max_diffusion_step": 2, "k": 10},
    "STID": {"embed_dim": 24, "node_dim": 24, "temp_dim_tid": 0, "temp_dim_diw": 0, "num_layer": 2},
    "STWA": {"channels": 16, "memory_size": 12, "dynamic": False},
    "MEGACRN": {"rnn_units": 32, "num_layers": 1, "cheb_k": 2, "mem_num": 16, "mem_dim": 32},
    "STAEFORMER": {"input_embedding_dim": 16, "tod_embedding_dim": 0, "dow_embedding_dim": 0, "adaptive_embedding_dim": 32, "feed_forward_dim": 128, "num_heads": 4, "num_layers": 2},
    "HIMNET": {"hidden_dim": 32, "num_layers": 1, "cheb_k": 2, "tod_embedding_dim": 8, "dow_embedding_dim": 8, "node_embedding_dim": 12, "st_embedding_dim": 12},
    "STDN": {"L": 1, "K": 8, "d": 8, "order": 2, "reference": 2},
}


CLASS_MAP = {
    "STGCN": ("STGCN.py", "STGCN"),
    "DCRNN": ("DCRNN.py", "DCRNN"),
    "AGCRN": ("AGCRN.py", "AGCRN"),
    "GWNET": ("GraphWaveNet.py", "GWNET"),
    "GRAPHWAVENET": ("GraphWaveNet.py", "GWNET"),
    "MTGNN": ("MTGNN.py", "MTGNN"),
    "GMAN": ("GMAN.py", "GMAN"),
    "STEMGNN": ("StemGNN.py", "StemGNN"),
    "STNORM": ("STNorm.py", "STNorm"),
    "GTS": ("GTS.py", "GTS"),
    "STID": ("STID.py", "STID"),
    "STWA": ("STWA.py", "STWA"),
    "MEGACRN": ("MegaCRN.py", "MegaCRN"),
    "HIMNET": ("HimNet.py", "HimNet"),
    "STDN": ("STDN.py", "STDN"),
}


def load_model_class(name: str):
    name = name.upper()
    if name == "STAEFORMER":
        return STAEformer
    file_name, class_name = CLASS_MAP[name]
    root = Path(__file__).resolve().parents[1]
    path = root / "models" / file_name
    module_name = f"pemsb_{path.stem.lower()}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return getattr(module, class_name)


def _merge(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if overrides:
        out.update(overrides)
    return out


class BackboneAdapter(nn.Module):
    def __init__(
        self,
        name: str,
        backbone: nn.Module,
        out_steps: int,
        debias_head: Optional[nn.Module] = None,
        scaler: Any = None,
    ) -> None:
        super().__init__()
        self.name = name.upper()
        self.backbone = backbone
        self.out_steps = out_steps
        self.debias_head = debias_head
        self.scaler = scaler
        self.batch_seen = 0
        self.extra_loss = None
        self.last_debias = None
        self.mega_loss = MegaCRNLoss()
        self.gts_loss = GTSLoss()

    def _calendar_features(self, cal: torch.Tensor, num_nodes: int, future: bool = False) -> torch.Tensor:
        if future:
            c = cal[:, -self.out_steps :, :]
        else:
            c = cal[:, : -self.out_steps, :]
        return c[:, :, None, :].expand(-1, -1, num_nodes, -1)

    def _flow_with_calendar(self, x: torch.Tensor, cal: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[..., :1], self._calendar_features(cal, x.shape[2])], dim=-1)

    def _normalize_output(self, out: torch.Tensor) -> torch.Tensor:
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() == 4 and out.shape[1] == self.out_steps:
            return out[..., :1]
        if out.dim() == 4 and out.shape[1] == self.out_steps and out.shape[-1] != 1:
            return out[..., :1]
        if out.dim() == 4 and out.shape[1] == self.out_steps:
            return out
        if out.dim() == 4 and out.shape[1] == self.out_steps:
            return out
        if out.dim() == 4 and out.shape[1] != self.out_steps and out.shape[-1] == 1:
            return out[:, : self.out_steps]
        if out.dim() == 4 and out.shape[1] == self.out_steps:
            return out
        if out.dim() == 4 and out.shape[1] != self.out_steps:
            # GraphWaveNet/MTGNN/STNorm return (B, out_steps, N, 1) already for common configs.
            return out[:, : self.out_steps, :, :1]
        raise RuntimeError(f"Unexpected output shape from {self.name}: {tuple(out.shape)}")

    def forward(self, x: torch.Tensor, cal: torch.Tensor, y_scaled: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.extra_loss = None
        self.last_debias = None
        name = self.name
        if name in {"STGCN", "AGCRN", "GWNET", "MTGNN", "STNORM", "STID", "STWA", "STAEFORMER"}:
            model_x = x
            out = self.backbone(model_x)
            return self._apply_debias(self._normalize_output(out), x)
        if name == "STEMGNN":
            out = self.backbone(x[..., :1])
            return self._apply_debias(self._normalize_output(out), x)
        if name == "DCRNN":
            if self.training and y_scaled is not None:
                out = self.backbone(x, y_scaled, self.batch_seen)
                self.batch_seen += 1
            else:
                out = self.backbone(x)
            return self._apply_debias(self._normalize_output(out), x)
        if name == "GTS":
            if self.training and y_scaled is not None:
                out, pred_adj, prior_adj = self.backbone(x, y_scaled, self.batch_seen)
                self.batch_seen += 1
            else:
                out, pred_adj, prior_adj = self.backbone(x)
            self.extra_loss = ("gts", pred_adj, prior_adj)
            return self._apply_debias(self._normalize_output(out), x)
        if name == "MEGACRN":
            y_cov = torch.empty(x.shape[0], self.out_steps, x.shape[2], 0, device=x.device)
            labels = y_scaled if y_scaled is not None else None
            restore_training = self.backbone.training
            if labels is None:
                self.backbone.eval()
            out, h_att, query, pos, neg = self.backbone(x, y_cov, labels, self.batch_seen)
            if restore_training:
                self.backbone.train()
            if self.training:
                self.batch_seen += 1
            self.extra_loss = ("megacrn", query, pos, neg)
            return self._apply_debias(self._normalize_output(out), x)
        if name == "HIMNET":
            hist = self._flow_with_calendar(x, cal)
            y_cov = self._calendar_features(cal, x.shape[2], future=True)
            labels = y_scaled if y_scaled is not None else None
            restore_training = self.backbone.training
            if labels is None:
                self.backbone.eval()
            out = self.backbone(hist, y_cov, labels, self.batch_seen)
            if restore_training:
                self.backbone.train()
            if self.training:
                self.batch_seen += 1
            return self._apply_debias(self._normalize_output(out), x)
        if name in {"GMAN", "STDN"}:
            hist = self._flow_with_calendar(x, cal)
            future_cal = self._calendar_features(cal, x.shape[2], future=True)
            out = self.backbone(hist, future_cal)
            return self._apply_debias(self._normalize_output(out), x)
        raise NotImplementedError(name)

    def _apply_debias(self, pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.debias_head is None:
            return pred
        residual = self.debias_head(x)
        self.last_debias = residual
        return pred + residual

    def auxiliary_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if not self.extra_loss:
            return pred.new_tensor(0.0)
        if self.extra_loss[0] == "gts":
            _, pred_adj, prior_adj = self.extra_loss
            return self.gts_loss(pred, true, pred_adj, prior_adj) - self.gts_loss.masked_mae_loss(pred, true)
        if self.extra_loss[0] == "megacrn":
            _, query, pos, neg = self.extra_loss
            return self.mega_loss.l1 * self.mega_loss.separate_loss(query, pos.detach(), neg.detach()) + self.mega_loss.l2 * self.mega_loss.compact_loss(query, pos.detach())
        return pred.new_tensor(0.0)

    def debias_regularization(self) -> torch.Tensor:
        if self.last_debias is None:
            return next(self.parameters()).new_tensor(0.0)
        return self.last_debias.abs().mean()

    def attach_debias_head(
        self,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        device: torch.device,
        input_mode: str = "full",
    ) -> None:
        if self.debias_head is None:
            self.debias_head = CovariateDebiasHead(num_nodes, in_steps, out_steps, input_mode=input_mode).to(device)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]


class CovariateDebiasHead(nn.Module):
    """Flow residual head driven only by speed/occupancy covariates."""

    def __init__(
        self,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        hidden_dim: int = 32,
        node_dim: int = 8,
        dropout: float = 0.1,
        input_mode: str = "full",
    ) -> None:
        super().__init__()
        if input_mode not in {"full", "no_covariates", "no_node", "speed_only", "occ_only"}:
            raise ValueError(f"Unknown debias input mode: {input_mode}")
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_mode = input_mode
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        input_dim = in_steps * 2 + 3 + node_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_steps),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.full((1, out_steps, 1, 1), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cov = x[..., 1:3]
        batch_size, _, num_nodes, _ = cov.shape
        speed_series = cov[..., 0]
        occ_series = cov[..., 1]
        if self.input_mode == "speed_only":
            occ_series = torch.zeros_like(occ_series)
        elif self.input_mode == "occ_only":
            speed_series = torch.zeros_like(speed_series)
        elif self.input_mode == "no_covariates":
            speed_series = torch.zeros_like(speed_series)
            occ_series = torch.zeros_like(occ_series)
        cov = torch.stack([speed_series, occ_series], dim=-1)
        speed = speed_series.mean(dim=1)
        occ = occ_series.mean(dim=1)
        cov_flat = cov.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.in_steps * 2)
        node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        if self.input_mode == "no_node":
            node_emb = torch.zeros_like(node_emb)
        stats = torch.stack([speed, occ, occ - speed], dim=-1)
        features = torch.cat([cov_flat, stats, node_emb], dim=-1)
        residual = self.net(features).transpose(1, 2).unsqueeze(-1)
        return torch.tanh(self.scale) * residual


def build_model(
    model_name: str,
    dataset: str,
    data_root: Path,
    aux_root: Path,
    num_nodes: int,
    in_steps: int,
    out_steps: int,
    device: torch.device,
    train_ratio: float = 0.6,
    light: bool = True,
    use_debias: bool = False,
) -> BackboneAdapter:
    name = model_name.upper()
    adj_path, se_path = ensure_auxiliary_files(dataset, data_root, aux_root)
    cls = load_model_class(name)
    overrides = LIGHT_OVERRIDES.get(name if name != "GRAPHWAVENET" else "GWNET", {}) if light else {}

    if name == "STGCN":
        if light:
            blocks = [[3], [32, 16, 32], [32, 16, 32], [64, 64], [out_steps]]
        else:
            blocks = [[3], [64, 16, 64], [64, 16, 64], [128, 128], [out_steps]]
        kwargs = _merge({"n_vertex": num_nodes, "adj_path": str(adj_path), "blocks": blocks, "T": in_steps}, overrides)
    elif name == "DCRNN":
        kwargs = _merge(
            {
                "num_nodes": num_nodes,
                "adj_path": str(adj_path),
                "device": device,
                "input_dim": 3,
                "output_dim": 1,
                "seq_len": in_steps,
                "horizon": out_steps,
                "rnn_units": 64,
                "num_rnn_layers": 1,
                "max_diffusion_step": 2,
                "filter_type": "dual_random_walk",
                "tf_decay_steps": 2000,
                "use_teacher_forcing": True,
            },
            overrides,
        )
    elif name == "AGCRN":
        kwargs = _merge({"num_nodes": num_nodes, "out_steps": out_steps, "input_dim": 3, "output_dim": 1, "embed_dim": 10, "rnn_units": 64, "num_layers": 1, "cheb_k": 2, "default_graph": True}, overrides)
    elif name in {"GWNET", "GRAPHWAVENET"}:
        kwargs = _merge({"device": device, "num_nodes": num_nodes, "adj_path": str(adj_path), "adj_type": "doubletransition", "in_dim": 3, "out_dim": out_steps, "gcn_bool": True, "addaptadj": True}, overrides)
    elif name == "MTGNN":
        kwargs = _merge({"gcn_true": True, "buildA_true": True, "gcn_depth": 2, "num_nodes": num_nodes, "device": device, "predefined_A": None, "seq_length": in_steps, "in_dim": 3, "out_dim": out_steps}, overrides)
    elif name == "GMAN":
        kwargs = _merge({"SE_file_path": str(se_path), "device": device, "timestep_in": in_steps}, overrides)
        ensure_auxiliary_files(dataset, data_root, aux_root, embed_dim=kwargs["att_heads"] * kwargs["att_dims"])
    elif name == "STEMGNN":
        kwargs = _merge({"units": num_nodes, "time_step": in_steps, "horizon": out_steps}, overrides)
    elif name == "STNORM":
        kwargs = _merge({"num_nodes": num_nodes, "in_dim": 3, "out_dim": out_steps}, overrides)
    elif name == "GTS":
        # GTS expects ../data/<dataset>/data.npz during construction. Bypass by monkey-patching node_feats after init.
        tmp_dir = Path("../data") / dataset
        tmp_dir.mkdir(parents=True, exist_ok=True)
        raw = full_scaled_flow_for_gts(dataset, data_root, train_ratio=1.0)
        tmp_npz = tmp_dir / f"data.{os.getpid()}.npz"
        np.savez_compressed(tmp_npz, data=raw[..., None])
        os.replace(tmp_npz, tmp_dir / "data.npz")
        num_train_for_gts = round(raw.shape[0] * train_ratio)
        dim_fc = max(1, (num_train_for_gts - 18) * 16)
        kwargs = _merge({"num_nodes": num_nodes, "input_dim": 3, "output_dim": 1, "seq_len": in_steps, "horizon": out_steps, "rnn_units": 64, "num_rnn_layers": 1, "max_diffusion_step": 2, "filter_type": "dual_random_walk", "tf_decay_steps": 2000, "use_teacher_forcing": True, "l1_decay": 0, "temp": 0.5, "k": min(10, num_nodes - 1), "dim_fc": dim_fc, "dataset_name": dataset, "trainset_ratio": train_ratio}, overrides)
    elif name == "STID":
        kwargs = _merge({"num_nodes": num_nodes, "input_len": in_steps, "output_len": out_steps, "input_dim": 3, "if_time_in_day": False, "if_day_in_week": False, "temp_dim_tid": 0, "temp_dim_diw": 0}, overrides)
    elif name == "STWA":
        kwargs = _merge({"device": device, "num_nodes": num_nodes, "input_dim": 3, "output_dim": 1, "lag": in_steps, "horizon": out_steps}, overrides)
    elif name == "MEGACRN":
        kwargs = _merge({"num_nodes": num_nodes, "input_dim": 3, "output_dim": 1, "horizon": out_steps, "ycov_dim": 0, "rnn_units": 64, "num_layers": 1, "cheb_k": 2, "tf_decay_steps": 2000, "use_teacher_forcing": True}, overrides)
    elif name == "STAEFORMER":
        kwargs = _merge({"num_nodes": num_nodes, "in_steps": in_steps, "out_steps": out_steps, "input_dim": 3, "output_dim": 1, "tod_embedding_dim": 0, "dow_embedding_dim": 0}, overrides)
    elif name == "HIMNET":
        kwargs = _merge({"num_nodes": num_nodes, "input_dim": 3, "output_dim": 1, "out_steps": out_steps, "ycov_dim": 2, "use_teacher_forcing": True}, overrides)
    elif name == "STDN":
        kwargs = _merge({"device": device, "num_of_vertices": num_nodes, "adj_path": str(adj_path), "num_his": in_steps, "num_pred": out_steps, "in_channels": 1, "out_channels": 1}, overrides)
    else:
        raise NotImplementedError(model_name)

    backbone = cls(**kwargs).to(device)
    debias_head = CovariateDebiasHead(num_nodes, in_steps, out_steps) if use_debias else None
    return BackboneAdapter(name, backbone, out_steps, debias_head=debias_head).to(device)
