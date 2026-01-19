#!/usr/bin/env python
"""
MegaCRN + REx: Use MegaCRN backbone with flow-only input, 
environment split based on speed/occupancy for REx regularization.
"""
import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add path for MegaCRN
sys.path.append('/home/bd2/DB/Torch-MTS/models')
from MegaCRN import MegaCRN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask) + 1e-8
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask) + 1e-8
    loss = torch.abs((preds - labels) / (labels + 1e-8))
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask) + 1e-8
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return torch.mean(loss)


def compute_rex_loss(model, x_flow, x_full, y, y_cov, num_envs=3, batches_seen=0):
    """
    Compute V-REx loss with environment split based on speed/occupancy.
    x_flow: (B, T, N, 1) - only flow for model input
    x_full: (B, T, N, 3) - full input with flow, speed, occ for env split
    y: (B, T, N, 1) - target flow
    y_cov: (B, T, N, 1) - decoder covariate
    """
    batch_size = x_flow.shape[0]
    
    # Environment split based on speed and occupancy patterns
    speed = x_full[..., 1]  # (B, T, N)
    occ = x_full[..., 2]    # (B, T, N)
    
    # Compute environment score: high speed + low occ vs low speed + high occ
    speed_mean = speed.mean(dim=(1, 2))  # (B,)
    occ_mean = occ.mean(dim=(1, 2))      # (B,)
    
    speed_norm = (speed_mean - speed_mean.mean()) / (speed_mean.std() + 1e-6)
    occ_norm = (occ_mean - occ_mean.mean()) / (occ_mean.std() + 1e-6)
    env_score = speed_norm - occ_norm  # High = free flow, Low = congested
    
    sorted_idx = torch.argsort(env_score)
    env_size = batch_size // num_envs
    
    env_indices = []
    for i in range(num_envs):
        if i < num_envs - 1:
            env_indices.append(sorted_idx[i * env_size:(i + 1) * env_size])
        else:
            env_indices.append(sorted_idx[i * env_size:])
    
    # Forward pass
    output, h_att, query, pos, neg = model(x_flow, y_cov, labels=y, batches_seen=batches_seen)
    
    # Compute loss per environment
    losses = []
    for idx in env_indices:
        if len(idx) > 0:
            loss = masked_mae(output[idx], y[idx])
            losses.append(loss)
    
    if len(losses) < 2:
        return masked_mae(output, y), torch.tensor(0.0, device=x_flow.device), h_att, query, pos, neg
    
    loss_stack = torch.stack(losses)
    mean_loss = loss_stack.mean()
    var_loss = ((loss_stack - mean_loss) ** 2).mean()
    
    return mean_loss, var_loss, h_att, query, pos, neg


def sep_loss(query, pos, neg):
    """Separation loss from MegaCRN for memory module."""
    positive = F.cosine_similarity(query, pos, dim=-1)
    negative = F.cosine_similarity(query, neg, dim=-1)
    loss = (1 - positive).mean() + negative.mean()
    return loss


def train_epoch(model, train_loader, optimizer, device, epoch, args, batches_seen):
    model.train()
    total_loss = 0
    total_pred = 0
    total_rex = 0
    total_sep = 0
    
    # Warmup for REx
    if epoch < args.warmup_epochs:
        rex_w = 0.0
    else:
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        rex_w = args.rex_weight * min(1.0, progress * 2)
    
    pbar = tqdm(train_loader, desc=f'Training')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # x: (B, T, N, 3) with flow, speed, occ
        # y: (B, T, N, 3) with flow, speed, occ
        x_flow = x[..., :1]  # Only flow for model input
        y_flow = y[..., :1]  # Only flow for target
        y_cov = torch.zeros_like(y_flow)  # Zero covariate
        
        optimizer.zero_grad()
        
        pred_loss, rex_loss, h_att, query, pos, neg = compute_rex_loss(
            model, x_flow, x, y_flow, y_cov, 
            num_envs=args.num_envs, batches_seen=batches_seen
        )
        
        # Separation loss from MegaCRN
        sep = sep_loss(query, pos, neg)
        
        loss = pred_loss + rex_w * rex_loss + args.sep_weight * sep
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_pred += pred_loss.item()
        total_rex += rex_loss.item()
        total_sep += sep.item()
        batches_seen += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pred': f'{pred_loss.item():.4f}',
            'rex': f'{rex_loss.item():.4f}',
            'rex_w': f'{rex_w:.3f}'
        })
    
    n = len(train_loader)
    return total_loss/n, total_pred/n, total_rex/n, total_sep/n, rex_w, batches_seen


@torch.no_grad()
def evaluate(model, val_loader, device, mean, std):
    model.eval()
    preds, labels = [], []
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        x_flow = x[..., :1]
        y_flow = y[..., :1]
        y_cov = torch.zeros_like(y_flow)
        
        output, _, _, _, _ = model(x_flow, y_cov)
        
        # Denormalize
        pred = output * std[0, 0, 0] + mean[0, 0, 0]
        label = y_flow * std[0, 0, 0] + mean[0, 0, 0]
        
        preds.append(pred.cpu())
        labels.append(label.cpu())
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    mae = masked_mae(preds, labels).item()
    rmse = masked_rmse(preds, labels).item()
    mape = masked_mape(preds, labels).item()
    
    return mae, rmse, mape, preds, labels


def load_data(dataset_name, batch_size):
    """Load PEMS dataset."""
    data_path = f'/home/bd2/DB/PEMSB/{dataset_name}.npz'
    
    data = np.load(data_path)['data']
    # Data shape: (N, T, F) -> transpose to (T, N, F)
    data = data.transpose(1, 0, 2)
    
    # Handle NaN - compute global mean/std per feature
    nan_mask = np.isnan(data)
    logger.info(f"Initial NaN count: {nan_mask.sum()} ({100*nan_mask.sum()/data.size:.2f}%)")
    
    # Global mean/std per feature (ignoring NaN)
    global_mean = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    global_std = np.nanstd(data.reshape(-1, data.shape[-1]), axis=0)
    global_std[global_std < 1e-6] = 1.0
    
    # Fill NaN with global mean
    for i in range(data.shape[-1]):
        data[nan_mask[..., i], i] = global_mean[i]
    
    # Normalize
    mean = global_mean.reshape(1, 1, -1)
    std = global_std.reshape(1, 1, -1)
    data = (data - mean) / std
    
    logger.info(f"After filling: NaN count = {np.isnan(data).sum()}")
    
    # Create sequences
    in_steps, out_steps = 12, 12
    x_list, y_list = [], []
    for i in range(len(data) - in_steps - out_steps + 1):
        x_list.append(data[i:i+in_steps])
        y_list.append(data[i+in_steps:i+in_steps+out_steps])
    
    x = np.stack(x_list)  # (samples, in_steps, nodes, features)
    y = np.stack(y_list)  # (samples, out_steps, nodes, features)
    
    # Train/val/test split
    n = len(x)
    train_size = int(n * 0.6)
    val_size = int(n * 0.2)
    
    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
    x_test, y_test = x[train_size+val_size:], y[train_size+val_size:]
    
    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test)),
        batch_size=batch_size, shuffle=False
    )
    
    num_nodes = data.shape[1]
    
    return train_loader, val_loader, test_loader, mean, std, num_nodes, len(x_train), len(x_val), len(x_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS03-B')
    parser.add_argument('--rnn_units', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cheb_k', type=int, default=3)
    parser.add_argument('--mem_num', type=int, default=20)
    parser.add_argument('--mem_dim', type=int, default=64)
    parser.add_argument('--rex_weight', type=float, default=0.5)
    parser.add_argument('--sep_weight', type=float, default=0.01)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_megacrn_rex')
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader, mean, std, num_nodes, n_train, n_val, n_test = load_data(
        args.dataset, args.batch_size
    )
    
    logger.info(f"Dataset: {args.dataset}, Nodes: {num_nodes}")
    logger.info(f"Train/Val/Test: {n_train}/{n_val}/{n_test}")
    logger.info(f"RNN units: {args.rnn_units}, Layers: {args.num_layers}")
    logger.info(f"Memory: num={args.mem_num}, dim={args.mem_dim}")
    logger.info(f"REx weight: {args.rex_weight}, Num envs: {args.num_envs}")
    
    # Create model
    model = MegaCRN(
        num_nodes=num_nodes,
        input_dim=1,  # Only flow
        output_dim=1,
        horizon=12,
        rnn_units=args.rnn_units,
        num_layers=args.num_layers,
        cheb_k=args.cheb_k,
        ycov_dim=1,
        mem_num=args.mem_num,
        mem_dim=args.mem_dim,
        tf_decay_steps=2000,
        use_teacher_forcing=True
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    
    mean_t = torch.FloatTensor(mean).to(args.device)
    std_t = torch.FloatTensor(std).to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_mae = float('inf')
    patience_counter = 0
    batches_seen = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        loss, pred_loss, rex_loss, sep_loss_val, rex_w, batches_seen = train_epoch(
            model, train_loader, optimizer, args.device, epoch, args, batches_seen
        )
        
        val_mae, val_rmse, val_mape, _, _ = evaluate(model, val_loader, args.device, mean_t, std_t)
        
        scheduler.step(val_mae)
        
        logger.info(f"Train: Loss={loss:.4f}, Pred={pred_loss:.4f}, REx={rex_loss:.4f}, Sep={sep_loss_val:.4f}, REx_w={rex_w:.3f}")
        logger.info(f"Val: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, MAPE={val_mape:.4f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_{args.dataset}.pt'))
            logger.info(f"New best model saved! MAE={val_mae:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Test
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'best_{args.dataset}.pt')))
    test_mae, test_rmse, test_mape, preds, labels = evaluate(model, test_loader, args.device, mean_t, std_t)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results on {args.dataset}")
    logger.info(f"{'='*50}")
    
    # Per-horizon results
    for h in [3, 6, 12]:
        h_mae = masked_mae(preds[:, h-1:h], labels[:, h-1:h]).item()
        h_rmse = masked_rmse(preds[:, h-1:h], labels[:, h-1:h]).item()
        h_mape = masked_mape(preds[:, h-1:h], labels[:, h-1:h]).item()
        logger.info(f"Horizon {h:2d}: MAE={h_mae:.4f}, RMSE={h_rmse:.4f}, MAPE={h_mape:.4f}")
    
    logger.info(f"Overall: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.4f}")


if __name__ == '__main__':
    main()
