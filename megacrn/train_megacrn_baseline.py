#!/usr/bin/env python
"""
Pure MegaCRN baseline without REx for comparison.
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

sys.path.append('/home/bd2/DB/Torch-MTS/models')
from MegaCRN import MegaCRN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask) + 1e-8
    loss = torch.abs(preds - labels) * mask
    loss[torch.isnan(loss)] = 0
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask) + 1e-8
    loss = torch.abs((preds - labels) / (labels + 1e-8)) * mask
    loss[torch.isnan(loss)] = 0
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask) + 1e-8
    loss = ((preds - labels) ** 2) * mask
    loss[torch.isnan(loss)] = 0
    return torch.sqrt(torch.mean(loss))


def sep_loss(query, pos, neg):
    positive = F.cosine_similarity(query, pos, dim=-1)
    negative = F.cosine_similarity(query, neg, dim=-1)
    return (1 - positive).mean() + negative.mean()


def train_epoch(model, train_loader, optimizer, device, batches_seen):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        x_flow = x[..., :1]
        y_flow = y[..., :1]
        y_cov = torch.zeros_like(y_flow)
        
        optimizer.zero_grad()
        output, h_att, query, pos, neg = model(x_flow, y_cov, labels=y_flow, batches_seen=batches_seen)
        
        pred_loss = masked_mae(output, y_flow)
        sep = sep_loss(query, pos, neg)
        loss = pred_loss + 0.01 * sep
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        batches_seen += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader), batches_seen


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
        pred = output * std + mean
        label = y_flow * std + mean
        preds.append(pred.cpu())
        labels.append(label.cpu())
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    mae = masked_mae(preds, labels).item()
    rmse = masked_rmse(preds, labels).item()
    mape = masked_mape(preds, labels).item()
    
    return mae, rmse, mape, preds, labels


def load_data(dataset_name, batch_size):
    data_path = f'/home/bd2/DB/PEMSB/{dataset_name}.npz'
    data = np.load(data_path)['data'].transpose(1, 0, 2)
    
    nan_mask = np.isnan(data)
    logger.info(f"NaN: {100*nan_mask.sum()/data.size:.1f}%")
    
    global_mean = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    global_std = np.nanstd(data.reshape(-1, data.shape[-1]), axis=0)
    global_std[global_std < 1e-6] = 1.0
    
    for i in range(data.shape[-1]):
        data[nan_mask[..., i], i] = global_mean[i]
    
    mean = global_mean.reshape(1, 1, -1)
    std = global_std.reshape(1, 1, -1)
    data = (data - mean) / std
    
    in_steps, out_steps = 12, 12
    x_list, y_list = [], []
    for i in range(len(data) - in_steps - out_steps + 1):
        x_list.append(data[i:i+in_steps])
        y_list.append(data[i+in_steps:i+in_steps+out_steps])
    
    x, y = np.stack(x_list), np.stack(y_list)
    
    n = len(x)
    train_size, val_size = int(n * 0.6), int(n * 0.2)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x[:train_size]), torch.FloatTensor(y[:train_size])),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x[train_size:train_size+val_size]), 
                      torch.FloatTensor(y[train_size:train_size+val_size])),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x[train_size+val_size:]), 
                      torch.FloatTensor(y[train_size+val_size:])),
        batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader, mean[0, 0, 0], std[0, 0, 0], data.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS03-B')
    parser.add_argument('--rnn_units', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_megacrn')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader, mean, std, num_nodes = load_data(args.dataset, args.batch_size)
    
    logger.info(f"Dataset: {args.dataset}, Nodes: {num_nodes}")
    
    model = MegaCRN(
        num_nodes=num_nodes, input_dim=1, output_dim=1, horizon=12,
        rnn_units=args.rnn_units, num_layers=1, cheb_k=3,
        ycov_dim=1, mem_num=20, mem_dim=64, tf_decay_steps=2000, use_teacher_forcing=True
    ).to(device)
    
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    best_mae = float('inf')
    patience_cnt = 0
    batches_seen = 0
    
    for epoch in range(1, args.epochs + 1):
        loss, batches_seen = train_epoch(model, train_loader, optimizer, device, batches_seen)
        val_mae, val_rmse, val_mape, _, _ = evaluate(model, val_loader, device, mean, std)
        scheduler.step(val_mae)
        
        logger.info(f"Epoch {epoch}: Loss={loss:.4f}, Val MAE={val_mae:.4f}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_{args.dataset}.pt'))
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info("Early stopping")
                break
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'best_{args.dataset}.pt')))
    test_mae, test_rmse, test_mape, preds, labels = evaluate(model, test_loader, device, mean, std)
    
    logger.info(f"\n=== {args.dataset} Test ===")
    for h in [3, 6, 12]:
        h_mae = masked_mae(preds[:, h-1:h], labels[:, h-1:h]).item()
        h_rmse = masked_rmse(preds[:, h-1:h], labels[:, h-1:h]).item()
        logger.info(f"Horizon {h:2d}: MAE={h_mae:.4f}, RMSE={h_rmse:.4f}")
    logger.info(f"Overall: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")


if __name__ == '__main__':
    main()
