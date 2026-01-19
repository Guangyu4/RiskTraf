#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import logging
import sys
import pickle
from datetime import datetime
from tqdm import tqdm

sys.path.append('/home/bd2/DB')
sys.path.append('/home/bd2/DB/Dish-TS')

from STSSDL import STSSDL
from DishTS import DishTS
from steve_model import STEVE


class PEMSBDatasetOOD:
    def __init__(self, data, indices, in_steps=12, out_steps=12, TDAY=288):
        self.data = data
        self.indices = indices
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.TDAY = TDAY
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.data[t:t+self.in_steps]
        y = self.data[t+self.in_steps:t+self.in_steps+self.out_steps]
        
        x_flow = x[..., 0:1]
        y_flow = y[..., 0:1]
        x_aux = x[..., 1:3]
        y_aux = y[..., 1:3]
        
        x_tod = np.arange(t, t+self.in_steps) % self.TDAY / self.TDAY
        y_tod = np.arange(t+self.in_steps, t+self.in_steps+self.out_steps) % self.TDAY / self.TDAY
        x_cov = np.tile(x_tod[:, np.newaxis, np.newaxis], (1, x.shape[0] // self.in_steps * x.shape[1] if len(x.shape) > 2 else x.shape[0], 1))
        y_cov = np.tile(y_tod[:, np.newaxis, np.newaxis], (1, y.shape[0] // self.out_steps * y.shape[1] if len(y.shape) > 2 else y.shape[0], 1))
        
        num_nodes = x.shape[0] if len(x.shape) == 2 else x.shape[1]
        x_cov = np.tile(x_tod[:, np.newaxis, np.newaxis], (1, num_nodes, 1))
        y_cov = np.tile(y_tod[:, np.newaxis, np.newaxis], (1, num_nodes, 1))
        
        return {
            'x_flow': x_flow.astype(np.float32),
            'y_flow': y_flow.astype(np.float32),
            'x_aux': x_aux.astype(np.float32),
            'y_aux': y_aux.astype(np.float32),
            'x_cov': x_cov.astype(np.float32),
            'y_cov': y_cov.astype(np.float32),
        }


def load_data_ood(file_path, in_steps=12, out_steps=12, train_ratio=0.6, val_ratio=0.2, TDAY=288):
    data_dict = np.load(file_path)
    data = data_dict['data']
    
    num_nodes, num_timesteps, num_features = data.shape
    data = data.transpose(1, 0, 2)
    
    data = np.where(np.isnan(data), 0, data)
    data = np.where(np.isinf(data), 0, data)
    
    train_data = data[:int(num_timesteps * train_ratio)]
    mean = train_data.mean(axis=(0, 1), keepdims=True)
    std = train_data.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1, std)
    
    data = (data - mean) / std
    data = np.where(np.isnan(data), 0, data)
    
    mean = torch.FloatTensor(mean)
    std = torch.FloatTensor(std)
    
    total_steps = num_timesteps - in_steps - out_steps + 1
    train_steps = int(total_steps * train_ratio)
    val_steps = int(total_steps * val_ratio)
    
    train_indices = np.arange(train_steps)
    val_indices = np.arange(train_steps, train_steps + val_steps)
    test_indices = np.arange(train_steps + val_steps, total_steps)
    
    train_dataset = PEMSBDatasetOOD(data, train_indices, in_steps, out_steps, TDAY)
    val_dataset = PEMSBDatasetOOD(data, val_indices, in_steps, out_steps, TDAY)
    test_dataset = PEMSBDatasetOOD(data, test_indices, in_steps, out_steps, TDAY)
    
    return train_dataset, val_dataset, test_dataset, mean, std, num_nodes


def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        result[key] = torch.FloatTensor(np.stack([b[key] for b in batch]))
    return result


def get_dataloaders_ood(file_path, batch_size=64, in_steps=12, out_steps=12, num_workers=4, TDAY=288):
    train_dataset, val_dataset, test_dataset, mean, std, num_nodes = load_data_ood(
        file_path, in_steps, out_steps, TDAY=TDAY
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, mean, std, num_nodes


def build_pearson_adj(data_path, threshold=0.3):
    data_dict = np.load(data_path)
    data = data_dict['data']
    num_nodes = data.shape[0]
    flow = data[:, :, 0]
    corr = np.corrcoef(flow)
    corr = np.nan_to_num(corr, nan=0.0)
    adj = np.where(corr > threshold, corr, 0.0)
    np.fill_diagonal(adj, 0)
    return adj.astype(np.float32)


def get_adj_for_model(data_path, dataset_name, model_name):
    adj_dir = './adj_files'
    os.makedirs(adj_dir, exist_ok=True)
    adj_path = os.path.join(adj_dir, f'{dataset_name}_pearson.pkl')
    
    if not os.path.exists(adj_path):
        logging.info(f"Building Pearson correlation adjacency matrix for {dataset_name}...")
        adj = build_pearson_adj(data_path)
        with open(adj_path, 'wb') as f:
            pickle.dump(adj, f)
        logging.info(f"Adjacency matrix saved to {adj_path}")
    
    with open(adj_path, 'rb') as f:
        adj = pickle.load(f)
    
    adj = torch.FloatTensor(adj)
    row_sum = adj.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    adj_norm = adj / row_sum
    
    return [adj_norm]


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_metrics(preds, labels, null_val=0.0):
    mask = (labels != null_val)
    mask = mask.float()
    
    mae = torch.abs(preds - labels)
    mae = (mae * mask).sum() / mask.sum()
    
    mse = (preds - labels) ** 2
    mse = (mse * mask).sum() / mask.sum()
    rmse = torch.sqrt(mse)
    
    mape = torch.abs((preds - labels) / (labels + 1e-8))
    mape = (mape * mask).sum() / mask.sum()
    
    return mae.item(), rmse.item(), mape.item()


class STEVEArgs:
    def __init__(self, num_nodes, device, input_dim=3, d_model=64, input_length=12, 
                 dropout=0.1, batch_size=64, kw=0.5, mi_w=2, bank_gamma=0.9,
                 ablation='none', yita=0.5, lr_mode='add'):
        self.num_nodes = num_nodes
        self.device = device
        self.d_input = input_dim
        self.d_model = d_model
        self.d_output = 1
        self.input_length = input_length
        self.dropout = dropout
        self.batch_size = batch_size
        self.kw = kw
        self.mi_w = mi_w
        self.bank_gamma = bank_gamma
        self.ablation = ablation
        self.yita = yita
        self.lr_mode = lr_mode


class GRUBackbone(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, horizon, num_layers=2):
        super(GRUBackbone, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.encoder = nn.GRU(input_dim * num_nodes, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.GRU(output_dim * num_nodes, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim * num_nodes)
        
    def forward(self, x):
        B, T, N, C = x.shape
        x = x.reshape(B, T, N * C)
        _, h = self.encoder(x)
        
        outputs = []
        dec_input = torch.zeros(B, 1, N, device=x.device)
        dec_input = dec_input.reshape(B, 1, N)
        
        for _ in range(self.horizon):
            out, h = self.decoder(dec_input, h)
            pred = self.proj(out)
            outputs.append(pred)
            dec_input = pred
        
        output = torch.cat(outputs, dim=1)
        output = output.reshape(B, self.horizon, N, 1)
        return output


class DishTSModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, horizon, seq_len):
        super(DishTSModel, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        class DishArgs:
            def __init__(self):
                self.dish_init = 'standard'
                self.n_series = num_nodes
                self.seq_len = seq_len
        
        self.dish = DishTS(DishArgs())
        self.backbone = GRUBackbone(num_nodes, input_dim, hidden_dim, output_dim, horizon)
        
    def forward(self, x):
        B, T, N, C = x.shape
        x_flow = x[..., 0].permute(0, 1, 2)
        x_norm, _ = self.dish(x_flow, mode='forward')
        x_norm = x_norm.unsqueeze(-1)
        
        x_input = torch.cat([x_norm, x[..., 1:]], dim=-1)
        output = self.backbone(x_input)
        
        output_for_dish = output.squeeze(-1)
        output_denorm = self.dish(output_for_dish, mode='inverse')
        output = output_denorm.unsqueeze(-1)
        
        return output


def train_epoch_stssdl(model, dataloader, optimizer, device, mean, std, batches_seen):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        x_flow = batch['x_flow'].to(device)
        y_flow = batch['y_flow'].to(device)
        x_aux = batch['x_aux'].to(device)
        x_cov = batch['x_cov'].to(device)
        y_cov = batch['y_cov'].to(device)
        
        x_combined = torch.cat([x_flow, x_aux], dim=-1)
        x_his = x_combined.clone()
        
        optimizer.zero_grad()
        output, query, pos, neg, mask, latent_dis, prototype_dis = model(
            x_combined, x_cov, x_his, y_cov, labels=y_flow[..., 0:1], batches_seen=batches_seen
        )
        
        pred = output
        
        mean_flow = mean[..., 0:1].to(device)
        std_flow = std[..., 0:1].to(device)
        pred_denorm = pred * std_flow + mean_flow
        y_denorm = y_flow * std_flow + mean_flow
        
        loss = masked_mae(pred_denorm, y_denorm, null_val=0.0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        batches_seen += 1
    
    return total_loss / len(dataloader), batches_seen


def evaluate_stssdl(model, dataloader, device, mean, std):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x_flow = batch['x_flow'].to(device)
            y_flow = batch['y_flow'].to(device)
            x_aux = batch['x_aux'].to(device)
            x_cov = batch['x_cov'].to(device)
            y_cov = batch['y_cov'].to(device)
            
            x_combined = torch.cat([x_flow, x_aux], dim=-1)
            x_his = x_combined.clone()
            
            output, query, pos, neg, mask, latent_dis, prototype_dis = model(
                x_combined, x_cov, x_his, y_cov, labels=None, batches_seen=None
            )
            
            pred = output[..., 0:1]
            
            mean_flow = mean[..., 0:1].to(device)
            std_flow = std[..., 0:1].to(device)
            pred_denorm = pred * std_flow + mean_flow
            y_denorm = y_flow * std_flow + mean_flow
            
            all_preds.append(pred_denorm.cpu())
            all_labels.append(y_denorm.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    horizons = [3, 6, 12]
    results = {}
    
    for h in horizons:
        h_preds = all_preds[:, h-1:h, :, :]
        h_labels = all_labels[:, h-1:h, :, :]
        mae, rmse, mape = compute_metrics(h_preds, h_labels, null_val=0.0)
        results[h] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    overall_mae, overall_rmse, overall_mape = compute_metrics(all_preds, all_labels, null_val=0.0)
    results['overall'] = {'MAE': overall_mae, 'RMSE': overall_rmse, 'MAPE': overall_mape}
    
    return results


def train_epoch_steve(model, dataloader, optimizer, device, mean, std, adj):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        x_flow = batch['x_flow'].to(device)
        y_flow = batch['y_flow'].to(device)
        x_aux = batch['x_aux'].to(device)
        
        x_combined = torch.cat([x_flow, x_aux], dim=-1)
        
        optimizer.zero_grad()
        H, Z = model(x_combined, adj)
        pred = model.predict_test(Z, H)
        
        mean_flow = mean[..., 0:1].to(device)
        std_flow = std[..., 0:1].to(device)
        pred_denorm = pred * std_flow + mean_flow
        y_denorm = y_flow * std_flow + mean_flow
        
        loss = masked_mae(pred_denorm, y_denorm, null_val=0.0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_steve(model, dataloader, device, mean, std, adj):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x_flow = batch['x_flow'].to(device)
            y_flow = batch['y_flow'].to(device)
            x_aux = batch['x_aux'].to(device)
            
            x_combined = torch.cat([x_flow, x_aux], dim=-1)
            
            H, Z = model(x_combined, adj)
            pred = model.predict_test(Z, H)
            
            mean_flow = mean[..., 0:1].to(device)
            std_flow = std[..., 0:1].to(device)
            pred_denorm = pred * std_flow + mean_flow
            y_denorm = y_flow[:, 0:1, :, :] * std_flow + mean_flow
            
            all_preds.append(pred_denorm.cpu())
            all_labels.append(y_denorm.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    overall_mae, overall_rmse, overall_mape = compute_metrics(all_preds, all_labels, null_val=0.0)
    
    results = {
        3: {'MAE': overall_mae, 'RMSE': overall_rmse, 'MAPE': overall_mape},
        6: {'MAE': overall_mae, 'RMSE': overall_rmse, 'MAPE': overall_mape},
        12: {'MAE': overall_mae, 'RMSE': overall_rmse, 'MAPE': overall_mape},
        'overall': {'MAE': overall_mae, 'RMSE': overall_rmse, 'MAPE': overall_mape}
    }
    
    return results


def train_epoch_dishts(model, dataloader, optimizer, device, mean, std):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        x_flow = batch['x_flow'].to(device)
        y_flow = batch['y_flow'].to(device)
        x_aux = batch['x_aux'].to(device)
        
        x_combined = torch.cat([x_flow, x_aux], dim=-1)
        
        optimizer.zero_grad()
        output = model(x_combined)
        
        mean_flow = mean[..., 0:1].to(device)
        std_flow = std[..., 0:1].to(device)
        pred_denorm = output * std_flow + mean_flow
        y_denorm = y_flow * std_flow + mean_flow
        
        loss = masked_mae(pred_denorm, y_denorm, null_val=0.0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_dishts(model, dataloader, device, mean, std):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x_flow = batch['x_flow'].to(device)
            y_flow = batch['y_flow'].to(device)
            x_aux = batch['x_aux'].to(device)
            
            x_combined = torch.cat([x_flow, x_aux], dim=-1)
            output = model(x_combined)
            
            mean_flow = mean[..., 0:1].to(device)
            std_flow = std[..., 0:1].to(device)
            pred_denorm = output * std_flow + mean_flow
            y_denorm = y_flow * std_flow + mean_flow
            
            all_preds.append(pred_denorm.cpu())
            all_labels.append(y_denorm.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    horizons = [3, 6, 12]
    results = {}
    
    for h in horizons:
        h_preds = all_preds[:, h-1:h, :, :]
        h_labels = all_labels[:, h-1:h, :, :]
        mae, rmse, mape = compute_metrics(h_preds, h_labels, null_val=0.0)
        results[h] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    overall_mae, overall_rmse, overall_mape = compute_metrics(all_preds, all_labels, null_val=0.0)
    results['overall'] = {'MAE': overall_mae, 'RMSE': overall_rmse, 'MAPE': overall_mape}
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ST-SSDL', choices=['ST-SSDL', 'STEVE', 'Dish-TS'])
    parser.add_argument('--dataset', type=str, default='PEMS03-B')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--in_steps', type=int, default=12)
    parser.add_argument('--out_steps', type=int, default=12)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--rnn_units', type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs('./checkpoints_ood', exist_ok=True)
    os.makedirs('./logs_ood', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'./logs_ood/{args.dataset}_{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    data_path = f'./{args.dataset}.npz'
    train_loader, val_loader, test_loader, mean, std, num_nodes = get_dataloaders_ood(
        data_path, batch_size=args.batch_size, in_steps=args.in_steps, out_steps=args.out_steps
    )
    
    logging.info(f"Dataset: {args.dataset}, Model: {args.model}")
    logging.info(f"Num nodes: {num_nodes}")
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Val samples: {len(val_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")
    
    adj_mx = get_adj_for_model(data_path, args.dataset, args.model)
    adj_mx = [a.to(args.device) for a in adj_mx]
    
    scaler = None
    adj_single = adj_mx[0] if adj_mx else None
    
    if args.model == 'ST-SSDL':
        model = STSSDL(
            num_nodes=num_nodes,
            input_dim=3,
            output_dim=1,
            horizon=args.out_steps,
            rnn_units=args.rnn_units,
            rnn_layers=1,
            cheb_k=3,
            ycov_dim=1,
            prototype_num=20,
            prototype_dim=64,
            tod_embed_dim=10,
            adj_mx=adj_mx,
            cl_decay_steps=2000,
            TDAY=288,
            use_curriculum_learning=True,
            use_STE=False,
            device=args.device,
            adaptive_embedding_dim=48,
            node_embedding_dim=20,
            input_embedding_dim=64
        ).to(args.device)
    elif args.model == 'STEVE':
        model = STEVE(
            num_nodes=num_nodes,
            input_dim=3,
            embed_size=64,
            input_length=args.in_steps,
            output_dim=1,
            dropout=0.1,
            device=args.device
        ).to(args.device)
        
    elif args.model == 'Dish-TS':
        model = DishTSModel(
            num_nodes=num_nodes,
            input_dim=3,
            hidden_dim=64,
            output_dim=1,
            horizon=args.out_steps,
            seq_len=args.in_steps
        ).to(args.device)
    else:
        logging.error(f"Model {args.model} not implemented")
        return
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_mae = float('inf')
    patience_counter = 0
    batches_seen = 0
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\nEpoch {epoch}/{args.epochs}")
        
        if args.model == 'ST-SSDL':
            train_loss, batches_seen = train_epoch_stssdl(model, train_loader, optimizer, args.device, mean, std, batches_seen)
        elif args.model == 'STEVE':
            train_loss = train_epoch_steve(model, train_loader, optimizer, args.device, mean, std, adj_single)
        elif args.model == 'Dish-TS':
            train_loss = train_epoch_dishts(model, train_loader, optimizer, args.device, mean, std)
        
        logging.info(f"Train Loss: {train_loss:.4f}")
        
        if args.model == 'ST-SSDL':
            val_results = evaluate_stssdl(model, val_loader, args.device, mean, std)
        elif args.model == 'STEVE':
            val_results = evaluate_steve(model, val_loader, args.device, mean, std, adj_single)
        elif args.model == 'Dish-TS':
            val_results = evaluate_dishts(model, val_loader, args.device, mean, std)
        
        val_mae = val_results['overall']['MAE']
        val_rmse = val_results['overall']['RMSE']
        val_mape = val_results['overall']['MAPE']
        
        logging.info(f"Val Overall - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.4f}")
        for h in [3, 6, 12]:
            logging.info(f"  Horizon {h:2d} - MAE: {val_results[h]['MAE']:.4f}, RMSE: {val_results[h]['RMSE']:.4f}, MAPE: {val_results[h]['MAPE']:.4f}")
        
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), f'./checkpoints_ood/{args.dataset}_{args.model}_best.pth')
            logging.info(f"Model saved! Best Val MAE: {best_val_mae:.4f}")
        else:
            patience_counter += 1
            logging.info(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")
        
        if patience_counter >= args.early_stop_patience:
            logging.info("Early stopping triggered!")
            break
    
    model.load_state_dict(torch.load(f'./checkpoints_ood/{args.dataset}_{args.model}_best.pth'))
    
    if args.model == 'ST-SSDL':
        test_results = evaluate_stssdl(model, test_loader, args.device, mean, std)
    elif args.model == 'STEVE':
        test_results = evaluate_steve(model, test_loader, args.device, mean, std, adj_single)
    elif args.model == 'Dish-TS':
        test_results = evaluate_dishts(model, test_loader, args.device, mean, std)
    
    logging.info("\n" + "=" * 60)
    logging.info(f"Test Results: {args.dataset} - {args.model}")
    logging.info("=" * 60)
    logging.info(f"Overall - MAE: {test_results['overall']['MAE']:.4f}, RMSE: {test_results['overall']['RMSE']:.4f}, MAPE: {test_results['overall']['MAPE']:.4f}")
    for h in [3, 6, 12]:
        logging.info(f"Horizon {h:2d} - MAE: {test_results[h]['MAE']:.4f}, RMSE: {test_results[h]['RMSE']:.4f}, MAPE: {test_results[h]['MAPE']:.4f}")
    logging.info("=" * 60)
    
    results_file = './checkpoints_ood/results.txt'
    with open(results_file, 'a') as f:
        line = f"{args.dataset},{args.model},"
        line += f"{test_results[3]['MAE']:.4f},{test_results[3]['RMSE']:.4f},{test_results[3]['MAPE']:.4f},"
        line += f"{test_results[6]['MAE']:.4f},{test_results[6]['RMSE']:.4f},{test_results[6]['MAPE']:.4f},"
        line += f"{test_results[12]['MAE']:.4f},{test_results[12]['RMSE']:.4f},{test_results[12]['MAPE']:.4f},"
        line += f"{test_results['overall']['MAE']:.4f},{test_results['overall']['RMSE']:.4f},{test_results['overall']['MAPE']:.4f}\n"
        f.write(line)
    
    logging.info(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()

