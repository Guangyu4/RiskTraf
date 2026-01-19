import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import logging
import sys
from datetime import datetime
from tqdm import tqdm

sys.path.append('/home/bd2/DB/Torch-MTS/models')

from STGCN import STGCN
from GraphWaveNet import GWNET
from MegaCRN import MegaCRN
from dataset import get_dataloaders


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


def save_adj_pickle(adj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(adj, f)


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


def masked_rmse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask & (labels != 0)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / (labels + 1e-10))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def train_epoch(model, train_loader, optimizer, criterion, device, model_name, batches_seen=0):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc='Training')):
        x = x.float().to(device)
        y = y.float().to(device)
        x = x[..., :1]
        y = y[..., :1]
        optimizer.zero_grad()
        if model_name == 'MegaCRN':
            y_cov = torch.zeros(x.shape[0], 12, x.shape[2], 1, device=device)
            output, _, _, _, _ = model(x, y_cov, y, batches_seen + batch_idx)
        else:
            output = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader), batches_seen + len(train_loader)


def evaluate(model, val_loader, mean, std, device, model_name, horizons=[3, 6, 12]):
    model.eval()
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc='Evaluating'):
            x = x.float().to(device)
            y = y.float().to(device)
            x = x[..., :1]
            y = y[..., :1]
            if model_name == 'MegaCRN':
                y_cov = torch.zeros(x.shape[0], 12, x.shape[2], 1, device=device)
                output, _, _, _, _ = model(x, y_cov)
            else:
                output = model(x)
            preds_list.append(output.cpu())
            labels_list.append(y.cpu())
    preds = torch.cat(preds_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    preds = preds * std[0:1] + mean[0:1]
    labels = labels * std[0:1] + mean[0:1]
    results = {}
    for horizon in horizons:
        if horizon <= preds.shape[1]:
            pred_h = preds[:, horizon-1:horizon, :, :]
            label_h = labels[:, horizon-1:horizon, :, :]
            mae = masked_mae(pred_h, label_h).item()
            rmse = masked_rmse(pred_h, label_h).item()
            mape = masked_mape(pred_h, label_h).item()
            results[horizon] = {'mae': mae, 'rmse': rmse, 'mape': mape}
    overall_mae = masked_mae(preds, labels).item()
    overall_rmse = masked_rmse(preds, labels).item()
    overall_mape = masked_mape(preds, labels).item()
    results['overall'] = {'mae': overall_mae, 'rmse': overall_rmse, 'mape': overall_mape}
    return results


def get_model(model_name, num_nodes, adj_path, device, in_steps=12, out_steps=12):
    if model_name == 'STGCN':
        model = STGCN(
            n_vertex=num_nodes,
            adj_path=adj_path,
            Kt=3,
            Ks=2,
            blocks=[[1], [8, 4, 8], [8, 4, 8], [16, 16], [out_steps]],
            T=in_steps,
            act_func="glu",
            graph_conv_type="cheb_graph_conv",
            bias=True,
            droprate=0.8
        )
    elif model_name == 'GWNet':
        model = GWNET(
            device=device,
            num_nodes=num_nodes,
            adj_path=adj_path,
            adj_type="doubletransition",
            dropout=0.6,
            gcn_bool=True,
            addaptadj=True,
            in_dim=1,
            out_dim=out_steps,
            residual_channels=8,
            dilation_channels=8,
            skip_channels=64,
            end_channels=128,
            kernel_size=2,
            blocks=4,
            layers=2
        )
    elif model_name == 'MegaCRN':
        model = MegaCRN(
            num_nodes=num_nodes,
            input_dim=1,
            output_dim=1,
            horizon=out_steps,
            rnn_units=8,
            num_layers=1,
            cheb_k=2,
            ycov_dim=1,
            mem_num=3,
            mem_dim=8,
            tf_decay_steps=300,
            use_teacher_forcing=True
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['STGCN', 'GWNet', 'MegaCRN'])
    parser.add_argument('--dataset', type=str, required=True, choices=['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B'])
    parser.add_argument('--in_steps', type=int, default=12)
    parser.add_argument('--out_steps', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_mae10')
    parser.add_argument('--corr_threshold', type=float, default=0.3)
    args = parser.parse_args()
    
    log_dir = f'./logs_mae10/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    data_path = f'./{args.dataset}.npz'
    adj_dir = './adj_files'
    os.makedirs(adj_dir, exist_ok=True)
    adj_path = os.path.join(adj_dir, f'{args.dataset}_pearson.pkl')
    
    if not os.path.exists(adj_path):
        logging.info(f"Building Pearson correlation adjacency matrix for {args.dataset}...")
        adj = build_pearson_adj(data_path, args.corr_threshold)
        save_adj_pickle(adj, adj_path)
        logging.info(f"Adjacency matrix saved to {adj_path}")
    
    train_loader, val_loader, test_loader, mean, std, num_nodes = get_dataloaders(
        data_path, args.batch_size, args.in_steps, args.out_steps, args.num_workers
    )
    
    logging.info(f'Dataset: {args.dataset}, Model: {args.model}')
    logging.info(f'Num nodes: {num_nodes}')
    logging.info(f'Train samples: {len(train_loader.dataset)}')
    
    model = get_model(args.model, num_nodes, adj_path, args.device, args.in_steps, args.out_steps)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total trainable parameters: {total_params}')
    
    criterion = masked_mae
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_mae = float('inf')
    batches_seen = 0
    
    for epoch in range(args.epochs):
        logging.info(f'\nEpoch {epoch+1}/{args.epochs}')
        train_loss, batches_seen = train_epoch(model, train_loader, optimizer, criterion, args.device, args.model, batches_seen)
        logging.info(f'Train Loss: {train_loss:.4f}')
        
        val_results = evaluate(model, val_loader, mean, std, args.device, args.model)
        logging.info(f'Val Overall - MAE: {val_results["overall"]["mae"]:.4f}')
        
        val_mae = val_results['overall']['mae']
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.dataset}_{args.model}_best.pth'))
            logging.info(f'Model saved! Best Val MAE: {best_val_mae:.4f}')
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'{args.dataset}_{args.model}_best.pth')))
    test_results = evaluate(model, test_loader, mean, std, args.device, args.model)
    logging.info(f'\n{"="*60}')
    logging.info(f'Test Results: {args.dataset} - {args.model}')
    logging.info(f'{"="*60}')
    logging.info(f'Overall - MAE: {test_results["overall"]["mae"]:.4f}, RMSE: {test_results["overall"]["rmse"]:.4f}, MAPE: {test_results["overall"]["mape"]:.4f}')
    for h in [3, 6, 12]:
        if h in test_results:
            logging.info(f'Horizon {h:2d} - MAE: {test_results[h]["mae"]:.4f}, RMSE: {test_results[h]["rmse"]:.4f}, MAPE: {test_results[h]["mape"]:.4f}')
    logging.info(f'{"="*60}')
    
    result_file = os.path.join(args.save_dir, 'results.txt')
    with open(result_file, 'a') as f:
        f.write(f'{args.dataset},{args.model}')
        for h in [3, 6, 12]:
            r = test_results[h]
            f.write(f',{r["mae"]:.4f},{r["rmse"]:.4f},{r["mape"]:.4f}')
        r = test_results['overall']
        f.write(f',{r["mae"]:.4f},{r["rmse"]:.4f},{r["mape"]:.4f}\n')


if __name__ == '__main__':
    main()
