import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import logging
from datetime import datetime
from tqdm import tqdm

from STAEformer import STAEformer
from dataset import get_dataloaders


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


def masked_mse(preds, labels, null_val=np.nan):
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
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val))


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


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc='Training')):
        x = x.float().to(device)
        y = y.float().to(device)
        x = x[..., :1]
        y = y[..., :1]
        
        if batch_idx == 0:
            print(f"Batch 0 - x: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
            print(f"Batch 0 - y: min={y.min().item():.4f}, max={y.max().item():.4f}, mean={y.mean().item():.4f}")
            print(f"Batch 0 - x has NaN: {torch.isnan(x).any().item()}, y has NaN: {torch.isnan(y).any().item()}")
        
        optimizer.zero_grad()
        output = model(x)
        
        if batch_idx == 0:
            print(f"Batch 0 - output: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
            print(f"Batch 0 - output has NaN: {torch.isnan(output).any().item()}")
        
        loss = criterion(output, y)
        
        if batch_idx == 0:
            print(f"Batch 0 - loss: {loss.item():.6f}")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, mean, std, device, horizons=[3, 6, 12]):
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc='Evaluating'):
            x = x.float().to(device)
            y = y.float().to(device)
            x = x[..., :1]
            y = y[..., :1]
            
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS03-B', choices=['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B'])
    parser.add_argument('--in_steps', type=int, default=12)
    parser.add_argument('--out_steps', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--input_embedding_dim', type=int, default=24)
    parser.add_argument('--tod_embedding_dim', type=int, default=0)
    parser.add_argument('--dow_embedding_dim', type=int, default=0)
    parser.add_argument('--adaptive_embedding_dim', type=int, default=80)
    parser.add_argument('--feed_forward_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    
    log_dir = f'./logs/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    data_path = f'./{args.dataset}.npz'
    train_loader, val_loader, test_loader, mean, std, num_nodes = get_dataloaders(
        data_path, args.batch_size, args.in_steps, args.out_steps, args.num_workers
    )
    
    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'Num nodes: {num_nodes}')
    logging.info(f'Train samples: {len(train_loader.dataset)}')
    logging.info(f'Val samples: {len(val_loader.dataset)}')
    logging.info(f'Test samples: {len(test_loader.dataset)}')
    
    model = STAEformer(
        num_nodes=num_nodes,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        input_embedding_dim=args.input_embedding_dim,
        tod_embedding_dim=args.tod_embedding_dim,
        dow_embedding_dim=args.dow_embedding_dim,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        feed_forward_dim=args.feed_forward_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    criterion = masked_mae
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_mae = float('inf')
    patience_counter = 0
    early_stop_patience = 3
    
    for epoch in range(args.epochs):
        logging.info(f'\nEpoch {epoch+1}/{args.epochs}')
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        logging.info(f'Train Loss: {train_loss:.4f}')
        
        val_results = evaluate(model, val_loader, mean, std, args.device)
        logging.info(f'Val Overall - MAE: {val_results["overall"]["mae"]:.4f}, RMSE: {val_results["overall"]["rmse"]:.4f}, MAPE: {val_results["overall"]["mape"]:.4f}')
        for h in [3, 6, 12]:
            if h in val_results:
                logging.info(f'  Horizon {h:2d} - MAE: {val_results[h]["mae"]:.4f}, RMSE: {val_results[h]["rmse"]:.4f}, MAPE: {val_results[h]["mape"]:.4f}')
        
        val_mae = val_results['overall']['mae']
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.dataset}_best.pth'))
            logging.info(f'Model saved! Best Val MAE: {best_val_mae:.4f}')
        else:
            patience_counter += 1
            logging.info(f'No improvement. Patience: {patience_counter}/{early_stop_patience}')
            if patience_counter >= early_stop_patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'{args.dataset}_best.pth')))
    test_results = evaluate(model, test_loader, mean, std, args.device)
    logging.info(f'\n{"="*60}')
    logging.info(f'Test Results on {args.dataset}')
    logging.info(f'{"="*60}')
    logging.info(f'Overall - MAE: {test_results["overall"]["mae"]:.4f}, RMSE: {test_results["overall"]["rmse"]:.4f}, MAPE: {test_results["overall"]["mape"]:.4f}')
    for h in [3, 6, 12]:
        if h in test_results:
            logging.info(f'Horizon {h:2d} - MAE: {test_results[h]["mae"]:.4f}, RMSE: {test_results[h]["rmse"]:.4f}, MAPE: {test_results[h]["mape"]:.4f}')
    logging.info(f'{"="*60}')


if __name__ == '__main__':
    main()

