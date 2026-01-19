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


def compute_rex_penalty(model, x, y, num_envs=3, env_split='magnitude'):
    batch_size = x.shape[0]
    if batch_size < num_envs * 2:
        return torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)
    
    output = model(x)
    
    if env_split == 'magnitude':
        flow_mean = y[..., 0].mean(dim=(1, 2))
        sorted_idx = torch.argsort(flow_mean)
    elif env_split == 'variance':
        flow_var = y[..., 0].var(dim=1).mean(dim=-1)
        sorted_idx = torch.argsort(flow_var)
    elif env_split == 'temporal':
        speed_change = torch.abs(x[:, 1:, :, 1] - x[:, :-1, :, 1]).mean(dim=(1, 2))
        sorted_idx = torch.argsort(speed_change)
    else:
        sorted_idx = torch.randperm(batch_size, device=x.device)
    
    env_size = batch_size // num_envs
    losses = []
    
    for i in range(num_envs):
        start_idx = i * env_size
        end_idx = (i + 1) * env_size if i < num_envs - 1 else batch_size
        env_indices = sorted_idx[start_idx:end_idx]
        
        output_env = output[env_indices]
        y_env = y[env_indices]
        
        loss = masked_mae(output_env, y_env[..., :1])
        losses.append(loss)
    
    loss_stack = torch.stack(losses)
    mean_loss = loss_stack.mean()
    var_loss = loss_stack.var()
    
    return mean_loss, var_loss


def compute_vrex_penalty(model, x, y, num_envs=3, env_split='magnitude'):
    batch_size = x.shape[0]
    if batch_size < num_envs * 2:
        output = model(x)
        return masked_mae(output, y[..., :1]), torch.tensor(0.0, device=x.device)
    
    output = model(x)
    
    if env_split == 'magnitude':
        flow_mean = y[..., 0].mean(dim=(1, 2))
        sorted_idx = torch.argsort(flow_mean)
    elif env_split == 'variance':
        flow_var = y[..., 0].var(dim=1).mean(dim=-1)
        sorted_idx = torch.argsort(flow_var)
    elif env_split == 'combined':
        flow_mean = y[..., 0].mean(dim=(1, 2))
        flow_var = y[..., 0].var(dim=1).mean(dim=-1)
        flow_mean_norm = (flow_mean - flow_mean.mean()) / (flow_mean.std() + 1e-6)
        flow_var_norm = (flow_var - flow_var.mean()) / (flow_var.std() + 1e-6)
        combined = flow_mean_norm + flow_var_norm
        sorted_idx = torch.argsort(combined)
    else:
        sorted_idx = torch.randperm(batch_size, device=x.device)
    
    env_size = batch_size // num_envs
    losses = []
    
    for i in range(num_envs):
        start_idx = i * env_size
        end_idx = (i + 1) * env_size if i < num_envs - 1 else batch_size
        env_indices = sorted_idx[start_idx:end_idx]
        
        output_env = output[env_indices]
        y_env = y[env_indices]
        
        loss = masked_mae(output_env, y_env[..., :1])
        losses.append(loss)
    
    loss_stack = torch.stack(losses)
    mean_loss = loss_stack.mean()
    var_loss = ((loss_stack - mean_loss) ** 2).mean()
    
    return mean_loss, var_loss


def train_epoch(model, train_loader, optimizer, device, rex_weight, num_envs, env_split, epoch, warmup_epochs):
    model.train()
    total_loss = 0
    total_erm_loss = 0
    total_rex_loss = 0
    
    if epoch < warmup_epochs:
        current_weight = rex_weight * (epoch + 1) / warmup_epochs
    else:
        current_weight = rex_weight
    
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc='Training')):
        x = x.float().to(device)
        y = y.float().to(device)
        
        optimizer.zero_grad()
        
        erm_loss, var_loss = compute_vrex_penalty(model, x, y, num_envs=num_envs, env_split=env_split)
        
        loss = erm_loss + current_weight * var_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        total_loss += loss.item()
        total_erm_loss += erm_loss.item()
        total_rex_loss += var_loss.item()
    
    return (total_loss / len(train_loader), 
            total_erm_loss / len(train_loader),
            total_rex_loss / len(train_loader),
            current_weight)


def evaluate(model, val_loader, mean, std, device, horizons=[3, 6, 12]):
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc='Evaluating'):
            x = x.float().to(device)
            y = y.float().to(device)
            
            output = model(x)
            
            preds_list.append(output.cpu())
            labels_list.append(y[..., :1].cpu())
    
    preds = torch.cat(preds_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    preds = preds * std[..., 0:1] + mean[..., 0:1]
    labels = labels * std[..., 0:1] + mean[..., 0:1]
    
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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_rex')
    
    parser.add_argument('--rex_weight', type=float, default=1.0)
    parser.add_argument('--num_envs', type=int, default=3)
    parser.add_argument('--env_split', type=str, default='magnitude', choices=['magnitude', 'variance', 'combined', 'random'])
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--input_embedding_dim', type=int, default=24)
    parser.add_argument('--tod_embedding_dim', type=int, default=0)
    parser.add_argument('--dow_embedding_dim', type=int, default=0)
    parser.add_argument('--adaptive_embedding_dim', type=int, default=80)
    parser.add_argument('--feed_forward_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    
    log_dir = f'./logs_rex/{args.dataset}'
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
    logging.info(f'REx weight: {args.rex_weight}')
    logging.info(f'Num envs: {args.num_envs}')
    logging.info(f'Env split: {args.env_split}')
    logging.info(f'Warmup epochs: {args.warmup_epochs}')
    logging.info(f'Patience: {args.patience}')
    
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
    
    if args.eval_only:
        if args.model_path is None:
            args.model_path = os.path.join(args.save_dir, f'{args.dataset}_best.pth')
        
        if not os.path.exists(args.model_path):
            logging.error(f'Model file not found: {args.model_path}')
            return
        
        logging.info(f'Loading model from: {args.model_path}')
        model.load_state_dict(torch.load(args.model_path))
        
        test_results = evaluate(model, test_loader, mean, std, args.device)
        logging.info(f'\n{"="*70}')
        logging.info(f'Test Results on {args.dataset}')
        logging.info(f'{"="*70}')
        logging.info(f'Overall - MAE: {test_results["overall"]["mae"]:.4f}, RMSE: {test_results["overall"]["rmse"]:.4f}, MAPE: {test_results["overall"]["mape"]:.4f}')
        for h in [3, 6, 12]:
            if h in test_results:
                logging.info(f'Horizon {h:2d} - MAE: {test_results[h]["mae"]:.4f}, RMSE: {test_results[h]["rmse"]:.4f}, MAPE: {test_results[h]["mape"]:.4f}')
        return
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logging.info(f'\nEpoch {epoch+1}/{args.epochs}')
        
        train_loss, erm_loss, rex_loss, current_weight = train_epoch(
            model, train_loader, optimizer, args.device, 
            args.rex_weight, args.num_envs, args.env_split, epoch, args.warmup_epochs
        )
        logging.info(f'Train Loss: {train_loss:.4f} (ERM: {erm_loss:.4f}, REx: {rex_loss:.6f}, Weight: {current_weight:.4f})')
        
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
            logging.info(f'No improvement. Patience: {patience_counter}/{args.patience}')
            if patience_counter >= args.patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'{args.dataset}_best.pth')))
    test_results = evaluate(model, test_loader, mean, std, args.device)
    logging.info(f'\n{"="*70}')
    logging.info(f'Test Results on {args.dataset} (V-REx)')
    logging.info(f'REx Weight: {args.rex_weight} | Num Envs: {args.num_envs} | Env Split: {args.env_split}')
    logging.info(f'{"="*70}')
    logging.info(f'Overall - MAE: {test_results["overall"]["mae"]:.4f}, RMSE: {test_results["overall"]["rmse"]:.4f}, MAPE: {test_results["overall"]["mape"]:.4f}')
    for h in [3, 6, 12]:
        if h in test_results:
            logging.info(f'Horizon {h:2d} - MAE: {test_results[h]["mae"]:.4f}, RMSE: {test_results[h]["rmse"]:.4f}, MAPE: {test_results[h]["mape"]:.4f}')
    logging.info(f'{"="*70}')


if __name__ == '__main__':
    main()
