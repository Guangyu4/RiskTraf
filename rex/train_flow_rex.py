import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import logging
from datetime import datetime
from tqdm import tqdm

from dataset import get_dataloaders


class TemporalBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        B, T, N, C = x.shape
        res = x
        x_t = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        x_t = self.act(self.conv1(x_t))
        x_t = self.act(self.conv2(x_t))
        x_t = x_t.reshape(B, N, C, T).permute(0, 3, 1, 2)
        return self.norm(res + x_t)


class SpatialBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        
    def forward(self, x):
        B, T, N, C = x.shape
        x_s = x.reshape(B * T, N, C)
        x_s = self.norm1(x_s)
        attn_out, _ = self.attn(x_s, x_s, x_s)
        x_s = x_s + attn_out
        x_s = x_s + self.ffn(self.norm2(x_s))
        return x_s.reshape(B, T, N, C)


class FlowRExModel(nn.Module):
    def __init__(self, num_nodes, in_steps, out_steps, hidden_dim=48, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        
        self.input_proj = nn.Linear(1, hidden_dim)
        self.temporal_emb = nn.Parameter(torch.randn(1, in_steps, 1, hidden_dim) * 0.02)
        self.spatial_emb = nn.Parameter(torch.randn(1, 1, num_nodes, hidden_dim) * 0.02)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TemporalBlock(hidden_dim))
            self.blocks.append(SpatialBlock(hidden_dim, num_heads, dropout))
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * in_steps, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_steps),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        flow = x[..., 0:1]
        B, T, N, _ = flow.shape
        
        h = self.input_proj(flow)
        h = h + self.temporal_emb + self.spatial_emb
        h = self.dropout(h)
        
        for block in self.blocks:
            h = block(h)
        
        h = h.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.output_proj(h)
        out = out.permute(0, 2, 1).unsqueeze(-1)
        
        return out


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


def compute_rex_loss(model, x, y, num_envs=4):
    B = x.shape[0]
    if B < num_envs * 2:
        output = model(x)
        return masked_mae(output, y[..., :1]), torch.tensor(0.0, device=x.device)
    
    output = model(x)
    
    speed = x[..., 1]
    occ = x[..., 2]
    
    speed_mean = speed.mean(dim=(1, 2))
    occ_mean = occ.mean(dim=(1, 2))
    
    speed_norm = (speed_mean - speed_mean.mean()) / (speed_mean.std() + 1e-6)
    occ_norm = (occ_mean - occ_mean.mean()) / (occ_mean.std() + 1e-6)
    
    env_score = speed_norm - occ_norm
    sorted_idx = torch.argsort(env_score)
    
    env_size = B // num_envs
    losses = []
    
    for i in range(num_envs):
        start = i * env_size
        end = (i + 1) * env_size if i < num_envs - 1 else B
        idx = sorted_idx[start:end]
        loss = masked_mae(output[idx], y[idx, ..., :1])
        losses.append(loss)
    
    loss_stack = torch.stack(losses)
    mean_loss = loss_stack.mean()
    
    max_loss = loss_stack.max()
    min_loss = loss_stack.min()
    var_loss = (max_loss - min_loss) ** 2
    
    return mean_loss, var_loss


def train_epoch(model, train_loader, optimizer, device, epoch, args):
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_rex_loss = 0
    
    if epoch < args.warmup_epochs:
        rex_weight = args.rex_weight * (epoch + 1) / args.warmup_epochs
    else:
        rex_weight = args.rex_weight
    
    for x, y in tqdm(train_loader, desc='Training'):
        x = x.float().to(device)
        y = y.float().to(device)
        
        optimizer.zero_grad()
        
        pred_loss, rex_loss = compute_rex_loss(model, x, y, num_envs=args.num_envs)
        
        loss = pred_loss + rex_weight * rex_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_rex_loss += rex_loss.item()
    
    n = len(train_loader)
    return total_loss/n, total_pred_loss/n, total_rex_loss/n, rex_weight


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
    parser.add_argument('--dataset', type=str, default='PEMS03-B')
    parser.add_argument('--in_steps', type=int, default=12)
    parser.add_argument('--out_steps', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_flow_rex')
    
    parser.add_argument('--hidden_dim', type=int, default=48)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--rex_weight', type=float, default=0.5)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=25)
    
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    
    args = parser.parse_args()
    
    log_dir = f'./logs_flow_rex/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    data_path = f'./{args.dataset}.npz'
    train_loader, val_loader, test_loader, mean, std, num_nodes = get_dataloaders(
        data_path, args.batch_size, args.in_steps, args.out_steps, args.num_workers
    )
    
    logging.info(f'Dataset: {args.dataset}, Nodes: {num_nodes}')
    logging.info(f'Train/Val/Test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}')
    logging.info(f'Hidden: {args.hidden_dim}, Layers: {args.num_layers}')
    logging.info(f'REx weight: {args.rex_weight}, Num envs: {args.num_envs}')
    
    model = FlowRExModel(
        num_nodes=num_nodes,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total parameters: {total_params:,}')
    
    if args.eval_only:
        if args.model_path is None:
            args.model_path = os.path.join(args.save_dir, f'{args.dataset}_best.pth')
        model.load_state_dict(torch.load(args.model_path))
        test_results = evaluate(model, test_loader, mean, std, args.device)
        logging.info(f'Test: MAE={test_results["overall"]["mae"]:.4f}, RMSE={test_results["overall"]["rmse"]:.4f}')
        for h in [3, 6, 12]:
            if h in test_results:
                logging.info(f'H{h}: MAE={test_results[h]["mae"]:.4f}')
        return
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logging.info(f'\nEpoch {epoch+1}/{args.epochs}')
        
        loss, pred_loss, rex_loss, rex_w = train_epoch(model, train_loader, optimizer, args.device, epoch, args)
        logging.info(f'Loss: {loss:.4f} (Pred: {pred_loss:.4f}, REx: {rex_loss:.6f}, W: {rex_w:.3f})')
        
        val_results = evaluate(model, val_loader, mean, std, args.device)
        logging.info(f'Val: MAE={val_results["overall"]["mae"]:.4f}, RMSE={val_results["overall"]["rmse"]:.4f}')
        for h in [3, 6, 12]:
            if h in val_results:
                logging.info(f'  H{h}: MAE={val_results[h]["mae"]:.4f}')
        
        scheduler.step()
        val_mae = val_results['overall']['mae']
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.dataset}_best.pth'))
            logging.info(f'Saved! Best MAE: {best_val_mae:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f'Early stop at epoch {epoch+1}')
                break
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'{args.dataset}_best.pth')))
    test_results = evaluate(model, test_loader, mean, std, args.device)
    
    logging.info(f'\n{"="*60}')
    logging.info(f'Test Results on {args.dataset} (FlowREx)')
    logging.info(f'REx Weight: {args.rex_weight} | Num Envs: {args.num_envs}')
    logging.info(f'{"="*60}')
    logging.info(f'Overall: MAE={test_results["overall"]["mae"]:.4f}, RMSE={test_results["overall"]["rmse"]:.4f}, MAPE={test_results["overall"]["mape"]:.4f}')
    for h in [3, 6, 12]:
        if h in test_results:
            logging.info(f'Horizon {h:2d}: MAE={test_results[h]["mae"]:.4f}, RMSE={test_results[h]["rmse"]:.4f}, MAPE={test_results[h]["mape"]:.4f}')
    logging.info(f'{"="*60}')


if __name__ == '__main__':
    main()
