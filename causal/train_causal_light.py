import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import os
import logging
from datetime import datetime
from tqdm import tqdm

from dataset import get_dataloaders


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class LightSTBlock(nn.Module):
    def __init__(self, num_nodes, in_steps, hidden_dim, num_heads=4):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1),
            TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
        )
        self.spatial_attn = SpatialAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
    def forward(self, x):
        B, T, N, C = x.shape
        x_t = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        x_t = self.temporal_conv(x_t)
        x_t = x_t.reshape(B, N, C, T).permute(0, 3, 1, 2)
        x = x + x_t
        x_s = x.reshape(B * T, N, C)
        x_s = self.norm1(x_s)
        x_s = x_s + self.spatial_attn(x_s)
        x_s = x_s + self.ffn(self.norm2(x_s))
        x = x_s.reshape(B, T, N, C)
        return x


class CausalRExModel(nn.Module):
    def __init__(self, num_nodes, in_steps, out_steps, input_dim=3, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.temporal_emb = nn.Parameter(torch.randn(1, in_steps, 1, hidden_dim) * 0.02)
        self.spatial_emb = nn.Parameter(torch.randn(1, 1, num_nodes, hidden_dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            LightSTBlock(num_nodes, in_steps, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * in_steps, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_steps),
        )
        
    def forward(self, x, return_features=False):
        B, T, N, C = x.shape
        x = self.input_proj(x)
        x = x + self.temporal_emb + self.spatial_emb
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        features = x.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.output_proj(features)
        out = out.permute(0, 2, 1).unsqueeze(-1)
        
        if return_features:
            return out, features
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


def contrastive_loss(features, temperature=0.1):
    B, N, C = features.shape
    features = F.normalize(features, dim=-1)
    sim = torch.bmm(features, features.transpose(1, 2)) / temperature
    labels = torch.arange(N, device=features.device).unsqueeze(0).expand(B, -1)
    loss = F.cross_entropy(sim.reshape(B * N, N), labels.reshape(-1))
    return loss


def causal_intervention(x, intervention_type='mixup', alpha=0.3):
    B, T, N, C = x.shape
    
    if intervention_type == 'mixup':
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(B, device=x.device)
        x_mixed = lam * x + (1 - lam) * x[idx]
        return x_mixed, lam, idx
    
    elif intervention_type == 'cutmix':
        lam = np.random.beta(alpha, alpha)
        cut_len = int(T * (1 - lam))
        start = np.random.randint(0, T - cut_len + 1)
        idx = torch.randperm(B, device=x.device)
        x_cut = x.clone()
        x_cut[:, start:start+cut_len] = x[idx, start:start+cut_len]
        lam = 1 - cut_len / T
        return x_cut, lam, idx
    
    elif intervention_type == 'dropout_nodes':
        mask = torch.rand(B, 1, N, 1, device=x.device) > alpha
        x_drop = x * mask
        return x_drop, 1.0, None
    
    elif intervention_type == 'noise':
        noise = torch.randn_like(x) * alpha * x.std()
        return x + noise, 1.0, None
    
    return x, 1.0, None


def compute_vrex_loss(model, x, y, num_envs=3):
    B = x.shape[0]
    if B < num_envs * 2:
        output = model(x)
        return masked_mae(output, y[..., :1]), torch.tensor(0.0, device=x.device)
    
    output = model(x)
    
    flow_mean = y[..., 0].mean(dim=(1, 2))
    sorted_idx = torch.argsort(flow_mean)
    
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
    var_loss = ((loss_stack - mean_loss) ** 2).mean()
    
    return mean_loss, var_loss


def train_epoch(model, train_loader, optimizer, device, epoch, args):
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_vrex_loss = 0
    total_contrast_loss = 0
    
    warmup_epochs = args.warmup_epochs
    if epoch < warmup_epochs:
        vrex_weight = args.vrex_weight * (epoch + 1) / warmup_epochs
        contrast_weight = args.contrast_weight * (epoch + 1) / warmup_epochs
    else:
        vrex_weight = args.vrex_weight
        contrast_weight = args.contrast_weight
    
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc='Training')):
        x = x.float().to(device)
        y = y.float().to(device)
        
        optimizer.zero_grad()
        
        pred_loss, vrex_loss = compute_vrex_loss(model, x, y, num_envs=args.num_envs)
        
        if args.use_contrast:
            output, features = model(x, return_features=True)
            con_loss = contrastive_loss(features, temperature=0.1)
        else:
            con_loss = torch.tensor(0.0, device=device)
        
        if args.use_intervention and np.random.rand() < 0.5:
            intervention_type = np.random.choice(['mixup', 'cutmix', 'noise'])
            x_int, lam, idx = causal_intervention(x, intervention_type, alpha=0.3)
            output_int = model(x_int)
            
            if idx is not None:
                y_int = lam * y + (1 - lam) * y[idx]
            else:
                y_int = y
            
            int_loss = masked_mae(output_int, y_int[..., :1])
            pred_loss = 0.5 * pred_loss + 0.5 * int_loss
        
        loss = pred_loss + vrex_weight * vrex_loss + contrast_weight * con_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_vrex_loss += vrex_loss.item()
        total_contrast_loss += con_loss.item()
    
    n = len(train_loader)
    return total_loss/n, total_pred_loss/n, total_vrex_loss/n, total_contrast_loss/n, vrex_weight


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
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_causal_light')
    
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--vrex_weight', type=float, default=0.5)
    parser.add_argument('--contrast_weight', type=float, default=0.1)
    parser.add_argument('--num_envs', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--use_contrast', action='store_true', default=True)
    parser.add_argument('--use_intervention', action='store_true', default=True)
    
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    
    args = parser.parse_args()
    
    log_dir = f'./logs_causal_light/{args.dataset}'
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
    logging.info(f'Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}')
    logging.info(f'V-REx weight: {args.vrex_weight}, Contrast weight: {args.contrast_weight}')
    logging.info(f'Use contrast: {args.use_contrast}, Use intervention: {args.use_intervention}')
    
    model = CausalRExModel(
        num_nodes=num_nodes,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        input_dim=3,
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
        logging.info(f'Test Results: MAE={test_results["overall"]["mae"]:.4f}, RMSE={test_results["overall"]["rmse"]:.4f}')
        for h in [3, 6, 12]:
            if h in test_results:
                logging.info(f'Horizon {h}: MAE={test_results[h]["mae"]:.4f}, RMSE={test_results[h]["rmse"]:.4f}')
        return
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logging.info(f'\nEpoch {epoch+1}/{args.epochs}')
        
        loss, pred_loss, vrex_loss, con_loss, vrex_w = train_epoch(
            model, train_loader, optimizer, args.device, epoch, args
        )
        logging.info(f'Loss: {loss:.4f} (Pred: {pred_loss:.4f}, VREx: {vrex_loss:.6f}, Con: {con_loss:.4f}, W: {vrex_w:.3f})')
        
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
    logging.info(f'Test Results on {args.dataset} (CausalREx)')
    logging.info(f'{"="*60}')
    logging.info(f'Overall: MAE={test_results["overall"]["mae"]:.4f}, RMSE={test_results["overall"]["rmse"]:.4f}, MAPE={test_results["overall"]["mape"]:.4f}')
    for h in [3, 6, 12]:
        if h in test_results:
            logging.info(f'Horizon {h:2d}: MAE={test_results[h]["mae"]:.4f}, RMSE={test_results[h]["rmse"]:.4f}, MAPE={test_results[h]["mape"]:.4f}')
    logging.info(f'{"="*60}')


if __name__ == '__main__':
    main()
