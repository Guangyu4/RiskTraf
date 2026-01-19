#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import logging
from tqdm import tqdm
from dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, cheb_k=3):
        super().__init__()
        self.cheb_k = cheb_k
        self.W = nn.Linear(in_dim * cheb_k, out_dim)
    
    def forward(self, x, supports):
        B, T, N, C = x.shape
        out = [x]
        for support in supports:
            x_g = x
            for _ in range(self.cheb_k - 1):
                x_g = torch.einsum('btnc,nm->btmc', x_g, support)
                out.append(x_g)
        out = torch.cat(out[:self.cheb_k], dim=-1)
        return self.W(out)


class DilatedTemporalBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(in_dim, out_dim, (kernel_size, 1), padding=(padding, 0), dilation=(dilation, 1))
        self.gate = nn.Conv2d(in_dim, out_dim, (kernel_size, 1), padding=(padding, 0), dilation=(dilation, 1))
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        B, T, N, C = x.shape
        x_in = x.permute(0, 3, 1, 2)
        h = self.conv(x_in)[..., :T, :] * torch.sigmoid(self.gate(x_in)[..., :T, :])
        h = self.dropout(h)
        out = h + self.residual(x_in)
        return out.permute(0, 2, 3, 1)


class STBlock(nn.Module):
    def __init__(self, hidden_dim, cheb_k=3, dilation_factor=1, dropout=0.2):
        super().__init__()
        self.temporal = DilatedTemporalBlock(hidden_dim, hidden_dim, dilation=dilation_factor, dropout=dropout)
        self.spatial = GraphConv(hidden_dim, hidden_dim, cheb_k)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, supports):
        h = self.temporal(x)
        h = self.spatial(h, supports)
        return self.norm(h + x)


class LightSTGCN(nn.Module):
    def __init__(self, num_nodes, in_steps, out_steps, hidden_dim=32, num_blocks=4, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        
        self.input_proj = nn.Linear(1, hidden_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(STBlock(hidden_dim, dilation_factor=2**i, dropout=dropout))
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * in_steps, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_steps)
        )
    
    def forward(self, x):
        B, T, N, _ = x.shape
        
        g1 = F.softmax(F.relu(self.node_emb1 @ self.node_emb2.T), dim=-1)
        g2 = F.softmax(F.relu(self.node_emb2 @ self.node_emb1.T), dim=-1)
        supports = [g1, g2]
        
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, supports)
        
        h = h.permute(0, 2, 3, 1).reshape(B, N, -1)
        out = self.output_proj(h)
        return out.permute(0, 2, 1).unsqueeze(-1)


def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    return (torch.abs(preds - labels) * mask).sum() / mask.sum()


def masked_rmse(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    return torch.sqrt(((preds - labels) ** 2 * mask).sum() / mask.sum())


def masked_mape(preds, labels, null_val=0.0):
    mask = (labels != null_val) & (labels > 1.0)
    mask = mask.float()
    mask /= torch.mean(mask) + 1e-8
    loss = torch.abs((preds - labels) / (labels + 1e-8)) * mask
    loss[torch.isnan(loss)] = 0
    loss[torch.isinf(loss)] = 0
    return torch.mean(loss)


# ============ 消融实验: REx (主实验) ============
def compute_rex_loss(model, x_flow, x_full, y, num_envs=4):
    B = x_flow.shape[0]
    speed = x_full[..., 1]
    occ = x_full[..., 2]
    speed_mean = speed.mean(dim=(1, 2))
    occ_mean = occ.mean(dim=(1, 2))
    speed_norm = (speed_mean - speed_mean.mean()) / (speed_mean.std() + 1e-6)
    occ_norm = (occ_mean - occ_mean.mean()) / (occ_mean.std() + 1e-6)
    env_score = speed_norm - occ_norm
    sorted_idx = torch.argsort(env_score)
    env_size = B // num_envs
    
    output = model(x_flow)
    losses = []
    for i in range(num_envs):
        idx = sorted_idx[i * env_size:(i + 1) * env_size] if i < num_envs - 1 else sorted_idx[i * env_size:]
        if len(idx) > 0:
            losses.append(masked_mae(output[idx], y[idx]))
    
    if len(losses) < 2:
        return masked_mae(output, y), torch.tensor(0.0, device=x_flow.device)
    
    loss_stack = torch.stack(losses)
    mean_loss = loss_stack.mean()
    var_loss = ((loss_stack - mean_loss) ** 2).mean()
    return mean_loss, var_loss


# ============ 消融实验1: IRM ============
def compute_irm_loss(model, x_flow, x_full, y, num_envs=4):
    B = x_flow.shape[0]
    speed = x_full[..., 1]
    occ = x_full[..., 2]
    speed_mean = speed.mean(dim=(1, 2))
    occ_mean = occ.mean(dim=(1, 2))
    speed_norm = (speed_mean - speed_mean.mean()) / (speed_mean.std() + 1e-6)
    occ_norm = (occ_mean - occ_mean.mean()) / (occ_mean.std() + 1e-6)
    env_score = speed_norm - occ_norm
    sorted_idx = torch.argsort(env_score)
    env_size = B // num_envs
    
    output = model(x_flow)
    
    # IRM: 计算每个环境的梯度惩罚
    irm_penalties = []
    total_loss = torch.tensor(0.0, device=x_flow.device)
    for i in range(num_envs):
        idx = sorted_idx[i * env_size:(i + 1) * env_size] if i < num_envs - 1 else sorted_idx[i * env_size:]
        if len(idx) > 0:
            env_loss = masked_mae(output[idx], y[idx])
            total_loss = total_loss + env_loss
            
            # IRM penalty: ||∇w (1·loss)||^2
            scale = torch.ones(1, requires_grad=True, device=x_flow.device)
            grad = torch.autograd.grad(env_loss * scale, scale, create_graph=True)[0]
            irm_penalties.append(grad ** 2)
    
    mean_loss = total_loss / num_envs
    irm_penalty = torch.stack(irm_penalties).mean() if irm_penalties else torch.tensor(0.0, device=x_flow.device)
    return mean_loss, irm_penalty


# ============ 消融实验2: 无 REx ============
def compute_no_rex_loss(model, x_flow, y):
    output = model(x_flow)
    return masked_mae(output, y), torch.tensor(0.0, device=x_flow.device)


# ============ 消融实验3: 随机划分环境 ============
def compute_random_rex_loss(model, x_flow, y, num_envs=4):
    B = x_flow.shape[0]
    
    # 随机划分环境（不使用 speed/occ）
    random_idx = torch.randperm(B, device=x_flow.device)
    env_size = B // num_envs
    
    output = model(x_flow)
    losses = []
    for i in range(num_envs):
        idx = random_idx[i * env_size:(i + 1) * env_size] if i < num_envs - 1 else random_idx[i * env_size:]
        if len(idx) > 0:
            losses.append(masked_mae(output[idx], y[idx]))
    
    if len(losses) < 2:
        return masked_mae(output, y), torch.tensor(0.0, device=x_flow.device)
    
    loss_stack = torch.stack(losses)
    mean_loss = loss_stack.mean()
    var_loss = ((loss_stack - mean_loss) ** 2).mean()
    return mean_loss, var_loss


def train_epoch(model, train_loader, optimizer, device, epoch, args):
    model.train()
    total_loss, total_pred, total_reg = 0, 0, 0
    
    if epoch < args.warmup_epochs:
        reg_w = 0.0
    else:
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        reg_w = args.rex_weight * min(1.0, progress * 2)
    
    pbar = tqdm(train_loader, desc='Training')
    for x, y in pbar:
        x, y = x.float().to(device), y.float().to(device)
        x_flow = x[..., :1]
        y_flow = y[..., :1]
        
        optimizer.zero_grad()
        
        if args.ablation == 'irm':
            pred_loss, reg_loss = compute_irm_loss(model, x_flow, x, y_flow, args.num_envs)
        elif args.ablation == 'no_rex':
            pred_loss, reg_loss = compute_no_rex_loss(model, x_flow, y_flow)
        elif args.ablation == 'random_rex':
            pred_loss, reg_loss = compute_random_rex_loss(model, x_flow, y_flow, args.num_envs)
        else:  # rex (主实验)
            pred_loss, reg_loss = compute_rex_loss(model, x_flow, x, y_flow, args.num_envs)
        
        loss = pred_loss + reg_w * reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_pred += pred_loss.item()
        total_reg += reg_loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'w': f'{reg_w:.3f}'})
    
    n = len(train_loader)
    return total_loss/n, total_pred/n, total_reg/n, reg_w


def evaluate(model, loader, device, mean, std):
    model.eval()
    preds_list, labels_list = [], []
    mean_d = mean[0:1].to(device)
    std_d = std[0:1].to(device)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)[..., :1]
            y = y.float().to(device)[..., :1]
            out = model(x)
            out = out * std_d + mean_d
            y = y * std_d + mean_d
            preds_list.append(out.cpu())
            labels_list.append(y.cpu())
    
    preds = torch.cat(preds_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    mae = masked_mae(preds, labels).item()
    rmse = masked_rmse(preds, labels).item()
    mape = masked_mape(preds, labels).item()
    return mae, rmse, mape, preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS03-B')
    parser.add_argument('--ablation', type=str, default='rex', choices=['rex', 'irm', 'no_rex', 'random_rex'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--rex_weight', type=float, default=0.1)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_ablation')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    data_path = f'./{args.dataset}.npz'
    train_loader, val_loader, test_loader, mean, std, num_nodes = get_dataloaders(
        data_path, args.batch_size, in_steps=12, out_steps=12, num_workers=4
    )
    
    logger.info(f"Dataset: {args.dataset}, Ablation: {args.ablation}")
    logger.info(f"Nodes: {num_nodes}, Hidden: {args.hidden_dim}")
    
    model = LightSTGCN(
        num_nodes=num_nodes, in_steps=12, out_steps=12,
        hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    best_mae = float('inf')
    patience_cnt = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        loss, pred_loss, reg_loss, reg_w = train_epoch(model, train_loader, optimizer, device, epoch, args)
        val_mae, val_rmse, val_mape, _, _ = evaluate(model, val_loader, device, mean, std)
        
        scheduler.step(val_mae)
        
        logger.info(f"Loss: {loss:.4f} (Pred: {pred_loss:.4f}, Reg: {reg_loss:.4f}, W: {reg_w:.3f})")
        logger.info(f"Val: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, MAPE={val_mape:.4f}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_{args.dataset}_{args.ablation}.pt'))
            logger.info(f"Saved! Best MAE: {val_mae:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info("Early stopping")
                break
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'best_{args.dataset}_{args.ablation}.pt')))
    test_mae, test_rmse, test_mape, preds, labels = evaluate(model, test_loader, device, mean, std)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {args.dataset} - {args.ablation}")
    logger.info(f"{'='*50}")
    
    for h in [3, 6, 12]:
        h_mae = masked_mae(preds[:, h-1:h], labels[:, h-1:h]).item()
        h_rmse = masked_rmse(preds[:, h-1:h], labels[:, h-1:h]).item()
        h_mape = masked_mape(preds[:, h-1:h], labels[:, h-1:h]).item()
        logger.info(f"Horizon {h:2d}: MAE={h_mae:.4f}, RMSE={h_rmse:.4f}, MAPE={h_mape:.4f}")
    
    logger.info(f"Overall: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.4f}")


if __name__ == '__main__':
    main()
