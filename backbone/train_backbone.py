#!/usr/bin/env python
"""多Backbone时空预测模型对比实验"""
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


# ==================== GraphConv (共用) ====================
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, order=2):
        super().__init__()
        self.order = order
        self.fc = nn.Linear(in_dim * (order + 1), out_dim)
    
    def forward(self, x, adj):
        out = [x]
        h = x
        for _ in range(self.order):
            h = torch.matmul(adj, h)
            out.append(h)
        out = torch.cat(out, dim=-1)
        return self.fc(out)


# ==================== Backbone 1: LightSTGCN ====================
class TemporalConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=2, dilation=1):
        super().__init__()
        self.filter_conv = nn.Conv2d(in_dim, out_dim, (1, kernel_size), dilation=(1, dilation))
        self.gate_conv = nn.Conv2d(in_dim, out_dim, (1, kernel_size), dilation=(1, dilation))
        self.bn = nn.BatchNorm2d(out_dim)
        self.dilation = dilation
        self.kernel_size = kernel_size
    
    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        x_pad = F.pad(x, (pad, 0))
        filter_out = torch.tanh(self.filter_conv(x_pad))
        gate_out = torch.sigmoid(self.gate_conv(x_pad))
        return self.bn(filter_out * gate_out)


class STBlock(nn.Module):
    def __init__(self, dim, num_nodes, kernel_size=2, dilation=1, dropout=0.1):
        super().__init__()
        self.tcn = TemporalConv(dim, dim, kernel_size, dilation)
        self.gcn = GraphConv(dim, dim, order=2)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x, adj):
        residual = x
        x = self.tcn(x)
        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x = self.gcn(x, adj)
        x = x.reshape(B, T, N, C).permute(0, 3, 2, 1)
        x = self.dropout(x)
        if residual.shape[-1] != x.shape[-1]:
            residual = self.skip(residual[..., -x.shape[-1]:])
        return x + residual


class LightSTGCN(nn.Module):
    def __init__(self, num_nodes, in_steps=12, out_steps=12, hidden_dim=32, num_blocks=4, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 10) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, 10) * 0.1)
        self.input_proj = nn.Conv2d(1, hidden_dim, 1)
        self.blocks = nn.ModuleList()
        for d in [1, 2, 4, 8][:num_blocks]:
            self.blocks.append(STBlock(hidden_dim, num_nodes, kernel_size=2, dilation=d, dropout=dropout))
        self.end_conv1 = nn.Conv2d(hidden_dim, hidden_dim * 2, 1)
        self.end_conv2 = nn.Conv2d(hidden_dim * 2, out_steps, 1)
        
    def forward(self, x):
        B, T, N, C = x.shape
        adj = F.softmax(F.relu(torch.mm(self.node_emb1, self.node_emb2.T)), dim=-1)
        x = x.permute(0, 3, 2, 1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, adj)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x[..., -1:])
        x = x.permute(0, 2, 3, 1).squeeze(2).unsqueeze(-1).permute(0, 2, 1, 3)
        return x


# ==================== Backbone 2: Transformer ====================
class SpatialTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = self.norm(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ffn(x))
        return x


class STTransformer(nn.Module):
    def __init__(self, num_nodes, in_steps=12, out_steps=12, hidden_dim=32, num_layers=4, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        
        self.node_emb = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.1)
        self.time_emb = nn.Parameter(torch.randn(in_steps, hidden_dim) * 0.1)
        self.input_proj = nn.Linear(1, hidden_dim)
        
        self.t_layers = nn.ModuleList([SpatialTemporalAttention(hidden_dim, 4, dropout) for _ in range(num_layers)])
        self.s_layers = nn.ModuleList([SpatialTemporalAttention(hidden_dim, 4, dropout) for _ in range(num_layers)])
        
        self.out_proj = nn.Linear(in_steps * hidden_dim, out_steps)
        
    def forward(self, x):
        B, T, N, C = x.shape
        x = self.input_proj(x)
        x = x + self.time_emb.unsqueeze(0).unsqueeze(2)
        x = x + self.node_emb.unsqueeze(0).unsqueeze(1)
        
        for t_attn, s_attn in zip(self.t_layers, self.s_layers):
            x = x.reshape(B * N, T, -1)
            x = t_attn(x)
            x = x.reshape(B, N, T, -1).permute(0, 2, 1, 3)
            x = x.reshape(B * T, N, -1)
            x = s_attn(x)
            x = x.reshape(B, T, N, -1)
        
        x = x.permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.out_proj(x)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        return x


# ==================== Backbone 3: GRU ====================
class STGRU(nn.Module):
    def __init__(self, num_nodes, in_steps=12, out_steps=12, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_steps = out_steps
        self.hidden_dim = hidden_dim
        
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 10) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, 10) * 0.1)
        
        self.input_proj = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.gcn = GraphConv(hidden_dim, hidden_dim, order=2)
        self.out_proj = nn.Linear(hidden_dim, out_steps)
        
    def forward(self, x):
        B, T, N, C = x.shape
        adj = F.softmax(F.relu(torch.mm(self.node_emb1, self.node_emb2.T)), dim=-1)
        
        x = self.input_proj(x)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = x.reshape(B, N, -1)
        x = self.gcn(x, adj)
        x = self.out_proj(x)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        return x


# ==================== Backbone 4: MLP ====================
class STMLP(nn.Module):
    def __init__(self, num_nodes, in_steps=12, out_steps=12, hidden_dim=32, num_layers=4, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 10) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, 10) * 0.1)
        
        self.input_proj = nn.Linear(in_steps, hidden_dim)
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.mlp = nn.Sequential(*layers)
        self.gcn = GraphConv(hidden_dim, hidden_dim, order=2)
        self.out_proj = nn.Linear(hidden_dim, out_steps)
        
    def forward(self, x):
        B, T, N, C = x.shape
        adj = F.softmax(F.relu(torch.mm(self.node_emb1, self.node_emb2.T)), dim=-1)
        
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.mlp(x)
        x = self.gcn(x, adj)
        x = self.out_proj(x)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        return x


BACKBONES = {
    'stgcn': LightSTGCN,
    'transformer': STTransformer,
    'gru': STGRU,
    'mlp': STMLP,
}


def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask) + 1e-8
    loss = torch.abs(preds - labels) * mask
    loss[torch.isnan(loss)] = 0
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask) + 1e-8
    loss = ((preds - labels) ** 2) * mask
    loss[torch.isnan(loss)] = 0
    return torch.sqrt(torch.mean(loss))


def masked_mape(preds, labels, null_val=0.0):
    mask = (labels != null_val) & (labels > 1.0)
    mask = mask.float()
    mask /= torch.mean(mask) + 1e-8
    loss = torch.abs((preds - labels) / (labels + 1e-8)) * mask
    loss[torch.isnan(loss)] = 0
    loss[torch.isinf(loss)] = 0
    return torch.mean(loss)


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
        if i < num_envs - 1:
            idx = sorted_idx[i * env_size:(i + 1) * env_size]
        else:
            idx = sorted_idx[i * env_size:]
        if len(idx) > 0:
            loss = masked_mae(output[idx], y[idx])
            losses.append(loss)
    
    if len(losses) < 2:
        return masked_mae(output, y), torch.tensor(0.0, device=x_flow.device)
    
    loss_stack = torch.stack(losses)
    mean_loss = loss_stack.mean()
    var_loss = ((loss_stack - mean_loss) ** 2).mean()
    return mean_loss, var_loss


def train_epoch(model, train_loader, optimizer, device, epoch, args):
    model.train()
    total_loss, total_pred, total_rex = 0, 0, 0
    
    if epoch < args.warmup_epochs:
        rex_w = 0.0
    else:
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        rex_w = args.rex_weight * min(1.0, progress * 2)
    
    pbar = tqdm(train_loader, desc='Training')
    for x, y in pbar:
        x, y = x.float().to(device), y.float().to(device)
        x_flow = x[..., :1]
        y_flow = y[..., :1]
        
        optimizer.zero_grad()
        pred_loss, rex_loss = compute_rex_loss(model, x_flow, x, y_flow, args.num_envs)
        loss = pred_loss + rex_w * rex_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_pred += pred_loss.item()
        total_rex += rex_loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n = len(train_loader)
    return total_loss/n, total_pred/n, total_rex/n, rex_w


@torch.no_grad()
def evaluate(model, loader, device, mean, std):
    model.eval()
    preds, labels = [], []
    mean_d = mean[0:1].to(device)
    std_d = std[0:1].to(device)
    
    for x, y in loader:
        x, y = x.float().to(device), y.float().to(device)
        x, y = x[..., :1], y[..., :1]
        output = model(x)
        pred = output * std_d + mean_d
        label = y * std_d + mean_d
        preds.append(pred.cpu())
        labels.append(label.cpu())
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    mae = masked_mae(preds, labels).item()
    rmse = masked_rmse(preds, labels).item()
    mape = masked_mape(preds, labels).item()
    
    return mae, rmse, mape, preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='stgcn', choices=list(BACKBONES.keys()))
    parser.add_argument('--dataset', type=str, default='PEMS03-B')
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
    parser.add_argument('--save_dir', type=str, default='./checkpoints_backbone')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    data_path = f'./{args.dataset}.npz'
    train_loader, val_loader, test_loader, mean, std, num_nodes = get_dataloaders(
        data_path, args.batch_size, in_steps=12, out_steps=12, num_workers=4
    )
    
    logger.info(f"Backbone: {args.backbone}, Dataset: {args.dataset}, Nodes: {num_nodes}")
    
    ModelClass = BACKBONES[args.backbone]
    if args.backbone == 'stgcn':
        model = ModelClass(num_nodes, 12, 12, args.hidden_dim, args.num_blocks, args.dropout)
    else:
        model = ModelClass(num_nodes, 12, 12, args.hidden_dim, args.num_blocks, args.dropout)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    best_mae = float('inf')
    patience_cnt = 0
    ckpt_name = f'{args.backbone}_{args.dataset}.pt'
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        loss, pred_loss, rex_loss, rex_w = train_epoch(model, train_loader, optimizer, device, epoch, args)
        val_mae, val_rmse, val_mape, _, _ = evaluate(model, val_loader, device, mean, std)
        scheduler.step(val_mae)
        
        logger.info(f"Loss: {loss:.4f}, Val: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, MAPE={val_mape:.4f}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, ckpt_name))
            logger.info(f"Saved! Best MAE: {val_mae:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info("Early stopping")
                break
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir, ckpt_name)))
    test_mae, test_rmse, test_mape, preds, labels = evaluate(model, test_loader, device, mean, std)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {args.backbone} on {args.dataset}")
    logger.info(f"{'='*50}")
    
    results = {'backbone': args.backbone, 'dataset': args.dataset, 'params': total_params}
    for h in [3, 6, 12]:
        h_mae = masked_mae(preds[:, h-1:h], labels[:, h-1:h]).item()
        h_rmse = masked_rmse(preds[:, h-1:h], labels[:, h-1:h]).item()
        h_mape = masked_mape(preds[:, h-1:h], labels[:, h-1:h]).item()
        logger.info(f"Horizon {h:2d}: MAE={h_mae:.4f}, RMSE={h_rmse:.4f}, MAPE={h_mape:.4f}")
        results[f'h{h}_mae'] = h_mae
        results[f'h{h}_rmse'] = h_rmse
        results[f'h{h}_mape'] = h_mape
    
    results['overall_mae'] = test_mae
    results['overall_rmse'] = test_rmse
    results['overall_mape'] = test_mape
    logger.info(f"Overall: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.4f}")
    
    result_file = os.path.join(args.save_dir, f'{args.backbone}_{args.dataset}_results.txt')
    with open(result_file, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Results saved to {result_file}")


if __name__ == '__main__':
    main()
