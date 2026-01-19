
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


def frequency_filter(x, cutoff_ratio=0.3):
    batch_size, in_steps, num_nodes, features = x.shape
    x_filtered = torch.zeros_like(x)
    
    for b in range(batch_size):
        for n in range(num_nodes):
            for f in range(features):
                signal = x[b, :, n, f]
                
                fft_signal = torch.fft.rfft(signal)
                
                freqs = torch.fft.rfftfreq(in_steps)
                cutoff = cutoff_ratio
                mask = freqs <= cutoff
                
                fft_filtered = fft_signal * mask.to(fft_signal.device)
                
                signal_filtered = torch.fft.irfft(fft_filtered, n=in_steps)
                
                x_filtered[b, :, n, f] = signal_filtered
    
    return x_filtered


def detect_anomaly_mask(x, threshold=2.0):
    speed = x[..., 1:2]
    occ = x[..., 2:3]
    
    speed_mean = speed.mean(dim=(0, 1), keepdim=True)
    speed_std = speed.std(dim=(0, 1), keepdim=True) + 1e-6
    occ_mean = occ.mean(dim=(0, 1), keepdim=True)
    occ_std = occ.std(dim=(0, 1), keepdim=True) + 1e-6
    
    speed_z = (speed - speed_mean) / speed_std
    occ_z = (occ - occ_mean) / occ_std
    
    anomaly_mask = ((speed_z < -threshold) & (occ_z > threshold)).float()
    
    return anomaly_mask


def generate_counterfactual_input(x, strategy='debias', cutoff_ratio=0.3, augment_ratio=0.2):
    batch_size, in_steps, num_nodes, features = x.shape
    x_cf = x.clone()
    
    if strategy == 'debias':
        anomaly_mask = detect_anomaly_mask(x, threshold=1.5)
        
        speed_normal = x[..., 1:2].clone()
        occ_normal = x[..., 2:3].clone()
        
        node_speed_median = []
        node_occ_median = []
        for n in range(num_nodes):
            node_data = x[:, :, n:n+1, :]
            node_mask = anomaly_mask[:, :, n:n+1, :]
            
            speed_n = node_data[..., 1:2]
            occ_n = node_data[..., 2:3]
            
            normal_speed = speed_n[node_mask.squeeze(-1) < 0.5]
            normal_occ = occ_n[node_mask.squeeze(-1) < 0.5]
            
            if len(normal_speed) > 0:
                node_speed_median.append(normal_speed.median())
                node_occ_median.append(normal_occ.median())
            else:
                node_speed_median.append(speed_n.median())
                node_occ_median.append(occ_n.median())
        
        node_speed_median = torch.stack(node_speed_median).view(1, 1, num_nodes, 1)
        node_occ_median = torch.stack(node_occ_median).view(1, 1, num_nodes, 1)
        
        speed_normal = torch.where(anomaly_mask > 0.5, node_speed_median, speed_normal)
        occ_normal = torch.where(anomaly_mask > 0.5, node_occ_median, occ_normal)
        
        x_cf[..., 1:2] = speed_normal
        x_cf[..., 2:3] = occ_normal
    
    elif strategy == 'mean':
        speed_mean = x[..., 1:2].mean(dim=(0, 1), keepdim=True)
        occ_mean = x[..., 2:3].mean(dim=(0, 1), keepdim=True)
        x_cf[..., 1:2] = (1 - augment_ratio) * x[..., 1:2] + augment_ratio * speed_mean
        x_cf[..., 2:3] = (1 - augment_ratio) * x[..., 2:3] + augment_ratio * occ_mean
    
    elif strategy == 'frequency':
        speed_filtered = frequency_filter(x[..., 1:2], cutoff_ratio=cutoff_ratio)
        occ_filtered = frequency_filter(x[..., 2:3], cutoff_ratio=cutoff_ratio)
        x_cf[..., 1:2] = (1 - augment_ratio) * x[..., 1:2] + augment_ratio * speed_filtered
        x_cf[..., 2:3] = (1 - augment_ratio) * x[..., 2:3] + augment_ratio * occ_filtered
    
    elif strategy == 'historical':
        batch_mean_speed = x[..., 1:2].mean(dim=1, keepdim=True)
        batch_mean_occ = x[..., 2:3].mean(dim=1, keepdim=True)
        x_cf[..., 1:2] = (1 - augment_ratio) * x[..., 1:2] + augment_ratio * batch_mean_speed
        x_cf[..., 2:3] = (1 - augment_ratio) * x[..., 2:3] + augment_ratio * batch_mean_occ
    
    elif strategy == 'noise':
        noise_speed = torch.randn_like(x[..., 1:2]) * x[..., 1:2].std() * augment_ratio
        noise_occ = torch.randn_like(x[..., 2:3]) * x[..., 2:3].std() * augment_ratio
        x_cf[..., 1:2] = x[..., 1:2] + noise_speed
        x_cf[..., 2:3] = x[..., 2:3] + noise_occ
    
    elif strategy == 'mixup':
        indices = torch.randperm(batch_size, device=x.device)
        x_cf[..., 1:2] = (1 - augment_ratio) * x[..., 1:2] + augment_ratio * x[indices, :, :, 1:2]
        x_cf[..., 2:3] = (1 - augment_ratio) * x[..., 2:3] + augment_ratio * x[indices, :, :, 2:3]
    
    return x_cf


def compute_irm_penalty(model, x, y, num_envs=2, version='v1', env_split='random'):
    batch_size = x.shape[0]
    
    if env_split == 'random':
        env_size = batch_size // num_envs
        env_indices = [list(range(i*env_size, (i+1)*env_size if i < num_envs-1 else batch_size)) 
                      for i in range(num_envs)]
    
    elif env_split == 'variance':
        flow_var = y[..., 0].var(dim=1).squeeze()
        sorted_idx = torch.argsort(flow_var)
        env_size = batch_size // num_envs
        env_indices = [sorted_idx[i*env_size:(i+1)*env_size if i < num_envs-1 else batch_size].tolist()
                      for i in range(num_envs)]
    
    elif env_split == 'magnitude':
        flow_mean = y[..., 0].mean(dim=1).squeeze()
        sorted_idx = torch.argsort(flow_mean)
        env_size = batch_size // num_envs
        env_indices = [sorted_idx[i*env_size:(i+1)*env_size if i < num_envs-1 else batch_size].tolist()
                      for i in range(num_envs)]
    
    else:
        env_size = batch_size // num_envs
        env_indices = [list(range(i*env_size, (i+1)*env_size if i < num_envs-1 else batch_size)) 
                      for i in range(num_envs)]
    
    if version == 'v1':
        output = model(x)
        
        dummy_w = torch.ones(1, device=x.device, requires_grad=True)
        
        penalties = []
        for i in range(num_envs):
            start_idx = i * env_size
            end_idx = (i + 1) * env_size if i < num_envs - 1 else batch_size
            
            output_env = output[start_idx:end_idx]
            y_env = y[start_idx:end_idx]
            
            loss = masked_mae(output_env * dummy_w, y_env[..., :1])
            
            grad = torch.autograd.grad(loss, dummy_w, create_graph=True)[0]
            penalties.append(grad ** 2)
        
        return torch.stack(penalties).mean()
    
    elif version == 'v1_fast':
        output = model(x)
        
        losses = []
        for i in range(num_envs):
            start_idx = i * env_size
            end_idx = (i + 1) * env_size if i < num_envs - 1 else batch_size
            
            output_env = output[start_idx:end_idx]
            y_env = y[start_idx:end_idx]
            
            loss = masked_mae(output_env, y_env[..., :1])
            losses.append(loss)
        
        loss_stack = torch.stack(losses)
        loss_var = torch.var(loss_stack)
        
        return loss_var
    
    elif version == 'simple':
        grad_norms = []
        for i in range(num_envs):
            start_idx = i * env_size
            end_idx = (i + 1) * env_size if i < num_envs - 1 else batch_size
            
            x_env = x[start_idx:end_idx]
            y_env = y[start_idx:end_idx]
            
            output = model(x_env)
            loss = masked_mae(output, y_env[..., :1])
            
            grad = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
            grad_norm = sum([(g ** 2).sum() for g in grad if g is not None])
            grad_norms.append(grad_norm)
        
        penalty = sum([(g - grad_norms[0]) ** 2 for g in grad_norms[1:]])
        return penalty / (num_envs - 1)


def train_epoch(model, train_loader, optimizer, device, cf_weight=0.3, cf_strategy='frequency', cutoff_ratio=0.3, augment_ratio=0.2):
    model.train()
    total_loss = 0
    total_factual_loss = 0
    total_cf_loss = 0
    
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc='Training')):
        x = x.float().to(device)
        y = y.float().to(device)
        
        optimizer.zero_grad()
        
        if cf_strategy == 'irm':
            output = model(x)
            factual_loss = masked_mae(output, y[..., :1])
            
            irm_penalty = compute_irm_penalty(model, x, y, num_envs=2, version='v1')
            
            loss = factual_loss + cf_weight * irm_penalty
            total_factual_loss += factual_loss.item()
            total_cf_loss += irm_penalty.item()
            
        elif cf_strategy == 'irm_fast':
            output = model(x)
            factual_loss = masked_mae(output, y[..., :1])
            
            irm_penalty = compute_irm_penalty(model, x, y, num_envs=2, version='v1_fast')
            
            loss = factual_loss + cf_weight * irm_penalty
            total_factual_loss += factual_loss.item()
            total_cf_loss += irm_penalty.item()
            
        elif cf_strategy == 'causal_attention':
            output = model(x)
            factual_loss = masked_mae(output, y[..., :1])
            
            x_speed = x[..., 1:2]
            x_occ = x[..., 2:3]
            
            x_speed_shuffled = x_speed[torch.randperm(x_speed.shape[0])]
            x_occ_shuffled = x_occ[torch.randperm(x_occ.shape[0])]
            
            x_shuffled = x.clone()
            x_shuffled[..., 1:2] = x_speed_shuffled
            x_shuffled[..., 2:3] = x_occ_shuffled
            
            output_shuffled = model(x_shuffled)
            
            invariance_loss = torch.abs(output - output_shuffled).mean()
            
            loss = factual_loss + cf_weight * invariance_loss
            total_factual_loss += factual_loss.item()
            total_cf_loss += invariance_loss.item()
        
        else:
            output_factual = model(x)
            factual_loss = masked_mae(output_factual, y[..., :1])
            
            x_cf = generate_counterfactual_input(x, strategy=cf_strategy, cutoff_ratio=cutoff_ratio, augment_ratio=augment_ratio)
            output_cf = model(x_cf)
            
            if cf_strategy == 'debias':
                cf_loss = masked_mae(output_cf, y[..., :1])
            else:
                cf_loss = masked_mae(output_cf, output_factual.detach())
            
            loss = factual_loss + cf_weight * cf_loss
            total_factual_loss += factual_loss.item()
            total_cf_loss += cf_loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return (total_loss / len(train_loader), 
            total_factual_loss / len(train_loader),
            total_cf_loss / len(train_loader))


def evaluate(model, val_loader, mean, std, device, use_counterfactual=False, cf_strategy='frequency', cutoff_ratio=0.3, augment_ratio=0.2, horizons=[3, 6, 12]):
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc='Evaluating'):
            x = x.float().to(device)
            y = y.float().to(device)
            
            if use_counterfactual:
                output_factual = model(x)
                x_cf = generate_counterfactual_input(x, strategy=cf_strategy, cutoff_ratio=cutoff_ratio, augment_ratio=augment_ratio)
                output_cf = model(x_cf)
                output = (output_factual + output_cf) / 2
            else:
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
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_cf')
    
    parser.add_argument('--cf_weight', type=float, default=0.1)
    parser.add_argument('--cf_strategy', type=str, default='irm_fast', choices=['irm', 'irm_fast', 'causal_attention', 'debias', 'mean', 'frequency', 'historical', 'noise', 'mixup'])
    parser.add_argument('--cutoff_ratio', type=float, default=0.3)
    parser.add_argument('--augment_ratio', type=float, default=0.2)
    parser.add_argument('--use_cf_inference', action='store_true')
    
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate, no training')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model for evaluation')
    
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--input_embedding_dim', type=int, default=24)
    parser.add_argument('--tod_embedding_dim', type=int, default=0)
    parser.add_argument('--dow_embedding_dim', type=int, default=0)
    parser.add_argument('--adaptive_embedding_dim', type=int, default=80)
    parser.add_argument('--feed_forward_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    # python train_cf.py --dataset PEMS03-B --eval_only --model_path ./checkpoints_cf/PEMS03-B_best.pth
    args = parser.parse_args()
    
    log_dir = f'./logs/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_cf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
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
    logging.info(f'Counterfactual weight: {args.cf_weight}')
    logging.info(f'Counterfactual strategy: {args.cf_strategy}')
    logging.info(f'Augment ratio: {args.augment_ratio}')
    if args.cf_strategy == 'frequency':
        logging.info(f'Frequency cutoff ratio: {args.cutoff_ratio}')
    logging.info(f'Use CF inference: {args.use_cf_inference}')
    
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
        logging.info('Model loaded successfully!')
        
        test_results = evaluate(model, test_loader, mean, std, args.device,
                              use_counterfactual=args.use_cf_inference,
                              cf_strategy=args.cf_strategy,
                              cutoff_ratio=args.cutoff_ratio,
                              augment_ratio=args.augment_ratio)
        logging.info(f'\n{"="*70}')
        logging.info(f'Test Results on {args.dataset} (Evaluation Only)')
        logging.info(f'Model: {args.model_path}')
        logging.info(f'Strategy: {args.cf_strategy} | Augment Ratio: {args.augment_ratio} | CF Inference: {args.use_cf_inference}')
        logging.info(f'{"="*70}')
        logging.info(f'Overall - MAE: {test_results["overall"]["mae"]:.4f}, RMSE: {test_results["overall"]["rmse"]:.4f}, MAPE: {test_results["overall"]["mape"]:.4f}')
        for h in [3, 6, 12]:
            if h in test_results:
                logging.info(f'Horizon {h:2d} - MAE: {test_results[h]["mae"]:.4f}, RMSE: {test_results[h]["rmse"]:.4f}, MAPE: {test_results[h]["mape"]:.4f}')
        logging.info(f'{"="*70}')
        return
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_mae = float('inf')
    patience_counter = 0
    early_stop_patience = 3
    
    for epoch in range(args.epochs):
        logging.info(f'\nEpoch {epoch+1}/{args.epochs}')
        
        train_loss, factual_loss, cf_loss = train_epoch(
            model, train_loader, optimizer, args.device, args.cf_weight, args.cf_strategy, args.cutoff_ratio, args.augment_ratio
        )
        logging.info(f'Train Loss: {train_loss:.4f} (Factual: {factual_loss:.4f}, Consistency: {cf_loss:.4f})')
        
        val_results = evaluate(model, val_loader, mean, std, args.device, 
                             use_counterfactual=args.use_cf_inference, 
                             cf_strategy=args.cf_strategy,
                             cutoff_ratio=args.cutoff_ratio,
                             augment_ratio=args.augment_ratio)
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
    test_results = evaluate(model, test_loader, mean, std, args.device,
                          use_counterfactual=args.use_cf_inference,
                          cf_strategy=args.cf_strategy,
                          cutoff_ratio=args.cutoff_ratio,
                          augment_ratio=args.augment_ratio)
    logging.info(f'\n{"="*70}')
    logging.info(f'Test Results on {args.dataset} (Counterfactual Data Augmentation)')
    logging.info(f'Strategy: {args.cf_strategy} | Augment Ratio: {args.augment_ratio} | CF Inference: {args.use_cf_inference}')
    logging.info(f'{"="*70}')
    logging.info(f'Overall - MAE: {test_results["overall"]["mae"]:.4f}, RMSE: {test_results["overall"]["rmse"]:.4f}, MAPE: {test_results["overall"]["mape"]:.4f}')
    for h in [3, 6, 12]:
        if h in test_results:
            logging.info(f'Horizon {h:2d} - MAE: {test_results[h]["mae"]:.4f}, RMSE: {test_results[h]["rmse"]:.4f}, MAPE: {test_results[h]["mape"]:.4f}')
    logging.info(f'{"="*70}')


if __name__ == '__main__':
    main()

