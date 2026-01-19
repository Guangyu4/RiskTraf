#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import pickle
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('/home/bd2/DB')
sys.path.append('/home/bd2/DB/Dish-TS')
sys.path.append('/home/bd2/DB/Torch-MTS/models')

from STSSDL import STSSDL
from DishTS import DishTS
from steve_model import STEVE
from STGCN import STGCN
from GraphWaveNet import GWNET
from MegaCRN import MegaCRN
from dataset import get_dataloaders


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
        
        num_nodes = x.shape[0] if len(x.shape) == 2 else x.shape[1]
        x_tod = np.arange(t, t+self.in_steps) % self.TDAY / self.TDAY
        y_tod = np.arange(t+self.in_steps, t+self.in_steps+self.out_steps) % self.TDAY / self.TDAY
        x_cov = np.tile(x_tod[:, np.newaxis, np.newaxis], (1, num_nodes, 1))
        y_cov = np.tile(y_tod[:, np.newaxis, np.newaxis], (1, num_nodes, 1))
        
        return {
            'x_flow': x_flow.astype(np.float32),
            'y_flow': y_flow.astype(np.float32),
            'x_aux': x_aux.astype(np.float32),
            'x_cov': x_cov.astype(np.float32),
            'y_cov': y_cov.astype(np.float32),
        }


def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        result[key] = torch.FloatTensor(np.stack([b[key] for b in batch]))
    return result


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
    test_indices = np.arange(train_steps + val_steps, total_steps)
    
    test_dataset = PEMSBDatasetOOD(data, test_indices, in_steps, out_steps, TDAY)
    return test_dataset, mean, std, num_nodes


def get_adj(data_path, dataset_name, device):
    adj_path = f'./adj_files/{dataset_name}_pearson.pkl'
    with open(adj_path, 'rb') as f:
        adj = pickle.load(f)
    adj = torch.FloatTensor(adj)
    row_sum = adj.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    adj_norm = adj / row_sum
    return [adj_norm.to(device)]


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
        for _ in range(self.horizon):
            out, h = self.decoder(dec_input, h)
            pred = self.proj(out)
            outputs.append(pred)
            dec_input = pred
        output = torch.cat(outputs, dim=1).reshape(B, self.horizon, N, 1)
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
        return output_denorm.unsqueeze(-1)


def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    if mask.sum() == 0:
        return torch.tensor(0.0)
    mae = torch.abs(preds - labels)
    return (mae * mask).sum() / mask.sum()


def eval_ood_model(model, model_name, test_loader, device, mean, std, adj=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {model_name}"):
            x_flow = batch['x_flow'].to(device)
            y_flow = batch['y_flow'].to(device)
            x_aux = batch['x_aux'].to(device)
            x_cov = batch['x_cov'].to(device)
            y_cov = batch['y_cov'].to(device)
            
            if model_name == 'ST-SSDL':
                x_combined = torch.cat([x_flow, x_aux], dim=-1)
                output, _, _, _, _, _, _ = model(x_combined, x_cov, x_combined.clone(), y_cov)
                pred = output
            elif model_name == 'STEVE':
                x_combined = torch.cat([x_flow, x_aux], dim=-1)
                H, Z = model(x_combined, adj)
                Y = model.predict_test(Z, H)
                pred = Y.repeat(1, 12, 1, 1)
            elif model_name == 'Dish-TS':
                x_combined = torch.cat([x_flow, x_aux], dim=-1)
                pred = model(x_combined)
            
            mean_f = mean[..., 0:1].to(device)
            std_f = std[..., 0:1].to(device)
            pred = pred * std_f + mean_f
            label = y_flow * std_f + mean_f
            
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())
    
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    mae_per_horizon = []
    for h in range(1, 13):
        if h <= preds.shape[1]:
            mae = masked_mae(preds[:, h-1:h, :, :], labels[:, h-1:h, :, :]).item()
        else:
            mae = masked_mae(preds[:, -1:, :, :], labels[:, h-1:h, :, :]).item()
        mae_per_horizon.append(mae)
    
    return mae_per_horizon


def eval_3dim_model(model, model_name, test_loader, device, mean, std):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"Eval {model_name}"):
            x = x.float().to(device)
            y = y.float().to(device)
            
            if model_name == 'MegaCRN':
                y_cov = torch.zeros(x.shape[0], 12, x.shape[2], 1, device=device)
                output, _, _, _, _ = model(x, y_cov)
            else:
                output = model(x)
            
            all_preds.append(output.cpu())
            all_labels.append(y.cpu())
    
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    preds = preds * std + mean
    labels = labels * std + mean
    
    preds_flow = preds[..., 0:1]
    labels_flow = labels[..., 0:1]
    
    mae_per_horizon = []
    for h in range(1, 13):
        mae = masked_mae(preds_flow[:, h-1:h, :, :], labels_flow[:, h-1:h, :, :]).item()
        mae_per_horizon.append(mae)
    
    return mae_per_horizon


def load_ood_model(model_name, dataset, num_nodes, device):
    adj_mx = get_adj(f'./{dataset}.npz', dataset, device)
    
    if model_name == 'ST-SSDL':
        model = STSSDL(
            num_nodes=num_nodes, input_dim=3, output_dim=1, horizon=12, rnn_units=64,
            rnn_layers=1, cheb_k=3, ycov_dim=1, prototype_num=20, prototype_dim=64,
            tod_embed_dim=10, adj_mx=adj_mx, cl_decay_steps=2000, TDAY=288,
            use_curriculum_learning=True, use_STE=False, device=device,
            adaptive_embedding_dim=48, node_embedding_dim=20, input_embedding_dim=64
        )
    elif model_name == 'STEVE':
        model = STEVE(num_nodes=num_nodes, input_dim=3, embed_size=64, input_length=12,
                      output_dim=1, dropout=0.1, device=device)
    elif model_name == 'Dish-TS':
        model = DishTSModel(num_nodes=num_nodes, input_dim=3, hidden_dim=64, 
                            output_dim=1, horizon=12, seq_len=12)
    
    ckpt_path = f'./checkpoints_ood/{dataset}_{model_name}_best.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device), adj_mx[0] if model_name == 'STEVE' else None


def load_3dim_model(model_name, dataset, num_nodes, device):
    adj_path = f'./adj_files/{dataset}_pearson.pkl'
    
    if model_name == 'STGCN':
        model = STGCN(n_vertex=num_nodes, adj_path=adj_path, Kt=3, Ks=3,
                      blocks=[[3], [64, 16, 64], [64, 16, 64], [128, 128], [12]],
                      T=12, act_func="glu", graph_conv_type="cheb_graph_conv", bias=True, droprate=0.5)
    elif model_name == 'GWNet':
        model = GWNET(device=device, num_nodes=num_nodes, adj_path=adj_path, adj_type="doubletransition",
                      dropout=0.3, gcn_bool=True, addaptadj=True, in_dim=3, out_dim=12,
                      residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                      kernel_size=2, blocks=4, layers=2)
    elif model_name == 'MegaCRN':
        model = MegaCRN(num_nodes=num_nodes, input_dim=3, output_dim=3, horizon=12, rnn_units=64,
                        num_layers=1, cheb_k=3, ycov_dim=1, mem_num=20, mem_dim=64,
                        tf_decay_steps=2000, use_teacher_forcing=True)
    
    ckpt_path = f'./checkpoints_3dim/{dataset}_{model_name}_best.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B']
    ood_models = ['ST-SSDL', 'Dish-TS', 'STEVE']
    dim3_models = ['STGCN', 'GWNet', 'MegaCRN']
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating {dataset}")
        print('='*60)
        
        all_results[dataset] = {}
        
        test_dataset_ood, mean_ood, std_ood, num_nodes = load_data_ood(f'./{dataset}.npz')
        test_loader_ood = torch.utils.data.DataLoader(test_dataset_ood, batch_size=64, shuffle=False, 
                                                       num_workers=4, collate_fn=collate_fn)
        
        _, _, test_loader_3dim, mean_3dim, std_3dim, _ = get_dataloaders(
            f'./{dataset}.npz', batch_size=64, in_steps=12, out_steps=12, num_workers=4)
        
        for model_name in ood_models:
            print(f"\n  {model_name}...")
            try:
                model, adj = load_ood_model(model_name, dataset, num_nodes, device)
                mae_list = eval_ood_model(model, model_name, test_loader_ood, device, mean_ood, std_ood, adj)
                all_results[dataset][model_name] = mae_list
                print(f"    MAE: {[f'{m:.2f}' for m in mae_list]}")
            except Exception as e:
                print(f"    Error: {e}")
                all_results[dataset][model_name] = [0]*12
        
        for model_name in dim3_models:
            print(f"\n  {model_name}...")
            try:
                model = load_3dim_model(model_name, dataset, num_nodes, device)
                mae_list = eval_3dim_model(model, model_name, test_loader_3dim, device, mean_3dim, std_3dim)
                all_results[dataset][model_name] = mae_list
                print(f"    MAE: {[f'{m:.2f}' for m in mae_list]}")
            except Exception as e:
                print(f"    Error: {e}")
                all_results[dataset][model_name] = [0]*12
    
    os.makedirs('./eval_results', exist_ok=True)
    with open('./eval_results/mae_per_horizon.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to ./eval_results/mae_per_horizon.json")
    
    plot_results(all_results)


def plot_results(results):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    colors = {
        'ST-SSDL': '#E74C3C',
        'Dish-TS': '#3498DB', 
        'STEVE': '#2ECC71',
        'STGCN': '#9B59B6',
        'GWNet': '#F39C12',
        'MegaCRN': '#1ABC9C',
        'RiskTraf': '#E91E63'
    }
    markers = {
        'ST-SSDL': 'o',
        'Dish-TS': 's',
        'STEVE': '^',
        'STGCN': 'D',
        'GWNet': 'v',
        'MegaCRN': 'p',
        'RiskTraf': '*'
    }
    linestyles = {
        'ST-SSDL': '-',
        'Dish-TS': '-',
        'STEVE': '-',
        'STGCN': '--',
        'GWNet': '--',
        'MegaCRN': '--',
        'RiskTraf': '-'
    }
    
    datasets = ['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B']
    horizons = list(range(1, 13))
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = results[dataset]
        
        for model_name in ['RiskTraf', 'ST-SSDL', 'Dish-TS', 'STEVE', 'MegaCRN', 'STGCN', 'GWNet']:
            if model_name in data:
                mae_list = data[model_name]
                if max(mae_list) < 80:
                    ax.plot(horizons, mae_list, color=colors[model_name], marker=markers[model_name],
                            label=model_name, linewidth=2.5, markersize=6, linestyle=linestyles[model_name])
        
        ax.set_xlabel(f'{dataset}', fontsize=16, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('MAE', fontsize=16)
        ax.set_xticks(horizons)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('./eval_results/mae_horizons.png', dpi=150, bbox_inches='tight')
    plt.savefig('./eval_results/mae_horizons.pdf', bbox_inches='tight')
    print("Plots saved to ./eval_results/mae_horizons.png and .pdf")
    plt.close()


if __name__ == '__main__':
    main()
