#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
import argparse

sys.path.append('/home/bd2/DB/Torch-MTS/models')
sys.path.append('/home/bd2/DB')
sys.path.append('/home/bd2/DB/Dish-TS')

from STGCN import STGCN
from GraphWaveNet import GWNET
from MegaCRN import MegaCRN
from STSSDL import STSSDL
from steve_model import STEVE
from train_light_stgcn import LightSTGCN
from train_ood_models import DishTSModel


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, input_data, extra_inputs=None, warmup=10, runs=100, model_name=""):
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _ in range(warmup):
            if model_name == "MegaCRN":
                y_cov = torch.zeros(input_data.shape[0], 12, input_data.shape[2], 1, device=device)
                _ = model(input_data, y_cov)
            elif model_name == "ST-SSDL":
                x_cov = extra_inputs['x_cov']
                y_cov = extra_inputs['y_cov']
                _ = model(input_data, x_cov, input_data, y_cov)
            elif model_name == "STEVE":
                adj = extra_inputs['adj']
                H, Z = model(input_data, adj)
                _ = model.predict_test(Z, H)
            else:
                _ = model(input_data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            if model_name == "MegaCRN":
                y_cov = torch.zeros(input_data.shape[0], 12, input_data.shape[2], 1, device=device)
                _ = model(input_data, y_cov)
            elif model_name == "ST-SSDL":
                x_cov = extra_inputs['x_cov']
                y_cov = extra_inputs['y_cov']
                _ = model(input_data, x_cov, input_data, y_cov)
            elif model_name == "STEVE":
                adj = extra_inputs['adj']
                H, Z = model(input_data, adj)
                _ = model.predict_test(Z, H)
            else:
                _ = model(input_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    return np.mean(times) * 1000, np.std(times) * 1000


def measure_memory(model, input_data, extra_inputs=None, model_name=""):
    if not torch.cuda.is_available():
        return 0.0
    
    device = next(model.parameters()).device
    model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        if model_name == "MegaCRN":
            y_cov = torch.zeros(input_data.shape[0], 12, input_data.shape[2], 1, device=device)
            _ = model(input_data, y_cov)
        elif model_name == "ST-SSDL":
            x_cov = extra_inputs['x_cov']
            y_cov = extra_inputs['y_cov']
            _ = model(input_data, x_cov, input_data, y_cov)
        elif model_name == "STEVE":
            adj = extra_inputs['adj']
            H, Z = model(input_data, adj)
            _ = model.predict_test(Z, H)
        else:
            _ = model(input_data)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    return peak_mem


def build_adj(num_nodes, device):
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2
    adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return adj.to(device)


def create_adj_file(num_nodes, adj_path):
    import pickle
    adj = np.random.rand(num_nodes, num_nodes).astype(np.float32)
    adj = (adj + adj.T) / 2
    adj[adj < 0.5] = 0
    np.fill_diagonal(adj, 0)
    with open(adj_path, 'wb') as f:
        pickle.dump(adj, f)
    return adj_path


def get_model_and_input(model_name, num_nodes, device, in_steps=12, out_steps=12, batch_size=32):
    adj_dir = './adj_files'
    os.makedirs(adj_dir, exist_ok=True)
    adj_path = os.path.join(adj_dir, f'efficiency_test_{num_nodes}.pkl')
    if not os.path.exists(adj_path):
        create_adj_file(num_nodes, adj_path)
    
    extra_inputs = {}
    
    if model_name == "STGCN":
        model = STGCN(
            n_vertex=num_nodes,
            adj_path=adj_path,
            Kt=3, Ks=3,
            blocks=[[3], [64, 16, 64], [64, 16, 64], [128, 128], [out_steps]],
            T=in_steps, act_func="glu", graph_conv_type="cheb_graph_conv",
            bias=True, droprate=0.5
        ).to(device)
        x = torch.randn(batch_size, in_steps, num_nodes, 3).to(device)
    
    elif model_name == "GWNet":
        model = GWNET(
            device=device, num_nodes=num_nodes, adj_path=adj_path,
            adj_type="doubletransition", dropout=0.3, gcn_bool=True,
            addaptadj=True, in_dim=3, out_dim=out_steps,
            residual_channels=32, dilation_channels=32,
            skip_channels=256, end_channels=512, kernel_size=2,
            blocks=4, layers=2
        ).to(device)
        x = torch.randn(batch_size, in_steps, num_nodes, 3).to(device)
    
    elif model_name == "MegaCRN":
        model = MegaCRN(
            num_nodes=num_nodes, input_dim=3, output_dim=3,
            horizon=out_steps, rnn_units=64, num_layers=1,
            cheb_k=3, ycov_dim=1, mem_num=20, mem_dim=64,
            tf_decay_steps=2000, use_teacher_forcing=False
        ).to(device)
        x = torch.randn(batch_size, in_steps, num_nodes, 3).to(device)
    
    elif model_name == "ST-SSDL":
        adj = build_adj(num_nodes, device)
        adj_mx = [adj]
        model = STSSDL(
            num_nodes=num_nodes, input_dim=3, output_dim=1,
            horizon=out_steps, rnn_units=64, rnn_layers=1,
            cheb_k=3, ycov_dim=1, prototype_num=20, prototype_dim=64,
            tod_embed_dim=10, adj_mx=adj_mx, cl_decay_steps=2000,
            TDAY=288, use_curriculum_learning=False, use_STE=False,
            device=device, adaptive_embedding_dim=48, node_embedding_dim=20,
            input_embedding_dim=64
        ).to(device)
        x = torch.randn(batch_size, in_steps, num_nodes, 3).to(device)
        x_cov = torch.randn(batch_size, in_steps, num_nodes, 1).to(device)
        y_cov = torch.randn(batch_size, out_steps, num_nodes, 1).to(device)
        extra_inputs = {'x_cov': x_cov, 'y_cov': y_cov}
    
    elif model_name == "STEVE":
        adj = build_adj(num_nodes, device)
        model = STEVE(
            num_nodes=num_nodes, input_dim=3, embed_size=64,
            input_length=in_steps, output_dim=1, dropout=0.1, device=device
        ).to(device)
        x = torch.randn(batch_size, in_steps, num_nodes, 3).to(device)
        extra_inputs = {'adj': adj}
    
    elif model_name == "Dish-TS":
        model = DishTSModel(
            num_nodes=num_nodes, input_dim=3, hidden_dim=64,
            output_dim=1, horizon=out_steps, seq_len=in_steps
        ).to(device)
        x = torch.randn(batch_size, in_steps, num_nodes, 3).to(device)
    
    elif model_name == "LightSTGCN":
        model = LightSTGCN(
            num_nodes=num_nodes, in_steps=in_steps, out_steps=out_steps,
            hidden_dim=32, num_blocks=4, dropout=0.2
        ).to(device)
        x = torch.randn(batch_size, in_steps, num_nodes, 1).to(device)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, x, extra_inputs


def generate_latex_table(results):
    models_order = ["STGCN", "GWNet", "MegaCRN", "ST-SSDL", "Dish-TS", "STEVE", "LightSTGCN"]
    
    categories = {
        "STGCN": "SOTA",
        "GWNet": "SOTA",
        "MegaCRN": "SOTA",
        "ST-SSDL": "OOD",
        "Dish-TS": "OOD",
        "STEVE": "OOD",
        "LightSTGCN": "Ours"
    }
    
    ours = results.get("LightSTGCN", {})
    ours_params = ours.get('params', 1)
    ours_time = ours.get('time_mean', 1)
    ours_mem = ours.get('memory', 1) if ours.get('memory', 0) > 0 else 1
    
    best_params = min(r['params'] for r in results.values())
    best_time = min(r['time_mean'] for r in results.values())
    best_mem = min(r['memory'] for r in results.values() if r['memory'] > 0)
    
    latex = r"""\begin{table}[t]
\centering
\caption{Efficiency comparison of different models on PEMS-B datasets. The batch size is 32, and all models are evaluated on a single NVIDIA RTX 3090 GPU. ``Ratio'' indicates the parameter ratio compared to our LightSTGCN model.}
\label{tab:efficiency}
\resizebox{\linewidth}{!}{
\begin{tabular}{l|c|rr|rr|rr}
\toprule
\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Category}} & \multicolumn{2}{c|}{\textbf{Parameters}} & \multicolumn{2}{c|}{\textbf{Inference Time}} & \multicolumn{2}{c}{\textbf{Memory}} \\
& & \textbf{Count (K)} & \textbf{Ratio} & \textbf{Time (ms)} & \textbf{Ratio} & \textbf{MB} & \textbf{Ratio} \\
\midrule
"""
    
    for model in models_order:
        if model not in results:
            continue
        r = results[model]
        cat = categories[model]
        
        param_ratio = r['params'] / ours_params
        time_ratio = r['time_mean'] / ours_time
        mem_ratio = r['memory'] / ours_mem if r['memory'] > 0 else 0
        
        params_str = f"{r['params']/1000:.1f}"
        if r['params'] == best_params:
            params_str = r"\textbf{" + params_str + "}"
        
        time_str = f"{r['time_mean']:.2f}"
        if r['time_mean'] == best_time:
            time_str = r"\textbf{" + time_str + "}"
        
        mem_str = f"{r['memory']:.1f}" if r['memory'] > 0 else "-"
        if r['memory'] > 0 and r['memory'] == best_mem:
            mem_str = r"\textbf{" + mem_str + "}"
        
        param_ratio_str = f"{param_ratio:.1f}$\\times$"
        if param_ratio == 1.0:
            param_ratio_str = r"\textbf{1.0$\times$}"
        
        time_ratio_str = f"{time_ratio:.1f}$\\times$"
        if time_ratio < 1.0:
            time_ratio_str = r"\textbf{" + f"{time_ratio:.1f}$\\times$" + "}"
        elif time_ratio == 1.0:
            time_ratio_str = r"\textbf{1.0$\times$}"
            
        mem_ratio_str = f"{mem_ratio:.1f}$\\times$" if mem_ratio > 0 else "-"
        if 0 < mem_ratio < 1.0:
            mem_ratio_str = r"\textbf{" + f"{mem_ratio:.1f}$\\times$" + "}"
        elif mem_ratio == 1.0:
            mem_ratio_str = r"\textbf{1.0$\times$}"
        
        latex += f"{model} & {cat} & {params_str} & {param_ratio_str} & {time_str} & {time_ratio_str} & {mem_str} & {mem_ratio_str} \\\\\n"
        
        if model == "MegaCRN":
            latex += r"\midrule" + "\n"
        elif model == "STEVE":
            latex += r"\midrule" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
}
\end{table}
"""
    
    avg_sota_params = np.mean([results[m]['params'] for m in ["STGCN", "GWNet", "MegaCRN"] if m in results])
    avg_ood_params = np.mean([results[m]['params'] for m in ["ST-SSDL", "Dish-TS", "STEVE"] if m in results])
    
    print(f"\nAnalysis Summary:")
    print(f"  Our LightSTGCN has {ours_params:,} params")
    print(f"  vs SOTA avg ({avg_sota_params/1000:.1f}K): {avg_sota_params/ours_params:.1f}x reduction")
    print(f"  vs OOD avg ({avg_ood_params/1000:.1f}K): {avg_ood_params/ours_params:.1f}x reduction")
    
    return latex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=358)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Nodes: {args.num_nodes}, Batch: {args.batch_size}")
    print("=" * 60)
    
    models = ["STGCN", "GWNet", "MegaCRN", "ST-SSDL", "Dish-TS", "STEVE", "LightSTGCN"]
    results = {}
    
    for model_name in models:
        print(f"\nAnalyzing {model_name}...")
        try:
            model, x, extra = get_model_and_input(
                model_name, args.num_nodes, device, batch_size=args.batch_size
            )
            
            params = count_params(model)
            print(f"  Parameters: {params:,}")
            
            time_mean, time_std = measure_inference_time(
                model, x, extra, args.warmup, args.runs, model_name
            )
            print(f"  Inference Time: {time_mean:.2f} ± {time_std:.2f} ms")
            
            memory = measure_memory(model, x, extra, model_name)
            print(f"  Peak Memory: {memory:.1f} MB")
            
            results[model_name] = {
                'params': params,
                'time_mean': time_mean,
                'time_std': time_std,
                'memory': memory
            }
            
            del model, x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"{'Model':<15} {'Params':>12} {'Time (ms)':>15} {'Memory (MB)':>12}")
    print("-" * 60)
    for model_name, r in results.items():
        print(f"{model_name:<15} {r['params']:>12,} {r['time_mean']:>10.2f}±{r['time_std']:.2f} {r['memory']:>12.1f}")
    
    latex = generate_latex_table(results)
    print("\n" + "=" * 60)
    print("LaTeX Table:")
    print("=" * 60)
    print(latex)
    
    with open('./eval_results/efficiency_table.tex', 'w') as f:
        f.write(latex)
    print("\nTable saved to ./eval_results/efficiency_table.tex")


if __name__ == '__main__':
    main()
