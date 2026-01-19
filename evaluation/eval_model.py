import torch
import numpy as np
import argparse
import sys
import os

sys.path.append('/home/bd2/DB/Torch-MTS/models')

from DCRNN import DCRNN
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


def evaluate(model, test_loader, mean, std, device, model_name, horizons=[3, 6, 12]):
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            x = x[..., :1]
            y = y[..., :1]
            
            if model_name == 'DCRNN':
                output = model(x)
            elif model_name == 'MegaCRN':
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['STGCN', 'DCRNN', 'GWNet', 'MegaCRN'])
    parser.add_argument('--dataset', type=str, required=True, choices=['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B'])
    parser.add_argument('--in_steps', type=int, default=12)
    parser.add_argument('--out_steps', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--corr_threshold', type=float, default=0.3)
    
    args = parser.parse_args()
    
    data_path = f'./{args.dataset}.npz'
    
    adj_dir = './adj_files'
    os.makedirs(adj_dir, exist_ok=True)
    adj_path = os.path.join(adj_dir, f'{args.dataset}_pearson.pkl')
    
    if not os.path.exists(adj_path):
        print(f"Building Pearson correlation adjacency matrix for {args.dataset}...")
        adj = build_pearson_adj(data_path, args.corr_threshold)
        save_adj_pickle(adj, adj_path)
        print(f"Adjacency matrix saved to {adj_path}")
    
    train_loader, val_loader, test_loader, mean, std, num_nodes = get_dataloaders(
        data_path, args.batch_size, args.in_steps, args.out_steps, args.num_workers
    )
    
    print(f'Dataset: {args.dataset}, Model: {args.model}')
    print(f'Num nodes: {num_nodes}')
    
    if args.model == 'DCRNN':
        model = DCRNN(
            num_nodes=num_nodes,
            adj_path=adj_path,
            device=args.device,
            input_dim=1,
            output_dim=1,
            seq_len=args.in_steps,
            horizon=args.out_steps,
            rnn_units=64,
            num_rnn_layers=2,
            max_diffusion_step=2,
            filter_type="dual_random_walk",
            use_teacher_forcing=False,
            tf_decay_steps=2000,
        )
    else:
        print(f"Model {args.model} evaluation not implemented in this script")
        return
    
    model = model.to(args.device)
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded from {args.model_path}")
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(args.device)
            x = x[..., :1]
            _ = model(x)
            break
    
    test_results = evaluate(model, test_loader, mean, std, args.device, args.model)
    print(f'\n{"="*60}')
    print(f'Test Results: {args.dataset} - {args.model}')
    print(f'{"="*60}')
    print(f'Overall - MAE: {test_results["overall"]["mae"]:.4f}, RMSE: {test_results["overall"]["rmse"]:.4f}, MAPE: {test_results["overall"]["mape"]:.4f}')
    for h in [3, 6, 12]:
        if h in test_results:
            print(f'Horizon {h:2d} - MAE: {test_results[h]["mae"]:.4f}, RMSE: {test_results[h]["rmse"]:.4f}, MAPE: {test_results[h]["mape"]:.4f}')
    print(f'{"="*60}')
    
    result_file = './checkpoints_baseline/results.txt'
    with open(result_file, 'a') as f:
        f.write(f'{args.dataset},{args.model}')
        for h in [3, 6, 12]:
            r = test_results[h]
            f.write(f',{r["mae"]:.4f},{r["rmse"]:.4f},{r["mape"]:.4f}')
        r = test_results['overall']
        f.write(f',{r["mae"]:.4f},{r["rmse"]:.4f},{r["mape"]:.4f}\n')
    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    main()

