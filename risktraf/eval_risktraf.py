import torch
import argparse
import numpy as np
from train_light_stgcn import LightSTGCN, get_dataloaders, masked_mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS03-B')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_light_stgcn')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path = f'./{args.dataset}.npz'
    _, _, test_loader, mean, std, num_nodes = get_dataloaders(
        data_path, args.batch_size, in_steps=12, out_steps=12, num_workers=4
    )
    
    model = LightSTGCN(
        num_nodes=num_nodes,
        in_steps=12,
        out_steps=12,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout
    ).to(device)
    
    ckpt_path = f'{args.save_dir}/best_{args.dataset}.pt'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    print(f"\n{'='*50}")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*50}")
    
    preds_list, labels_list = [], []
    mean_d = mean[0:1].to(device)
    std_d = std[0:1].to(device)
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(device)[..., :1]
            y = y.float().to(device)[..., :1]
            
            out = model(x)
            out = out * std_d + mean_d
            y = y * std_d + mean_d
            
            preds_list.append(out.cpu())
            labels_list.append(y.cpu())
    
    preds = torch.cat(preds_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    print(f"\n{'Horizon':<10} {'MAE':>10}")
    print("-" * 22)
    
    for h in range(1, 13):
        h_mae = masked_mae(preds[:, h-1:h], labels[:, h-1:h]).item()
        print(f"{h:<10} {h_mae:>10.4f}")
    
    overall_mae = masked_mae(preds, labels).item()
    print("-" * 22)
    print(f"{'Overall':<10} {overall_mae:>10.4f}")


if __name__ == '__main__':
    main()
