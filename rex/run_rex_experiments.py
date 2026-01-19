import subprocess
import os
import re

datasets = ['PEMS03-B', 'PEMS04-B', 'PEMS07-B', 'PEMS08-B']
save_dir = './checkpoints_rex'
os.makedirs(save_dir, exist_ok=True)

results = []

for dataset in datasets:
    print(f"\n{'='*70}")
    print(f"Training V-REx on {dataset}")
    print(f"{'='*70}")
    
    cmd = [
        'python', 'train_rex.py',
        '--dataset', dataset,
        '--rex_weight', '1.0',
        '--num_envs', '3',
        '--env_split', 'magnitude',
        '--warmup_epochs', '5',
        '--patience', '15',
        '--epochs', '100',
        '--batch_size', '64',
        '--save_dir', save_dir,
        '--lr', '0.001',
        '--num_layers', '3'
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    process.wait()
    
    output = ''.join(output_lines)
    
    test_section = output.split('Test Results on')[-1] if 'Test Results on' in output else output
    
    h3_match = re.search(r'Horizon\s+3\s+-\s+MAE:\s+([\d.]+),\s+RMSE:\s+([\d.]+),\s+MAPE:\s+([\d.]+)', test_section)
    h6_match = re.search(r'Horizon\s+6\s+-\s+MAE:\s+([\d.]+),\s+RMSE:\s+([\d.]+),\s+MAPE:\s+([\d.]+)', test_section)
    h12_match = re.search(r'Horizon\s+12\s+-\s+MAE:\s+([\d.]+),\s+RMSE:\s+([\d.]+),\s+MAPE:\s+([\d.]+)', test_section)
    overall_match = re.search(r'Overall\s+-\s+MAE:\s+([\d.]+),\s+RMSE:\s+([\d.]+),\s+MAPE:\s+([\d.]+)', test_section)
    
    if all([h3_match, h6_match, h12_match, overall_match]):
        row = [
            dataset, 'V-REx',
            h3_match.group(1), h3_match.group(2), h3_match.group(3),
            h6_match.group(1), h6_match.group(2), h6_match.group(3),
            h12_match.group(1), h12_match.group(2), h12_match.group(3),
            overall_match.group(1), overall_match.group(2), overall_match.group(3)
        ]
        results.append(','.join(row))
        print(f"\n{dataset} Result: {','.join(row)}")

results_file = os.path.join(save_dir, 'results.txt')
with open(results_file, 'w') as f:
    for r in results:
        f.write(r + '\n')

print(f"\n{'='*70}")
print(f"All experiments completed!")
print(f"Results saved to {results_file}")
print(f"{'='*70}")
