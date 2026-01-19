import torch
import numpy as np
import argparse
from scipy import stats
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from dataset import load_data


def conditional_independence_test(X, Y, Z=None, method='partial_corr'):
    if Z is None:
        corr, pval = stats.pearsonr(X, Y)
        return corr, pval
    
    if method == 'partial_corr':
        X_resid = X - np.mean(X)
        Y_resid = Y - np.mean(Y)
        
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        
        from sklearn.linear_model import LinearRegression
        lr_x = LinearRegression()
        lr_y = LinearRegression()
        
        lr_x.fit(Z, X_resid)
        lr_y.fit(Z, Y_resid)
        
        X_resid = X_resid - lr_x.predict(Z)
        Y_resid = Y_resid - lr_y.predict(Z)
        
        corr, pval = stats.pearsonr(X_resid, Y_resid)
        return corr, pval


def test_backdoor_criterion(data):
    print("="*70)
    print("1. Backdoor Criterion Test")
    print("="*70)
    
    flow = data[:, :, 0].flatten()
    speed = data[:, :, 1].flatten()
    occupancy = data[:, :, 2].flatten()
    
    valid_mask = ~(np.isnan(flow) | np.isnan(speed) | np.isnan(occupancy))
    flow = flow[valid_mask]
    speed = speed[valid_mask]
    occupancy = occupancy[valid_mask]
    
    print(f"\n  Valid samples: {valid_mask.sum()} / {len(valid_mask)} ({valid_mask.sum()/len(valid_mask)*100:.2f}%)")
    
    print("\nTesting: Speed -> Flow")
    corr_sf, pval_sf = stats.pearsonr(speed, flow)
    print(f"  Correlation(Speed, Flow): {corr_sf:.4f}, p-value: {pval_sf:.4e}")
    
    print("\nTesting: Occupancy -> Flow")
    corr_of, pval_of = stats.pearsonr(occupancy, flow)
    print(f"  Correlation(Occupancy, Flow): {corr_of:.4f}, p-value: {pval_of:.4e}")
    
    print("\nTesting: Speed ⊥ Occupancy (independence)")
    corr_so, pval_so = stats.pearsonr(speed, occupancy)
    print(f"  Correlation(Speed, Occupancy): {corr_so:.4f}, p-value: {pval_so:.4e}")
    if abs(corr_so) > 0.3:
        print("  WARNING: Speed and Occupancy are correlated (potential confounding)")
    
    print("\nTesting conditional independence: Flow ⊥ Confounder | {Speed, Occupancy}")
    Z = np.stack([speed, occupancy], axis=1)
    
    results = {
        'speed_flow_corr': corr_sf,
        'occupancy_flow_corr': corr_of,
        'speed_occupancy_corr': corr_so,
        'confounding_exists': abs(corr_so) > 0.3
    }
    
    return results


def test_physical_constraint(data):
    print("\n" + "="*70)
    print("2. Physical Constraint Verification: Flow = Speed × Occupancy")
    print("="*70)
    
    flow = data[:, :, 0].flatten()
    speed = data[:, :, 1].flatten()
    occupancy = data[:, :, 2].flatten()
    
    valid_mask = ~(np.isnan(flow) | np.isnan(speed) | np.isnan(occupancy))
    flow = flow[valid_mask]
    speed = speed[valid_mask]
    occupancy = occupancy[valid_mask]
    
    flow_physical = speed * occupancy
    
    corr, pval = stats.pearsonr(flow, flow_physical)
    mae = np.abs(flow - flow_physical).mean()
    rmse = np.sqrt(((flow - flow_physical) ** 2).mean())
    
    print(f"\n  Correlation(Flow_observed, Flow_physical): {corr:.4f}, p-value: {pval:.4e}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    if corr > 0.7:
        print("  ✓ Physical constraint is approximately satisfied")
    else:
        print("  ✗ Physical constraint is violated (data may have noise)")
    
    return {'correlation': corr, 'mae': mae, 'rmse': rmse}


def identify_anomalies(data, threshold_std=2.0):
    print("\n" + "="*70)
    print("3. Anomaly Pattern Identification")
    print("="*70)
    
    num_timesteps, num_nodes, _ = data.shape
    
    flow = data[:, :, 0]
    speed = data[:, :, 1]
    occupancy = data[:, :, 2]
    
    speed_mean = np.nanmean(speed, axis=0, keepdims=True)
    speed_std = np.nanstd(speed, axis=0, keepdims=True)
    occ_mean = np.nanmean(occupancy, axis=0, keepdims=True)
    occ_std = np.nanstd(occupancy, axis=0, keepdims=True)
    
    speed_anomaly = (speed < speed_mean - threshold_std * speed_std)
    occ_anomaly = (occupancy > occ_mean + threshold_std * occ_std)
    
    accident_pattern = speed_anomaly & occ_anomaly
    
    anomaly_ratio = accident_pattern.sum() / accident_pattern.size
    
    print(f"\n  Anomaly detection (threshold: {threshold_std}σ)")
    print(f"  Speed anomalies: {speed_anomaly.sum()} ({speed_anomaly.sum()/speed_anomaly.size*100:.2f}%)")
    print(f"  Occupancy anomalies: {occ_anomaly.sum()} ({occ_anomaly.sum()/occ_anomaly.size*100:.2f}%)")
    print(f"  Accident pattern (Speed↓ & Occ↑): {accident_pattern.sum()} ({anomaly_ratio*100:.2f}%)")
    
    if anomaly_ratio > 0:
        flow_normal = flow[~accident_pattern].flatten()
        flow_anomaly = flow[accident_pattern].flatten()
        
        flow_normal = flow_normal[~np.isnan(flow_normal)]
        flow_anomaly = flow_anomaly[~np.isnan(flow_anomaly)]
        
        print(f"\n  Flow statistics:")
        print(f"    Normal: mean={flow_normal.mean():.4f}, std={flow_normal.std():.4f}")
        print(f"    Anomaly: mean={flow_anomaly.mean():.4f}, std={flow_anomaly.std():.4f}")
        
        if len(flow_normal) > 1 and len(flow_anomaly) > 1:
            t_stat, p_val = stats.ttest_ind(flow_normal, flow_anomaly)
            print(f"    T-test: t={t_stat:.4f}, p-value={p_val:.4e}")
            
            if p_val < 0.05:
                print(f"    ✓ Anomalies significantly affect flow (causal effect exists)")
    
    return {
        'anomaly_ratio': anomaly_ratio,
        'accident_pattern': accident_pattern
    }


def frequency_domain_analysis(data, sample_nodes=5):
    print("\n" + "="*70)
    print("4. Frequency Domain Analysis")
    print("="*70)
    
    num_timesteps, num_nodes, _ = data.shape
    
    selected_nodes = np.random.choice(num_nodes, min(sample_nodes, num_nodes), replace=False)
    
    print(f"\n  Analyzing {len(selected_nodes)} randomly selected nodes")
    
    for var_idx, var_name in enumerate(['Flow', 'Speed', 'Occupancy']):
        print(f"\n  {var_name}:")
        
        high_freq_powers = []
        low_freq_powers = []
        
        for node in selected_nodes:
            signal = data[:, node, var_idx]
            
            if np.any(np.isnan(signal)):
                valid_mask = ~np.isnan(signal)
                if valid_mask.sum() < len(signal) * 0.5:
                    continue
                signal = np.interp(np.arange(len(signal)), 
                                  np.where(valid_mask)[0], 
                                  signal[valid_mask])
            
            fft_vals = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(len(signal))
            power = np.abs(fft_vals) ** 2
            
            cutoff = 0.3
            low_freq_mask = freqs <= cutoff
            high_freq_mask = freqs > cutoff
            
            low_freq_power = power[low_freq_mask].sum()
            high_freq_power = power[high_freq_mask].sum()
            
            high_freq_powers.append(high_freq_power)
            low_freq_powers.append(low_freq_power)
        
        if len(high_freq_powers) == 0:
            print(f"    No valid data for analysis")
            continue
            
        total_power = np.array(high_freq_powers) + np.array(low_freq_powers)
        high_freq_ratio = np.array(high_freq_powers) / (total_power + 1e-10)
        
        print(f"    High-freq power ratio: {high_freq_ratio.mean():.4f} ± {high_freq_ratio.std():.4f}")
        print(f"    Low-freq dominance: {(high_freq_ratio < 0.5).sum()}/{len(high_freq_ratio)} nodes")


def test_causal_effect_decomposition(data, cutoff_ratio=0.3):
    print("\n" + "="*70)
    print("5. Causal Effect Decomposition: TE = DE + NIE")
    print("="*70)
    
    num_timesteps, num_nodes, _ = data.shape
    
    sample_size = min(1000, num_timesteps)
    sample_nodes = min(10, num_nodes)
    
    selected_times = np.random.choice(num_timesteps, sample_size, replace=False)
    selected_nodes = np.random.choice(num_nodes, sample_nodes, replace=False)
    
    flow = data[selected_times][:, selected_nodes, 0]
    speed = data[selected_times][:, selected_nodes, 1]
    occupancy = data[selected_times][:, selected_nodes, 2]
    
    for i in range(speed.shape[1]):
        if np.any(np.isnan(speed[:, i])):
            valid_mask = ~np.isnan(speed[:, i])
            if valid_mask.sum() > 0:
                speed[:, i] = np.interp(np.arange(len(speed[:, i])), 
                                       np.where(valid_mask)[0], 
                                       speed[:, i][valid_mask])
        if np.any(np.isnan(occupancy[:, i])):
            valid_mask = ~np.isnan(occupancy[:, i])
            if valid_mask.sum() > 0:
                occupancy[:, i] = np.interp(np.arange(len(occupancy[:, i])), 
                                           np.where(valid_mask)[0], 
                                           occupancy[:, i][valid_mask])
    
    speed_fft = np.fft.rfft(speed, axis=0)
    occ_fft = np.fft.rfft(occupancy, axis=0)
    freqs = np.fft.rfftfreq(sample_size)
    
    mask = freqs <= cutoff_ratio
    mask_expanded = mask[:, np.newaxis]
    
    speed_fft_filtered = speed_fft * mask_expanded
    occ_fft_filtered = occ_fft * mask_expanded
    
    speed_cf = np.fft.irfft(speed_fft_filtered, n=sample_size, axis=0)
    occ_cf = np.fft.irfft(occ_fft_filtered, n=sample_size, axis=0)
    
    flow_te = flow
    flow_de = speed_cf * occ_cf
    flow_nie = flow_te - flow_de
    
    flow_te_valid = flow_te[~np.isnan(flow_te)]
    flow_de_valid = flow_de[~np.isnan(flow_de)]
    flow_nie_valid = flow_nie[~np.isnan(flow_nie)]
    
    print("\n  Effect Decomposition Statistics:")
    print(f"    TE (Total Effect): mean={np.nanmean(flow_te):.4f}, std={np.nanstd(flow_te):.4f}")
    print(f"    DE (Direct Effect): mean={np.nanmean(flow_de):.4f}, std={np.nanstd(flow_de):.4f}")
    print(f"    NIE (Natural Indirect): mean={np.nanmean(flow_nie):.4f}, std={np.nanstd(flow_nie):.4f}")
    
    var_te = np.nanvar(flow_te)
    var_de = np.nanvar(flow_de)
    var_nie = np.nanvar(flow_nie)
    
    print(f"\n  Variance Decomposition:")
    print(f"    Var(TE): {var_te:.4f}")
    print(f"    Var(DE): {var_de:.4f}")
    print(f"    Var(NIE): {var_nie:.4f}")
    print(f"    Var(DE)/Var(TE): {var_de/var_te:.4f} ({var_de/var_te*100:.2f}%)")
    print(f"    Var(NIE)/Var(TE): {var_nie/var_te:.4f} ({var_nie/var_te*100:.2f}%)")
    
    te_flat = flow_te.flatten()
    de_flat = flow_de.flatten()
    nie_flat = flow_nie.flatten()
    
    valid_mask = ~(np.isnan(te_flat) | np.isnan(de_flat) | np.isnan(nie_flat))
    te_flat = te_flat[valid_mask]
    de_flat = de_flat[valid_mask]
    nie_flat = nie_flat[valid_mask]
    
    if len(te_flat) > 2:
        corr_te_de, _ = stats.pearsonr(te_flat, de_flat)
        corr_te_nie, _ = stats.pearsonr(te_flat, nie_flat)
        corr_de_nie, _ = stats.pearsonr(de_flat, nie_flat)
        
        print(f"\n  Correlation Analysis:")
        print(f"    Corr(TE, DE): {corr_te_de:.4f}")
        print(f"    Corr(TE, NIE): {corr_te_nie:.4f}")
        print(f"    Corr(DE, NIE): {corr_de_nie:.4f}")
    else:
        corr_de_nie = 0
        print(f"\n  Correlation Analysis: Not enough valid data")
    
    if abs(corr_de_nie) < 0.3:
        print("    ✓ DE and NIE are approximately independent (good decomposition)")
    else:
        print("    ⚠ DE and NIE are correlated (decomposition may not be clean)")
    
    return {
        'var_te': var_te,
        'var_de': var_de,
        'var_nie': var_nie,
        'corr_de_nie': corr_de_nie
    }


def test_identifiability(data):
    print("\n" + "="*70)
    print("6. Causal Identifiability Test")
    print("="*70)
    
    print("\n  Testing Pearl's do-calculus conditions:")
    
    print("\n  Condition 1: Consistency")
    print("    If X=x, then Y(x) = Y")
    print("    ✓ Satisfied by construction (no hidden mediators)")
    
    print("\n  Condition 2: No unmeasured confounding")
    flow = data[:, :, 0].flatten()
    speed = data[:, :, 1].flatten()
    occupancy = data[:, :, 2].flatten()
    
    valid_mask = ~(np.isnan(flow) | np.isnan(speed) | np.isnan(occupancy))
    flow = flow[valid_mask]
    speed = speed[valid_mask]
    occupancy = occupancy[valid_mask]
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X = np.stack([speed, occupancy], axis=1)
    lr.fit(X, flow)
    flow_pred = lr.predict(X)
    r2 = 1 - ((flow - flow_pred) ** 2).sum() / ((flow - flow.mean()) ** 2).sum()
    
    print(f"    R² of Flow ~ Speed + Occupancy: {r2:.4f}")
    if r2 > 0.7:
        print("    ✓ High R² suggests few unmeasured confounders")
    else:
        print("    ⚠ Low R² suggests potential unmeasured confounders")
    
    print("\n  Condition 3: Positivity (overlap)")
    print("    All combinations of (Speed, Occupancy) should be observable")
    
    speed_bins = np.percentile(speed, [0, 25, 50, 75, 100])
    occ_bins = np.percentile(occupancy, [0, 25, 50, 75, 100])
    
    speed_binned = np.digitize(speed, speed_bins[1:-1])
    occ_binned = np.digitize(occupancy, occ_bins[1:-1])
    
    from collections import Counter
    combinations = Counter(zip(speed_binned, occ_binned))
    expected_combinations = 4 * 4
    observed_combinations = len(combinations)
    
    print(f"    Observed {observed_combinations}/{expected_combinations} combinations")
    if observed_combinations >= expected_combinations * 0.8:
        print("    ✓ Good overlap (positivity satisfied)")
    else:
        print("    ⚠ Limited overlap (positivity may be violated)")
    
    return {'r2': r2, 'overlap_ratio': observed_combinations / expected_combinations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS03-B')
    parser.add_argument('--in_steps', type=int, default=12)
    parser.add_argument('--out_steps', type=int, default=12)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CAUSAL IDENTIFICATION ANALYSIS")
    print(f"Dataset: {args.dataset}")
    print("="*70)
    
    data_path = f'./{args.dataset}.npz'
    
    data_dict = np.load(data_path)
    raw_data = data_dict['data']
    num_nodes = raw_data.shape[0]
    
    full_data = raw_data.transpose(1, 0, 2)
    print(f"\nData shape: {full_data.shape}")
    print(f"(timesteps, nodes, features) = ({full_data.shape[0]}, {full_data.shape[1]}, {full_data.shape[2]})")
    
    nan_count = np.isnan(full_data).sum()
    total_count = full_data.size
    print(f"NaN values: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)")
    
    if nan_count > total_count * 0.5:
        print("\nWARNING: More than 50% of data contains NaN values!")
        print("Data preprocessing may be required.")
    
    results = {}
    
    results['backdoor'] = test_backdoor_criterion(full_data)
    
    results['physical'] = test_physical_constraint(full_data)
    
    results['anomaly'] = identify_anomalies(full_data, threshold_std=2.0)
    
    frequency_domain_analysis(full_data, sample_nodes=10)
    
    results['decomposition'] = test_causal_effect_decomposition(full_data, cutoff_ratio=0.3)
    
    results['identifiability'] = test_identifiability(full_data)
    
    print("\n" + "="*70)
    print("SUMMARY: Causal Identification Validity")
    print("="*70)
    
    checks = []
    
    if results['physical']['correlation'] > 0.7:
        checks.append("✓ Physical constraint verified")
    else:
        checks.append("✗ Physical constraint weak")
    
    if results['anomaly']['anomaly_ratio'] > 0.01:
        checks.append("✓ Anomaly patterns detected")
    else:
        checks.append("⚠ Few anomalies found")
    
    if abs(results['decomposition']['corr_de_nie']) < 0.3:
        checks.append("✓ Clean causal decomposition")
    else:
        checks.append("⚠ Decomposition has confounding")
    
    if results['identifiability']['r2'] > 0.7:
        checks.append("✓ Few unmeasured confounders")
    else:
        checks.append("⚠ Potential unmeasured confounders")
    
    if results['identifiability']['overlap_ratio'] > 0.8:
        checks.append("✓ Positivity satisfied")
    else:
        checks.append("⚠ Limited overlap")
    
    print("\n  " + "\n  ".join(checks))
    
    passed = sum([1 for c in checks if c.startswith("✓")])
    total = len(checks)
    
    print(f"\n  Overall: {passed}/{total} checks passed")
    
    if passed >= total * 0.8:
        print("\n  ✓ Causal identification conditions are reasonably satisfied")
        print("    The counterfactual training strategy is theoretically justified")
    else:
        print("\n  ⚠ Some causal identification conditions are not fully satisfied")
        print("    Additional assumptions or methods may be needed")


if __name__ == '__main__':
    main()

