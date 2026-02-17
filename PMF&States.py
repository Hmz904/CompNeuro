import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, ttest_rel
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 2: PMF ANALYSIS AND MANUSCRIPT-ALIGNED STATISTICS
# ============================================================================

def create_enhanced_features(df):
    """Create features for GLM regression"""
    df = df.copy()
    df['choice_binary'] = (df.get('choice', df.get('response', 0)) == 1).astype(int)
    df['feedback_binary'] = (df.get('feedbackType', 0.5) == 1).astype(int)
    df = df.sort_values(['subject_id', 'date', 'trial_num']).reset_index(drop=True)
    
    # Choice lags
    for lag in range(1, 6):
        df[f'choice_lag_{lag}'] = df.groupby(['subject_id'])['choice_binary'].shift(lag).fillna(0.5)
    df['feedback_lag_1'] = df.groupby(['subject_id'])['feedback_binary'].shift(1).fillna(0.5)
    
    # Exponential prior
    df['positive_contrast'] = (df['contrastLR'] > 0).astype(int)
    priors = []
    for _, group in df.groupby(['subject_id']):
        group_priors = []
        for i in range(len(group)):
            past_contrasts = group['positive_contrast'].iloc[max(0, i-10):i]
            if len(past_contrasts) > 0:
                weights = np.exp(np.arange(len(past_contrasts)) * 0.1)
                prior = np.sum(past_contrasts * weights) / np.sum(weights)
            else:
                prior = 0.5
            group_priors.append(prior)
        priors.extend(group_priors)
    df['exponential_prior'] = priors
    df['choice_x_feedback'] = df['choice_lag_1'] * df['feedback_lag_1']
    return df


def calculate_pmf_with_ci(df_features, window_classifications):
    """Calculate PMF for each state with empirical 95% confidence intervals"""
    print("\n=== PMF ANALYSIS BY STATE ===")
    
    contrast_levels = np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1])
    
    # Expand data: assign state to trials
    expanded_data = []
    for _, window_row in window_classifications.iterrows():
        subject_data = df_features[df_features['subject_id'] == window_row['subject_id']].copy()
        subject_data = subject_data.sort_values(['date', 'trial_num']).reset_index(drop=True)
        
        window_center = window_row['window_center']
        window_start = max(0, int(window_center - 50))
        window_end = min(len(subject_data), int(window_center + 50))
        
        window_trials = subject_data.iloc[window_start:window_end].copy()
        window_trials['state_name'] = window_row['state_name']
        window_trials['cluster_id'] = window_row['cluster_id']
        window_trials['group'] = window_row['group']
        expanded_data.append(window_trials)
    
    all_trials_with_states = pd.concat(expanded_data, ignore_index=True)
    print(f"Total trials with state assignments: {len(all_trials_with_states)}")
    
    pmf_results = {}
    
    for cluster_id in range(1, 9):
        state_name = f"C{cluster_id}"
        state_data = all_trials_with_states[all_trials_with_states['cluster_id'] == cluster_id]
        
        if len(state_data) == 0:
            print(f"  {state_name}: No data")
            continue
        
        print(f"  {state_name}: {len(state_data)} trials")
        
        pmf_data = []
        
        for contrast_level in contrast_levels:
            tolerance = 0.05 if contrast_level != 0 else 0.03
            mask = np.abs(state_data['contrastLR'] - contrast_level) <= tolerance
            subset = state_data[mask]
            
            if len(subset) > 0:
                choices = subset['choice_binary'].values
                
                # Empirical mean and 95% CI using normal approximation
                mean_prob = np.mean(choices)
                sem = np.std(choices) / np.sqrt(len(choices))
                ci_lower = mean_prob - 1.96 * sem
                ci_upper = mean_prob + 1.96 * sem
                
                # Clip to [0, 1]
                ci_lower = np.clip(ci_lower, 0, 1)
                ci_upper = np.clip(ci_upper, 0, 1)
                
                # Calculate accuracy at zero contrast
                accuracy_at_zero = np.nan
                if abs(contrast_level) < 0.01 and 'feedbackType' in subset.columns:
                    correct_responses = subset['feedbackType'].values
                    n_correct = np.sum(correct_responses == 1)
                    n_total = len(correct_responses)
                    accuracy_at_zero = n_correct / n_total if n_total > 0 else np.nan
                
                pmf_data.append({
                    'contrast_level': contrast_level, 
                    'prob_right': mean_prob,
                    'accuracy_at_zero': accuracy_at_zero, 
                    'ci_lower': ci_lower, 
                    'ci_upper': ci_upper, 
                    'sem': sem,
                    'n_trials': len(subset)
                })
            else:
                pmf_data.append({
                    'contrast_level': contrast_level, 
                    'prob_right': np.nan,
                    'accuracy_at_zero': np.nan, 
                    'ci_lower': np.nan, 
                    'ci_upper': np.nan,
                    'sem': np.nan,
                    'n_trials': 0
                })
        
        pmf_results[state_name] = pd.DataFrame(pmf_data)
    
    return pmf_results, all_trials_with_states

def calculate_ideal_pmf_distance(pmf_results):
    """Calculate distance to perfect PMF for ranking states"""
    print("\n=== PMF RANKING BY DISTANCE TO PERFECT PMF ===")
    
    contrast_levels = np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1])
    perfect_pmf_values = np.where(contrast_levels > 0, 1.0, np.where(contrast_levels < 0, 0.0, 0.5))
    
    cluster_distances = []
    
    for state_name, pmf_data in pmf_results.items():
        valid_data = pmf_data.dropna(subset=['prob_right'])
        
        if len(valid_data) >= 3:
            observed_probs = []
            perfect_probs = []
            accuracy_at_zero = None
            
            for _, row in valid_data.iterrows():
                contrast = row['contrast_level']
                if abs(contrast) < 0.01:
                    if not pd.isna(row['accuracy_at_zero']):
                        accuracy_at_zero = row['accuracy_at_zero']
                else:
                    contrast_idx = np.abs(contrast_levels - contrast).argmin()
                    observed_probs.append(row['prob_right'])
                    perfect_probs.append(perfect_pmf_values[contrast_idx])
            
            if len(observed_probs) > 0:
                observed_probs = np.array(observed_probs)
                perfect_probs = np.array(perfect_probs)
                euclidean_dist = np.sqrt(np.sum((observed_probs - perfect_probs)**2))
                mae = np.mean(np.abs(observed_probs - perfect_probs))
            else:
                euclidean_dist = mae = np.inf
            
            accuracy_at_zero = accuracy_at_zero if accuracy_at_zero is not None else 0.5
            accuracy_deficit = abs(accuracy_at_zero - 1.0)
            combined_score = euclidean_dist + mae + accuracy_deficit * 3
            
            cluster_distances.append({
                'state': state_name, 
                'euclidean_distance': euclidean_dist, 
                'mae': mae,
                'accuracy_at_zero': accuracy_at_zero,
                'accuracy_deficit': accuracy_deficit,
                'combined_score': combined_score, 
                'n_points': len(valid_data)
            })
        else:
            cluster_distances.append({
                'state': state_name, 
                'euclidean_distance': np.inf, 
                'mae': np.inf,
                'accuracy_at_zero': 0.5,
                'accuracy_deficit': 0.5,
                'combined_score': np.inf, 
                'n_points': len(valid_data)
            })
    
    cluster_distances.sort(key=lambda x: x['combined_score'])
    
    print("\nCluster Ranking (lower score = better):")
    for rank, result in enumerate(cluster_distances, 1):
        print(f"  Rank {rank}. {result['state']}: Score={result['combined_score']:.2f}, "
              f"Acc@0={result['accuracy_at_zero']:.2f}, d_E={result['euclidean_distance']:.2f}, "
              f"d_MAE={result['mae']:.2f}")
    
    return cluster_distances, perfect_pmf_values

def calculate_quantile_based_statistics(windows_df):
    """Calculate all statistics based on quantile analysis (20 quantiles, 5% each)"""
    print("\n=== QUANTILE-BASED ANALYSIS (20 QUANTILES, 5% EACH) ===")
    
    stats = {}
    
    # 1. High-prior state occupancy (C1-C4) evolution
    print("\n1. HIGH-PRIOR STATE OCCUPANCY (C1-C4) BY QUANTILE:")
    print("-"*80)
    
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group]
        
        # Quantile 1 (0-5%)
        q1_data = group_data[group_data['quantile'] == 0]
        if len(q1_data) > 0:
            hp_q1 = (q1_data['cluster_id'] <= 4).mean() * 100
        else:
            hp_q1 = np.nan
        
        # Quantile 20 (95-100%)
        q20_data = group_data[group_data['quantile'] == 19]
        if len(q20_data) > 0:
            hp_q20 = (q20_data['cluster_id'] <= 4).mean() * 100
        else:
            hp_q20 = np.nan
        
        stats[f'{group}_high_prior_q1'] = hp_q1
        stats[f'{group}_high_prior_q20'] = hp_q20
        
        print(f"{group} quantile 1 (0-5%): {hp_q1:.0f}% in C1-C4")
        print(f"{group} quantile 20 (95-100%): {hp_q20:.0f}% in C1-C4")
    
    # 2. Zero-contrast accuracy by quantile
    print("\n2. ZERO-CONTRAST ACCURACY BY QUANTILE:")
    print("-"*80)
    
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group]
        
        # Quantile 1
        q1_data = group_data[group_data['quantile'] == 0]
        if len(q1_data) > 0:
            acc_q1 = q1_data['accuracy_zero_contrast'].dropna().mean()
        else:
            acc_q1 = np.nan
        
        # Quantile 20
        q20_data = group_data[group_data['quantile'] == 19]
        if len(q20_data) > 0:
            acc_q20 = q20_data['accuracy_zero_contrast'].dropna().mean()
        else:
            acc_q20 = np.nan
        
        stats[f'{group}_zero_acc_q1'] = acc_q1
        stats[f'{group}_zero_acc_q20'] = acc_q20
        
        print(f"{group} quantile 1: Zero-acc {acc_q1:.2f}")
        print(f"{group} quantile 20: Zero-acc {acc_q20:.2f}")
    
    # 3. Correlation between high-prior usage and zero-contrast accuracy
    print("\n3. CORRELATION: HIGH-PRIOR USAGE vs ZERO-CONTRAST ACCURACY:")
    print("-"*80)
    
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group]
        
        quantile_stats = []
        for q in range(20):
            qdata = group_data[group_data['quantile'] == q]
            if len(qdata) > 0:
                hp_prop = (qdata['cluster_id'] <= 4).mean() * 100
                acc = qdata['accuracy_zero_contrast'].dropna().mean()
                if not np.isnan(acc):
                    quantile_stats.append((hp_prop, acc))
        
        if len(quantile_stats) > 2:
            hp_props, accs = zip(*quantile_stats)
            corr, pval = pearsonr(hp_props, accs)
            stats[f'{group}_hp_acc_correlation'] = corr
            stats[f'{group}_hp_acc_pval'] = pval
            print(f"{group}: r = {corr:.2f}, p = {pval:.3f}")
    
    # 4. Early vs late zero-contrast accuracy (first vs last quantiles)
    print("\n4. EARLY VS LATE ZERO-CONTRAST ACCURACY:")
    print("-"*80)
    
    # Get ALL early data (trials 0-1000 approximation: quantiles 0-4)
    # Get ALL late data (trials 4000-5000 approximation: quantiles 16-19)
    
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group]
        
        # Early: quantiles 0-4 (0-25%)
        early_data = group_data[group_data['quantile'] < 5]
        if len(early_data) > 0:
            early_accs = early_data['accuracy_zero_contrast'].dropna()
            early_mean = early_accs.mean()
            early_std = early_accs.std()
        else:
            early_mean, early_std = np.nan, np.nan
        
        # Late: quantiles 16-19 (80-100%)
        late_data = group_data[group_data['quantile'] >= 16]
        if len(late_data) > 0:
            late_accs = late_data['accuracy_zero_contrast'].dropna()
            late_mean = late_accs.mean()
            late_std = late_accs.std()
        else:
            late_mean, late_std = np.nan, np.nan
        
        stats[f'{group}_early_zero_acc_mean'] = early_mean
        stats[f'{group}_early_zero_acc_std'] = early_std
        stats[f'{group}_late_zero_acc_mean'] = late_mean
        stats[f'{group}_late_zero_acc_std'] = late_std
        
        print(f"{group} early (0-25%): {early_mean:.2f} ± {early_std:.2f}")
        print(f"{group} late (80-100%): {late_mean:.2f} ± {late_std:.2f}")
    
    # Paired t-test across all mice (if available)
    all_early = []
    all_late = []
    for subject_id in windows_df['subject_id'].unique():
        subject_data = windows_df[windows_df['subject_id'] == subject_id]
        
        early_sub = subject_data[subject_data['quantile'] < 5]['accuracy_zero_contrast'].dropna()
        late_sub = subject_data[subject_data['quantile'] >= 16]['accuracy_zero_contrast'].dropna()
        
        if len(early_sub) > 0 and len(late_sub) > 0:
            all_early.append(early_sub.mean())
            all_late.append(late_sub.mean())
    
    if len(all_early) > 0 and len(all_late) > 0:
        t_stat, p_val = ttest_rel(all_late, all_early)
        stats['early_vs_late_pval'] = p_val
        print(f"\nPaired t-test (late vs early): p = {p_val:.4f}")
    
    # 5. Windows with high accuracy followed by regression
    print("\n5. HIGH-ACCURACY REGRESSION EVENTS:")
    print("-"*80)
    
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group].sort_values(['subject_id', 'window_center'])
        
        regression_count = 0
        total_high_acc = 0
        
        for subject_id in group_data['subject_id'].unique():
            subject_data_sorted = group_data[group_data['subject_id'] == subject_id]
            accs = subject_data_sorted['accuracy_zero_contrast'].values
            
            for i in range(len(accs) - 1):
                if not np.isnan(accs[i]) and not np.isnan(accs[i+1]):
                    if accs[i] > 0.80:
                        total_high_acc += 1
                        if accs[i+1] < 0.65:
                            regression_count += 1
        
        if total_high_acc > 0:
            regression_pct = (regression_count / total_high_acc) * 100
            stats[f'{group}_high_acc_regression_pct'] = regression_pct
            print(f"{group}: {regression_pct:.0f}% of windows >0.80 followed by <0.65")
    
    # 6. Quantile plateaus and dips
    print("\n6. QUANTILE-SPECIFIC PATTERNS (PLATEAUS AND DIPS):")
    print("-"*80)
    
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group]
        
        # Calculate accuracy by quantile
        quantile_accs = []
        for q in range(20):
            qdata = group_data[group_data['quantile'] == q]
            if len(qdata) > 0:
                acc = qdata['accuracy_zero_contrast'].dropna().mean()
                quantile_accs.append(acc)
            else:
                quantile_accs.append(np.nan)
        
        # Find notable dips
        for q in range(1, 19):
            if not np.isnan(quantile_accs[q-1]) and not np.isnan(quantile_accs[q]) and not np.isnan(quantile_accs[q+1]):
                if quantile_accs[q] < quantile_accs[q-1] and quantile_accs[q] < quantile_accs[q+1]:
                    drop = quantile_accs[q-1] - quantile_accs[q]
                    if drop > 0.03:  # >3% drop
                        print(f"{group} quantile {q+1}: dip from {quantile_accs[q-1]:.2f} to {quantile_accs[q]:.2f} "
                              f"(recovered to {quantile_accs[q+1]:.2f})")
                        stats[f'{group}_dip_quantile'] = q + 1
                        stats[f'{group}_dip_value'] = quantile_accs[q]
                        stats[f'{group}_dip_before'] = quantile_accs[q-1]
                        break
        
        # Find plateaus
        for start_q in range(15):
            plateau_found = False
            for end_q in range(start_q + 4, 20):  # At least 5 quantiles
                if all(not np.isnan(quantile_accs[q]) for q in range(start_q, end_q + 1)):
                    values = [quantile_accs[q] for q in range(start_q, end_q + 1)]
                    if max(values) - min(values) < 0.04:  # <4% variation
                        print(f"{group} plateau: quantiles {start_q+1}-{end_q+1} "
                              f"(accuracy ~{np.mean(values):.2f})")
                        stats[f'{group}_plateau_start'] = start_q + 1
                        stats[f'{group}_plateau_end'] = end_q + 1
                        stats[f'{group}_plateau_mean_acc'] = np.mean(values)
                        plateau_found = True
                        break
            if plateau_found:
                break
    
    # 7. Late training regressions (after trial 3000 ~ quantile 12)
    print("\n7. LATE TRAINING REGRESSIONS (AFTER 60% PROGRESS):")
    print("-"*80)
    
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group]
        
        late_data = group_data[group_data['quantile'] >= 12]  # After 60%
        if len(late_data) > 0:
            low_quality_pct = (late_data['cluster_id'] >= 7).mean() * 100
            stats[f'{group}_late_low_quality_pct'] = low_quality_pct
            print(f"{group}: {low_quality_pct:.0f}% of late windows in C7/C8")
    
    return stats

def calculate_state_occupancy_stats(windows_df, all_trials_with_states):
    """Calculate state occupancy statistics with all transition metrics"""
    print("\n=== STATE OCCUPANCY STATISTICS ===")
    
    stats = {}
    
    # High-prior suboptimal states (C3, C4 ONLY)
    suboptimal_high_prior = [3, 4]
    low_quality_states = [6, 7, 8]
    
    for group in ['ASD', 'WT']:
        group_windows = windows_df[windows_df['group'] == group]
        
        # Percentage in suboptimal high-prior states (C3/C4 ONLY)
        in_suboptimal = group_windows['cluster_id'].isin(suboptimal_high_prior).mean() * 100
        stats[f'{group}_suboptimal_high_prior_pct'] = in_suboptimal
        print(f"{group} time in C3/C4 (high-prior, low-contrast): {in_suboptimal:.1f}%")
    
    # Transition analysis - use EVERY 5TH WINDOW to get more meaningful transitions
    print("\n" + "="*80)
    print("TRANSITION ANALYSIS (using every 5th window for better state changes):")
    print("="*80)
    
    for group in ['ASD', 'WT']:
        group_windows = windows_df[windows_df['group'] == group].sort_values(['subject_id', 'window_center'])
        
        # Collect transitions with larger gaps
        all_transitions = []
        for subject_id in group_windows['subject_id'].unique():
            subject_data = group_windows[group_windows['subject_id'] == subject_id].reset_index(drop=True)
            states = subject_data['cluster_id'].values
            
            # Take every 5th window to allow more state changes
            for i in range(0, len(states) - 5, 5):
                all_transitions.append({
                    'from_state': states[i],
                    'to_state': states[i + 5]
                })
        
        if len(all_transitions) > 0:
            trans_df = pd.DataFrame(all_transitions)
            
            print(f"\n{group} - Total transitions analyzed: {len(trans_df)}")
            
            # 1. Transitions involving C3/C4
            involving_c3c4 = trans_df[
                (trans_df['from_state'].isin(suboptimal_high_prior)) | 
                (trans_df['to_state'].isin(suboptimal_high_prior))
            ]
            involving_pct = (len(involving_c3c4) / len(trans_df)) * 100
            
            stats[f'{group}_suboptimal_transitions_pct'] = involving_pct
            print(f"{group} transitions involving C3/C4: {involving_pct:.0f}%")
            
            # 2. C4 → C1 transitions (PROGRESS)
            c4_transitions = trans_df[trans_df['from_state'] == 4]
            c4_to_c1 = len(trans_df[(trans_df['from_state'] == 4) & (trans_df['to_state'] == 1)])
            
            if len(c4_transitions) > 0:
                c4_to_c1_prob = c4_to_c1 / len(c4_transitions)
                stats[f'{group}_C4_to_C1_prob'] = c4_to_c1_prob
                print(f"{group} P(C4→C1) [progress]: {c4_to_c1_prob:.2f} (n={c4_to_c1}/{len(c4_transitions)})")
            else:
                stats[f'{group}_C4_to_C1_prob'] = 0.0
                print(f"{group} P(C4→C1): No C4 transitions")
            
            # 3. C4 → low-quality (C6/C7/C8) transitions (REVERT)
            c4_to_low = trans_df[(trans_df['from_state'] == 4) & 
                                (trans_df['to_state'].isin(low_quality_states))]
            
            if len(c4_transitions) > 0:
                c4_revert_prob = len(c4_to_low) / len(c4_transitions)
                stats[f'{group}_C4_reversion_prob'] = c4_revert_prob
                print(f"{group} P(C4→low-quality C6/C7/C8) [revert]: {c4_revert_prob:.2f} (n={len(c4_to_low)}/{len(c4_transitions)})")
            else:
                stats[f'{group}_C4_reversion_prob'] = 0.0
            
            # 4. C8 transitions
            c8_transitions = trans_df[trans_df['from_state'] == 8]
            c8_to_c1 = len(trans_df[(trans_df['from_state'] == 8) & (trans_df['to_state'] == 1)])
            c8_to_c4 = len(trans_df[(trans_df['from_state'] == 8) & (trans_df['to_state'] == 4)])
            
            if len(c8_transitions) > 0:
                c8_to_c1_prob = c8_to_c1 / len(c8_transitions)
                c8_to_c4_prob = c8_to_c4 / len(c8_transitions)
                stats[f'{group}_C8_to_C1_prob'] = c8_to_c1_prob
                stats[f'{group}_C8_to_C4_prob'] = c8_to_c4_prob
                print(f"{group} P(C8→C1): {c8_to_c1_prob:.2f} (n={c8_to_c1}/{len(c8_transitions)})")
                print(f"{group} P(C8→C4): {c8_to_c4_prob:.2f} (n={c8_to_c4}/{len(c8_transitions)})")
            else:
                stats[f'{group}_C8_to_C1_prob'] = 0.0
                stats[f'{group}_C8_to_C4_prob'] = 0.0
    
    # Statistical comparisons
    print("\n" + "="*80)
    print("GROUP COMPARISONS (STATISTICAL TESTS):")
    print("="*80)
    
    # 1. Test for C3/C4 involvement difference
    if 'ASD_suboptimal_transitions_pct' in stats and 'WT_suboptimal_transitions_pct' in stats:
        asd_pct = stats['ASD_suboptimal_transitions_pct']
        wt_pct = stats['WT_suboptimal_transitions_pct']
        
        # We need raw counts for chi-square test
        # Re-collect for contingency table
        asd_windows = windows_df[windows_df['group'] == 'ASD'].sort_values(['subject_id', 'window_center'])
        wt_windows = windows_df[windows_df['group'] == 'WT'].sort_values(['subject_id', 'window_center'])
        
        asd_trans = []
        for subject_id in asd_windows['subject_id'].unique():
            subject_data = asd_windows[asd_windows['subject_id'] == subject_id].reset_index(drop=True)
            states = subject_data['cluster_id'].values
            for i in range(0, len(states) - 5, 5):
                involves_c3c4 = (states[i] in suboptimal_high_prior) or (states[i+5] in suboptimal_high_prior)
                asd_trans.append(1 if involves_c3c4 else 0)
        
        wt_trans = []
        for subject_id in wt_windows['subject_id'].unique():
            subject_data = wt_windows[wt_windows['subject_id'] == subject_id].reset_index(drop=True)
            states = subject_data['cluster_id'].values
            for i in range(0, len(states) - 5, 5):
                involves_c3c4 = (states[i] in suboptimal_high_prior) or (states[i+5] in suboptimal_high_prior)
                wt_trans.append(1 if involves_c3c4 else 0)
        
        if len(asd_trans) > 0 and len(wt_trans) > 0:
            from scipy.stats import chi2_contingency
            
            asd_involve = sum(asd_trans)
            asd_not = len(asd_trans) - asd_involve
            wt_involve = sum(wt_trans)
            wt_not = len(wt_trans) - wt_involve
            
            contingency = np.array([[asd_involve, asd_not], [wt_involve, wt_not]])
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            
            stats['c3c4_involvement_pval'] = p_val
            print(f"\nC3/C4 involvement: ASD {asd_pct:.0f}% vs WT {wt_pct:.0f}%")
            print(f"  Chi-square test: χ²={chi2:.2f}, p={p_val:.4f}")
    
    # 2. Test for C4→C1 probability difference
    if 'ASD_C4_to_C1_prob' in stats and 'WT_C4_to_C1_prob' in stats:
        asd_prob = stats['ASD_C4_to_C1_prob']
        wt_prob = stats['WT_C4_to_C1_prob']
        
        # Collect raw C4 transitions
        asd_c4_to_c1_count = 0
        asd_c4_total = 0
        wt_c4_to_c1_count = 0
        wt_c4_total = 0
        
        for group_name, group_windows in [('ASD', asd_windows), ('WT', wt_windows)]:
            for subject_id in group_windows['subject_id'].unique():
                subject_data = group_windows[group_windows['subject_id'] == subject_id].reset_index(drop=True)
                states = subject_data['cluster_id'].values
                for i in range(0, len(states) - 5, 5):
                    if states[i] == 4:
                        if group_name == 'ASD':
                            asd_c4_total += 1
                            if states[i+5] == 1:
                                asd_c4_to_c1_count += 1
                        else:
                            wt_c4_total += 1
                            if states[i+5] == 1:
                                wt_c4_to_c1_count += 1
        
        if asd_c4_total > 0 and wt_c4_total > 0:
            # Chi-square or Fisher's exact test
            contingency = np.array([
                [asd_c4_to_c1_count, asd_c4_total - asd_c4_to_c1_count],
                [wt_c4_to_c1_count, wt_c4_total - wt_c4_to_c1_count]
            ])
            
            if min(contingency.flatten()) >= 5:
                chi2, p_val, dof, expected = chi2_contingency(contingency)
                test_name = "Chi-square"
            else:
                from scipy.stats import fisher_exact
                odds_ratio, p_val = fisher_exact(contingency)
                test_name = "Fisher's exact"
            
            stats['c4_to_c1_pval'] = p_val
            print(f"\nP(C4→C1): ASD {asd_prob:.2f} vs WT {wt_prob:.2f}")
            print(f"  {test_name} test: p={p_val:.4f}")
    
    return stats
    
def calculate_performance_score_trajectories(windows_df, cluster_distances):
    """Calculate performance score trajectories based on 200-trial epochs"""
    print("\n=== PERFORMANCE TRAJECTORY STATISTICS (200-TRIAL EPOCHS) ===")
    
    stats = {}
    
    # Create score mapping
    score_map = {r['state']: r['combined_score'] for r in cluster_distances}
    windows_df['performance_score'] = windows_df['state_name'].apply(
        lambda x: score_map.get(x[:2], np.nan)
    )
    
    # Early vs late performance (based on quantiles, not absolute trial numbers)
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group]
        
        # Early: quantiles 0-4 (0-25%)
        early_windows = group_data[group_data['quantile'] < 5]
        if len(early_windows) > 0:
            early_score = early_windows['performance_score'].mean()
            stats[f'{group}_early_score'] = early_score
            print(f"{group} early score (0-25% quantiles): {early_score:.2f}")
        
        # Late: quantiles 16-19 (80-100%)
        late_windows = group_data[group_data['quantile'] >= 16]
        if len(late_windows) > 0:
            late_score = late_windows['performance_score'].mean()
            stats[f'{group}_late_score'] = late_score
            print(f"{group} late score (80-100% quantiles): {late_score:.2f}")
    
    # Correlation between state quality and performance score (200-trial epochs)
    high_quality_states = [1, 5]
    
    # Create 200-trial epochs
    all_epoch_data = []
    for subject_id in windows_df['subject_id'].unique():
        subject_data = windows_df[windows_df['subject_id'] == subject_id].sort_values('window_center')
        
        # Group into epochs based on window_center
        # Assuming 20 trials per window with step=20, roughly 10 windows per 200 trials
        subject_data['epoch'] = (subject_data['window_center'] // 200).astype(int)
        
        for epoch in subject_data['epoch'].unique():
            epoch_data = subject_data[subject_data['epoch'] == epoch]
            if len(epoch_data) > 0:
                mean_score = epoch_data['performance_score'].mean()
                high_quality_prop = epoch_data['cluster_id'].isin(high_quality_states).mean()
                
                all_epoch_data.append({
                    'subject_id': subject_id,
                    'epoch': epoch,
                    'mean_score': mean_score,
                    'high_quality_proportion': high_quality_prop
                })
    
    if len(all_epoch_data) > 10:
        epoch_df = pd.DataFrame(all_epoch_data)
        valid = epoch_df.dropna(subset=['mean_score', 'high_quality_proportion'])
        
        if len(valid) > 10:
            corr, pval = pearsonr(valid['mean_score'], valid['high_quality_proportion'])
            stats['score_vs_quality_correlation'] = corr
            stats['score_vs_quality_pval'] = pval
            print(f"Correlation (score vs high-quality proportion): r={corr:.2f}, p={pval:.4f}")
    
    return stats, windows_df

def calculate_performance_score_deltas(cluster_distances):
    """Calculate performance improvement deltas between state transitions"""
    print("\n=== STATE TRANSITION PERFORMANCE DELTAS ===")
    
    stats = {}
    
    # Create score dictionary
    score_dict = {r['state']: r['combined_score'] for r in cluster_distances}
    
    # Key transitions
    transitions = [
        ('C8', 'C1'),
        ('C8', 'C4'),
        ('C4', 'C1'),
    ]
    
    for from_state, to_state in transitions:
        if from_state in score_dict and to_state in score_dict:
            delta = score_dict[to_state] - score_dict[from_state]
            stats[f'delta_{from_state}_to_{to_state}'] = delta
            print(f"ΔS ({from_state}→{to_state}): {delta:.2f}")
    
    return stats

def plot_pmf_results(pmf_results, cluster_distances, perfect_pmf_values, output_dir='D:/Neurofn'):
    """Plot PMF analysis results"""
    print("\n=== GENERATING PMF PLOTS ===")
    
    contrast_levels = np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1])
    states_ranked = [result['state'] for result in cluster_distances]
    
    # Figure 1: Individual PMF curves (all 8 states)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    colors = plt.cm.Set3(np.linspace(0, 1, 8))
    
    for i, state_name in enumerate(states_ranked[:8]):
        row, col = i // 4, i % 4
        ax = fig.add_subplot(gs[row, col])
        
        if state_name in pmf_results:
            data = pmf_results[state_name].dropna(subset=['prob_right'])
            if len(data) > 0:
                ax.plot(data['contrast_level'], data['prob_right'], 'o-', 
                       color=colors[i], linewidth=2.5, markersize=7)
                if not data['ci_lower'].isna().all():
                    ax.fill_between(data['contrast_level'], data['ci_lower'], data['ci_upper'], 
                                   color=colors[i], alpha=0.2)
            
            rank = i + 1
            score = cluster_distances[i]['combined_score']
            acc = cluster_distances[i]['accuracy_at_zero']
            d_e = cluster_distances[i]['euclidean_distance']
            mae = cluster_distances[i]['mae']
            
            ax.set_title(f'Rank {rank}: {state_name}\nS={score:.2f}, Acc@0={acc:.2f}\nd_E={d_e:.2f}, MAE={mae:.2f}', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Contrast Level', fontsize=9)
            ax.set_ylabel('P(Choose Right)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            ax.set_xlim(-1.1, 1.1)
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pmf_all_states.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/pmf_all_states.pdf")
    
    # Figure 2: Combined PMF comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(contrast_levels, perfect_pmf_values, 'k-', linewidth=4, 
            label='Perfect PMF', alpha=0.9, marker='s', markersize=8)
    
    for i, state_name in enumerate(states_ranked[:8]):
        if state_name in pmf_results:
            data = pmf_results[state_name].dropna(subset=['prob_right'])
            if len(data) > 0:
                rank = i + 1
                ax.plot(data['contrast_level'], data['prob_right'], 'o-',
                       color=colors[i], linewidth=2, markersize=5, 
                       label=f'Rank {rank}: {state_name}', alpha=0.7)
    
    ax.set_xlabel('Contrast Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('P(Choose Right)', fontsize=14, fontweight='bold')
    ax.set_title('All States vs Perfect PMF', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(-1.1, 1.1)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pmf_combined.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/pmf_combined.pdf")

def save_pmf_results(pmf_results, cluster_distances, all_stats, output_dir='D:/Neurofn'):
    """Save PMF numerical results with comprehensive statistics"""
    
    # Save PMF data for each state
    for state_name, pmf_data in pmf_results.items():
        pmf_data.to_csv(f'{output_dir}/pmf_{state_name}.csv', index=False)
    print(f"Saved: PMF data for {len(pmf_results)} states")
    
    # Save ranking
    ranking_df = pd.DataFrame(cluster_distances)
    ranking_df.to_csv(f'{output_dir}/pmf_state_ranking.csv', index=False)
    print(f"Saved: {output_dir}/pmf_state_ranking.csv")
    
    # Save comprehensive statistics
    with open(f'{output_dir}/pmf_manuscript_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PMF ANALYSIS - MANUSCRIPT STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        # PMF State ranking
        f.write("STATE RANKING:\n")
        f.write("-"*80 + "\n")
        for rank, result in enumerate(cluster_distances, 1):
            f.write(f"\nRank {rank}: {result['state']}\n")
            f.write(f"  Combined Score (S): {result['combined_score']:.2f}\n")
            f.write(f"  Zero-Contrast Accuracy: {result['accuracy_at_zero']:.2f}\n")
            f.write(f"  Euclidean Distance (d_E): {result['euclidean_distance']:.2f}\n")
            f.write(f"  Mean Absolute Error (d_MAE): {result['mae']:.2f}\n")
        
        # Quantile-based statistics
        if 'quantile' in all_stats:
            f.write("\n" + "="*80 + "\n")
            f.write("QUANTILE-BASED STATISTICS (20 QUANTILES, 5% EACH):\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. HIGH-PRIOR STATE OCCUPANCY (C1-C4) EVOLUTION:\n")
            f.write("-"*80 + "\n")
            for group in ['ASD', 'WT']:
                q1_key = f'{group}_high_prior_q1'
                q20_key = f'{group}_high_prior_q20'
                if q1_key in all_stats['quantile']:
                    f.write(f"{group} quantile 1 (0-5%): {all_stats['quantile'][q1_key]:.0f}%\n")
                    f.write(f"{group} quantile 20 (95-100%): {all_stats['quantile'][q20_key]:.0f}%\n")
            
            f.write("\n2. ZERO-CONTRAST ACCURACY BY QUANTILE:\n")
            f.write("-"*80 + "\n")
            for group in ['ASD', 'WT']:
                q1_key = f'{group}_zero_acc_q1'
                q20_key = f'{group}_zero_acc_q20'
                if q1_key in all_stats['quantile']:
                    f.write(f"{group} quantile 1: {all_stats['quantile'][q1_key]:.2f}\n")
                    f.write(f"{group} quantile 20: {all_stats['quantile'][q20_key]:.2f}\n")
            
            f.write("\n3. CORRELATION (HIGH-PRIOR vs ZERO-CONTRAST ACCURACY):\n")
            f.write("-"*80 + "\n")
            for group in ['ASD', 'WT']:
                corr_key = f'{group}_hp_acc_correlation'
                pval_key = f'{group}_hp_acc_pval'
                if corr_key in all_stats['quantile']:
                    f.write(f"{group}: r = {all_stats['quantile'][corr_key]:.2f}, ")
                    f.write(f"p = {all_stats['quantile'][pval_key]:.3f}\n")
            
            f.write("\n4. EARLY VS LATE ZERO-CONTRAST ACCURACY:\n")
            f.write("-"*80 + "\n")
            for group in ['ASD', 'WT']:
                early_key = f'{group}_early_zero_acc_mean'
                early_std_key = f'{group}_early_zero_acc_std'
                late_key = f'{group}_late_zero_acc_mean'
                late_std_key = f'{group}_late_zero_acc_std'
                if early_key in all_stats['quantile']:
                    f.write(f"{group} early (0-25%): {all_stats['quantile'][early_key]:.2f} ± {all_stats['quantile'][early_std_key]:.2f}\n")
                    f.write(f"{group} late (80-100%): {all_stats['quantile'][late_key]:.2f} ± {all_stats['quantile'][late_std_key]:.2f}\n")
            
            if 'early_vs_late_pval' in all_stats['quantile']:
                f.write(f"\nPaired t-test (late vs early): p = {all_stats['quantile']['early_vs_late_pval']:.4f}\n")
            
            f.write("\n5. HIGH-ACCURACY REGRESSION EVENTS:\n")
            f.write("-"*80 + "\n")
            for group in ['ASD', 'WT']:
                reg_key = f'{group}_high_acc_regression_pct'
                if reg_key in all_stats['quantile']:
                    f.write(f"{group}: {all_stats['quantile'][reg_key]:.0f}% of >0.80 windows followed by <0.65\n")
            
            f.write("\n6. LATE TRAINING REGRESSIONS (AFTER 60%):\n")
            f.write("-"*80 + "\n")
            for group in ['ASD', 'WT']:
                late_key = f'{group}_late_low_quality_pct'
                if late_key in all_stats['quantile']:
                    f.write(f"{group}: {all_stats['quantile'][late_key]:.0f}% of late windows in C7/C8\n")
        
        # State occupancy
        if 'occupancy' in all_stats:
            f.write("\n" + "="*80 + "\n")
            f.write("STATE OCCUPANCY STATISTICS:\n")
            f.write("="*80 + "\n")
            for key, val in all_stats['occupancy'].items():
                if 'prob' in key.lower():
                    f.write(f"  {key}: {val:.2f}\n")
                else:
                    f.write(f"  {key}: {val:.1f}%\n")
        
        # Performance trajectories
        if 'trajectories' in all_stats:
            f.write("\n" + "="*80 + "\n")
            f.write("PERFORMANCE TRAJECTORY STATISTICS:\n")
            f.write("="*80 + "\n")
            for key, val in all_stats['trajectories'].items():
                if 'correlation' in key or 'pval' in key:
                    f.write(f"  {key}: {val:.3f}\n")
                else:
                    f.write(f"  {key}: {val:.2f}\n")
        
        # Performance deltas
        if 'deltas' in all_stats:
            f.write("\n" + "="*80 + "\n")
            f.write("STATE TRANSITION PERFORMANCE DELTAS:\n")
            f.write("="*80 + "\n")
            for key, val in all_stats['deltas'].items():
                f.write(f"  {key}: {val:.2f}\n")
    
    print(f"Saved: {output_dir}/pmf_manuscript_statistics.txt")
    
    return all_stats

def run_pmf_analysis_part2(windows_df, df_features, output_dir='D:/Neurofn'):
    """Run comprehensive PMF analysis with manuscript-aligned statistics"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BEHAVIORAL ANALYSIS - PART 2: PMF ANALYSIS")
    print("="*80 + "\n")
    
    # Calculate PMF
    pmf_results, all_trials_with_states = calculate_pmf_with_ci(df_features, windows_df)
    
    # Calculate distances to perfect PMF
    cluster_distances, perfect_pmf_values = calculate_ideal_pmf_distance(pmf_results)
    
    # Calculate all statistics
    quantile_stats = calculate_quantile_based_statistics(windows_df)
    occupancy_stats = calculate_state_occupancy_stats(windows_df, all_trials_with_states)
    trajectory_stats, windows_df = calculate_performance_score_trajectories(windows_df, cluster_distances)
    delta_stats = calculate_performance_score_deltas(cluster_distances)
    
    # Combine all statistics
    all_stats = {
        'quantile': quantile_stats,
        'occupancy': occupancy_stats,
        'trajectories': trajectory_stats,
        'deltas': delta_stats
    }
    
    # Save results
    print("\n" + "="*80)
    print("SAVING PMF RESULTS")
    print("="*80)
    all_stats = save_pmf_results(pmf_results, cluster_distances, all_stats, output_dir)
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING PMF PLOTS")
    print("="*80)
    plot_pmf_results(pmf_results, cluster_distances, perfect_pmf_values, output_dir)
    
    # Display key statistics in Jupyter
    print("\n" + "="*80)
    print("KEY MANUSCRIPT STATISTICS SUMMARY")
    print("="*80)
    
    best = cluster_distances[0]
    worst = cluster_distances[-1]
    print(f"\nBest state: {best['state']} (S={best['combined_score']:.2f}, Acc@0={best['accuracy_at_zero']:.2f})")
    print(f"Worst state: {worst['state']} (S={worst['combined_score']:.2f}, Acc@0={worst['accuracy_at_zero']:.2f})")
    
    print("\n" + "="*80)
    print("PART 2 COMPLETE")
    print("="*80)
    print(f"All PMF results saved to: {output_dir}")
    print("\nAccess statistics via results['statistics']")
    
    return {
        'pmf_results': pmf_results,
        'cluster_distances': cluster_distances,
        'perfect_pmf_values': perfect_pmf_values,
        'windows_df': windows_df,
        'statistics': all_stats
    }


# ============================================================================
# EXECUTE PART 2
# ============================================================================

if __name__ == "__main__":
    # Option 1: Load from Part 1 results if available
    try:
        if 'results' in globals() and results is not None:
            print("Using results from Part 1...")
            pmf_results = run_pmf_analysis_part2(
                results['windows_df'], 
                results['df_features'], 
                output_dir='D:/Neurofn'
            )
    except:
        # Option 2: Load saved data from Part 1
        print("Loading data from Part 1 files...")
        
        output_dir = 'D:/Neurofn'
        
        # Load windows data
        windows_df = pd.read_csv(f'{output_dir}/all_windows_data.csv')
        print(f"Loaded {len(windows_df)} windows")
        
        # Load original features (need to regenerate from raw data)
        df1 = pd.read_csv("D:\\hms\\82ASD_P2_BlockAdded.csv")
        df2 = pd.read_csv("D:\\hms\\78BWM_P2_BlockAdded.csv")
        
        # Prepare data (same as Part 1)
        df1['group'], df2['group'] = 'ASD', 'WT'
        
        for df, prefix in [(df1, 'ASD_'), (df2, 'WT_')]:
            for col in ['subject', 'subjectID', 'mouse', 'animal', 'ID']:
                if col in df.columns:
                    df.rename(columns={col: 'subject_id'}, inplace=True)
                    break
            else:
                df['subject_id'] = prefix + df.groupby('date').ngroup().astype(str)
            
            if not df['subject_id'].astype(str).str.startswith(prefix).all():
                df['subject_id'] = prefix + df['subject_id'].astype(str)
            
            if 'trial_num' not in df.columns:
                df['trial_num'] = df.groupby(['subject_id', 'date']).cumcount() + 1
        
        df_combined = pd.concat([df1, df2], ignore_index=True)
        
        # Get subjects from windows_df
        subjects_in_windows = windows_df['subject_id'].unique()
        df_combined = df_combined[df_combined['subject_id'].isin(subjects_in_windows)]
        
        print(f"Preparing features for {len(df_combined)} trials...")
        df_features = create_enhanced_features(df_combined)
        
        # Run PMF analysis
        pmf_results = run_pmf_analysis_part2(windows_df, df_features, output_dir)
