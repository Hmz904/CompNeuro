import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency, pearsonr, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: CORE ANALYSIS WITH COMPREHENSIVE DRIFT ANALYSIS
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

def sliding_window_glm(df_features, df_raw, window_size=100, step_size=20):
    """Sliding window GLM with zero-contrast accuracy calculation - STEP SIZE = 20"""
    all_windows = []
    feature_columns = ['choice_lag_1', 'choice_lag_2', 'choice_lag_3', 'choice_lag_4', 'choice_lag_5',
                      'contrastLR', 'feedback_lag_1', 'exponential_prior', 'choice_x_feedback']
    
    for subject_id in df_features['subject_id'].unique():
        subject_data = df_features[df_features['subject_id'] == subject_id].copy()
        subject_raw = df_raw[df_raw['subject_id'] == subject_id].copy()
        subject_data = subject_data.sort_values(['date', 'trial_num']).reset_index(drop=True)
        subject_raw = subject_raw.sort_values(['date', 'trial_num']).reset_index(drop=True)
        
        if len(subject_data) < window_size:
            continue
        
        group = subject_data['group'].iloc[0]
        
        for start_idx in range(0, len(subject_data) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = subject_data.iloc[start_idx:end_idx].copy()
            window_raw = subject_raw.iloc[start_idx:end_idx].copy()
            
            X, y = window_data[feature_columns].values, window_data['choice_binary'].values
            
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                continue
            
            # Calculate overall accuracy
            if 'feedbackType' in window_raw.columns:
                feedback = window_raw['feedbackType'].values
                accuracy_overall = np.sum(feedback == 1) / len(feedback)
            else:
                accuracy_overall = np.nan
            
            # Calculate zero-contrast accuracy
            if 'contrastLR' in window_raw.columns and 'feedbackType' in window_raw.columns:
                zero_contrast_mask = np.abs(window_raw['contrastLR']) < 0.01
                zero_contrast_trials = window_raw[zero_contrast_mask]
                
                if len(zero_contrast_trials) > 0:
                    accuracy_zero_contrast = np.sum(zero_contrast_trials['feedbackType'] == 1) / len(zero_contrast_trials)
                    n_zero_contrast = len(zero_contrast_trials)
                else:
                    accuracy_zero_contrast = np.nan
                    n_zero_contrast = 0
            else:
                accuracy_zero_contrast = np.nan
                n_zero_contrast = 0
            
            try:
                log_reg = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=42)
                log_reg.fit(X, y)
                coefficients = log_reg.coef_[0]
                
                result = {
                    'subject_id': subject_id, 'group': group, 'window_start': start_idx,
                    'window_center': (start_idx + end_idx) / 2, 
                    'accuracy': accuracy_score(y, log_reg.predict(X)),
                    'accuracy_overall': accuracy_overall,
                    'accuracy_zero_contrast': accuracy_zero_contrast,
                    'n_zero_contrast_trials': n_zero_contrast,
                    'window_id': f"{subject_id}_{start_idx}"
                }
                
                coef_names = ['choice_lag_1', 'choice_lag_2', 'choice_lag_3', 'choice_lag_4', 'choice_lag_5',
                             'contrast', 'feedback_lag_1', 'exponential_prior', 'choice_x_feedback']
                for i, name in enumerate(coef_names):
                    result[f'coef_{name}'] = coefficients[i]
                
                all_windows.append(result)
            except:
                continue
    
    return pd.DataFrame(all_windows)

def three_factor_clustering(all_windows_df):
    """Cluster windows by contrast, prior, WSLS with PCA analysis"""
    print("\n=== THREE-FACTOR CLUSTERING ===")
    
    contrast_coef = all_windows_df['coef_contrast'].values
    prior_coef = all_windows_df['coef_exponential_prior'].values
    wsls_coef = all_windows_df['coef_choice_x_feedback'].values
    
    contrast_median = np.median(contrast_coef)
    prior_median = np.median(prior_coef)
    wsls_median = np.median(wsls_coef)
    
    print(f"Median coefficients: β^contrast_med = {contrast_median:.2f}, β^prior_med = {prior_median:.2f}, β^WSLS_med = {wsls_median:.2f}")
    
    # PCA analysis for independence verification
    coef_matrix = np.column_stack([contrast_coef, prior_coef, wsls_coef])
    pca = PCA(n_components=2)
    pca.fit(coef_matrix)
    variance_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA: First 2 components explain {variance_explained:.1f}% of variance (66.7% expected under independence)")
    
    # Binary classification
    contrast_high = (contrast_coef >= contrast_median).astype(int)
    prior_high = (prior_coef >= prior_median).astype(int)
    wsls_high = (wsls_coef >= wsls_median).astype(int)
    
    # Cluster ID mapping
    temp_id = prior_high * 4 + contrast_high * 2 + wsls_high * 1
    cluster_mapping = {
        7: 1,  # 1,1,1
        6: 2,  # 1,1,0
        5: 3,  # 1,0,1
        4: 4,  # 1,0,0
        3: 5,  # 0,1,1
        2: 6,  # 0,1,0
        1: 7,  # 0,0,1
        0: 8   # 0,0,0
    }
    
    all_windows_df['cluster_id'] = pd.Series(temp_id).map(cluster_mapping).values
    
    cluster_names = {
        1: 'C1_High_Prior_High_Contrast_High_WSLS',
        2: 'C2_High_Prior_High_Contrast_Low_WSLS',
        3: 'C3_High_Prior_Low_Contrast_High_WSLS',
        4: 'C4_High_Prior_Low_Contrast_Low_WSLS',
        5: 'C5_Low_Prior_High_Contrast_High_WSLS',
        6: 'C6_Low_Prior_High_Contrast_Low_WSLS',
        7: 'C7_Low_Prior_Low_Contrast_High_WSLS',
        8: 'C8_Low_Prior_Low_Contrast_Low_WSLS'
    }
    
    all_windows_df['state_name'] = all_windows_df['cluster_id'].map(cluster_names)
    all_windows_df['is_high_prior'] = all_windows_df['cluster_id'] <= 4
    
    # Print cluster occupancy
    print("\nEmpirical cluster occupancies:")
    for cluster_id in range(1, 9):
        n = np.sum(all_windows_df['cluster_id'] == cluster_id)
        pct = n / len(all_windows_df) * 100
        print(f"  C{cluster_id}: {pct:.1f}% (expected 12.5% under independence)")
    
    # Store key statistics
    stats = {
        'pca_variance_2pc': variance_explained,
        'contrast_median': contrast_median,
        'prior_median': prior_median,
        'wsls_median': wsls_median,
        'cluster_occupancies': {f'C{i}': (all_windows_df['cluster_id'] == i).mean() * 100 for i in range(1, 9)}
    }
    
    return all_windows_df, stats

def calculate_group_statistics(windows_df):
    """Calculate key group-level statistics"""
    print("\n=== GROUP STATISTICS ===")
    
    stats = {}
    
    # Optimal state occupancy (C1 + C2)
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group]
        optimal_pct = ((group_data['cluster_id'] == 1) | (group_data['cluster_id'] == 2)).mean() * 100
        stats[f'{group}_optimal_occupancy'] = optimal_pct
        print(f"{group} optimal state (C1+C2) occupancy: {optimal_pct:.1f}%")
    
    # Specific state occupancies
    for cluster_id in [6, 8]:
        asd_pct = (windows_df[windows_df['group'] == 'ASD']['cluster_id'] == cluster_id).mean() * 100
        wt_pct = (windows_df[windows_df['group'] == 'WT']['cluster_id'] == cluster_id).mean() * 100
        stats[f'C{cluster_id}_ASD'] = asd_pct
        stats[f'C{cluster_id}_WT'] = wt_pct
        print(f"C{cluster_id}: ASD {asd_pct:.1f}% vs WT {wt_pct:.1f}%")
    
    # Self-transition probabilities
    for group in ['ASD', 'WT']:
        group_windows = windows_df[windows_df['group'] == group].sort_values(['subject_id', 'window_center'])
        
        self_transitions = []
        for subject_id in group_windows['subject_id'].unique():
            subject_data = group_windows[group_windows['subject_id'] == subject_id]
            states = subject_data['cluster_id'].values
            for i in range(len(states) - 1):
                if states[i] == states[i+1]:
                    self_transitions.append(1)
                else:
                    self_transitions.append(0)
        
        self_trans_prob = np.mean(self_transitions) if len(self_transitions) > 0 else 0
        stats[f'{group}_self_transition'] = self_trans_prob
        print(f"{group} mean self-transition probability: {self_trans_prob:.2f}")
    
    # C8 self-transition
    for group in ['ASD', 'WT']:
        group_windows = windows_df[windows_df['group'] == group].sort_values(['subject_id', 'window_center'])
        
        c8_self_trans = []
        for subject_id in group_windows['subject_id'].unique():
            subject_data = group_windows[group_windows['subject_id'] == subject_id]
            states = subject_data['cluster_id'].values
            for i in range(len(states) - 1):
                if states[i] == 8:
                    c8_self_trans.append(1 if states[i+1] == 8 else 0)
        
        c8_prob = np.mean(c8_self_trans) if len(c8_self_trans) > 0 else 0
        stats[f'{group}_C8_self_transition'] = c8_prob
        print(f"{group} C8 self-transition probability: {c8_prob:.2f}")
    
    # C1 to suboptimal transitions
    for group in ['ASD', 'WT']:
        group_windows = windows_df[windows_df['group'] == group].sort_values(['subject_id', 'window_center'])
        
        c1_to_suboptimal = []
        for subject_id in group_windows['subject_id'].unique():
            subject_data = group_windows[group_windows['subject_id'] == subject_id]
            states = subject_data['cluster_id'].values
            for i in range(len(states) - 1):
                if states[i] == 1:
                    if states[i+1] in [3, 4, 7, 8]:
                        c1_to_suboptimal.append(1)
                    else:
                        c1_to_suboptimal.append(0)
        
        c1_subopt_prob = np.mean(c1_to_suboptimal) if len(c1_to_suboptimal) > 0 else 0
        stats[f'{group}_C1_to_suboptimal'] = c1_subopt_prob
        print(f"{group} C1 to suboptimal transition probability: {c1_subopt_prob:.2f}")
    
    return stats

def calculate_temporal_evolution(windows_df):
    """Calculate temporal evolution statistics"""
    print("\n=== TEMPORAL EVOLUTION ===")
    
    stats = {}
    
    for group in ['ASD', 'WT']:
        group_data = windows_df[windows_df['group'] == group]
        
        # Quantile 1 (0-5%)
        q1_data = group_data[group_data['quantile'] == 0]
        if len(q1_data) > 0:
            hp_q1 = q1_data['is_high_prior'].mean() * 100
            acc_q1 = q1_data['accuracy_zero_contrast'].mean()
        else:
            hp_q1, acc_q1 = np.nan, np.nan
        
        # Quantile 20 (95-100%)
        q20_data = group_data[group_data['quantile'] == 19]
        if len(q20_data) > 0:
            hp_q20 = q20_data['is_high_prior'].mean() * 100
            acc_q20 = q20_data['accuracy_zero_contrast'].mean()
        else:
            hp_q20, acc_q20 = np.nan, np.nan
        
        stats[f'{group}_high_prior_q1'] = hp_q1
        stats[f'{group}_high_prior_q20'] = hp_q20
        stats[f'{group}_zero_acc_q1'] = acc_q1
        stats[f'{group}_zero_acc_q20'] = acc_q20
        
        print(f"{group} quantile 1: High-prior {hp_q1:.0f}%, Zero-contrast acc {acc_q1:.2f}")
        print(f"{group} quantile 20: High-prior {hp_q20:.0f}%, Zero-contrast acc {acc_q20:.2f}")
        
        # Correlation across quantiles
        quantile_stats = []
        for q in range(20):
            qdata = group_data[group_data['quantile'] == q]
            if len(qdata) > 0:
                hp_prop = qdata['is_high_prior'].mean() * 100
                acc = qdata['accuracy_zero_contrast'].dropna().mean()
                if not np.isnan(acc):
                    quantile_stats.append((hp_prop, acc))
        
        if len(quantile_stats) > 2:
            hp_props, accs = zip(*quantile_stats)
            corr, pval = pearsonr(hp_props, accs)
            stats[f'{group}_hp_acc_correlation'] = corr
            stats[f'{group}_hp_acc_pval'] = pval
            print(f"{group} high-prior vs accuracy correlation: r = {corr:.2f}, p = {pval:.3f}")
    
    return stats

def assign_quantiles(windows_df, n_quantiles=20):
    """Assign quantiles for temporal analysis"""
    windows_df['quantile'] = np.nan
    windows_df['subject_window_order'] = np.nan
    
    for subject_id in windows_df['subject_id'].unique():
        subject_mask = windows_df['subject_id'] == subject_id
        subject_windows = windows_df[subject_mask].sort_values('window_center')
        
        n_windows = len(subject_windows)
        window_orders = np.arange(n_windows)
        
        quantile_assignments = np.floor((window_orders / n_windows) * n_quantiles).astype(int)
        quantile_assignments[quantile_assignments >= n_quantiles] = n_quantiles - 1
        
        windows_df.loc[subject_windows.index, 'quantile'] = quantile_assignments
        windows_df.loc[subject_windows.index, 'subject_window_order'] = window_orders
    
    return windows_df

def detect_valid_block_switches(df):
    """Detect valid block switches for drift analysis"""
    valid_switches = []
    pre_switch_window = 50
    post_switch_max = 15
    
    for subject_id in df['subject_id'].unique():
        subject_data = df[df['subject_id'] == subject_id].sort_values(['date', 'trial_num']).reset_index(drop=True)
        
        if 'Block' not in subject_data.columns:
            continue
        
        # Find block switches
        switch_indices = []
        for i in range(1, len(subject_data)):
            if subject_data.iloc[i]['Block'] != subject_data.iloc[i-1]['Block']:
                switch_indices.append(i)
        
        # Validate each switch
        for switch_idx in switch_indices:
            if switch_idx < pre_switch_window:
                continue
            
            pre_switch_start = switch_idx - pre_switch_window
            pre_switch_data = subject_data.iloc[pre_switch_start:switch_idx]
            
            if len(pre_switch_data['Block'].unique()) == 1:
                if switch_idx + post_switch_max < len(subject_data):
                    valid_switches.append({
                        'subject_id': subject_id,
                        'group': subject_data.iloc[switch_idx]['group'],
                        'switch_idx': switch_idx,
                        'switch_id': f"{subject_id}_{switch_idx}"
                    })
    
    return pd.DataFrame(valid_switches)

def fit_glm_for_drift(window_data):
    """Helper function to fit GLM and return coefficients"""
    try:
        choices = (window_data['choice'] == 1).astype(int).values
        contrasts = window_data['contrastLR'].values
        
        # Choice history
        choice_history = np.zeros(len(choices))
        choice_history[1:] = choices[:-1]
        
        # Feedback history for WSLS
        if 'feedbackType' in window_data.columns:
            feedback = window_data['feedbackType'].values
            feedback_history = np.zeros(len(feedback))
            feedback_history[1:] = feedback[:-1]
            wsls = choice_history * feedback_history
        else:
            wsls = choice_history
        
        # Prior (positive contrast proportion)
        prior = np.zeros(len(contrasts))
        for i in range(len(contrasts)):
            if i > 0:
                prior[i] = np.mean(contrasts[max(0, i-10):i] > 0)
            else:
                prior[i] = 0.5
        
        if len(np.unique(choices)) >= 2 and len(window_data) >= 5:
            X = np.column_stack([contrasts, prior, wsls])
            y = choices
            lr = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=42)
            lr.fit(X, y)
            
            return {
                'contrast': lr.coef_[0][0],
                'prior': lr.coef_[0][1],
                'wsls': lr.coef_[0][2]
            }
    except:
        pass
    
    return None

def calculate_drift_metrics_comprehensive(df, valid_switches):
    """
    Calculate drift using multiple methods for comparison:
    1. Original incremental method (adding 1 trial at a time)
    2. Non-overlapping windows method
    3. Fixed baseline comparison method
    All with both mean and sum aggregations
    """
    print("\n=== COMPREHENSIVE DRIFT ANALYSIS ===")
    
    drift_results = []
    
    for _, switch_info in valid_switches.iterrows():
        subject_id = switch_info['subject_id']
        switch_idx = switch_info['switch_idx']
        
        subject_data = df[df['subject_id'] == subject_id].sort_values(['date', 'trial_num']).reset_index(drop=True)
        
        # ====================================================================
        # METHOD 1: Original Incremental (expanding windows)
        # ====================================================================
        method1_coefficients = {}
        
        for post_extension in range(0, 16):  # t=0 to t=15
            start_idx = switch_idx - 50
            end_idx = switch_idx + post_extension + 1
            
            if end_idx <= len(subject_data) and start_idx >= 0:
                window_data = subject_data.iloc[start_idx:end_idx]
                coefs = fit_glm_for_drift(window_data)
                if coefs is not None:
                    method1_coefficients[post_extension] = coefs
        
        # Calculate Method 1 drift (consecutive differences)
        method1_drifts = []
        method1_contrast_drifts = []
        method1_prior_drifts = []
        method1_wsls_drifts = []
        
        for t in range(1, 16):
            if t-1 in method1_coefficients and t in method1_coefficients:
                prev = method1_coefficients[t-1]
                curr = method1_coefficients[t]
                
                prev_vec = np.array([prev['contrast'], prev['prior'], prev['wsls']])
                curr_vec = np.array([curr['contrast'], curr['prior'], curr['wsls']])
                
                drift = np.linalg.norm(curr_vec - prev_vec)
                method1_drifts.append(drift)
                method1_contrast_drifts.append(abs(curr['contrast'] - prev['contrast']))
                method1_prior_drifts.append(abs(curr['prior'] - prev['prior']))
                method1_wsls_drifts.append(abs(curr['wsls'] - prev['wsls']))
        
        method1_mean_drift = np.mean(method1_drifts) if method1_drifts else np.nan
        method1_total_drift = np.sum(method1_drifts) if method1_drifts else np.nan
        
        # ====================================================================
        # METHOD 2: Non-overlapping windows
        # ====================================================================
        windows_method2 = {
            'baseline': (switch_idx - 50, switch_idx),
            'post1': (switch_idx, switch_idx + 5),
            'post2': (switch_idx + 5, switch_idx + 10),
            'post3': (switch_idx + 10, switch_idx + 15)
        }
        
        method2_coefficients = {}
        for window_name, (start, end) in windows_method2.items():
            if start >= 0 and end <= len(subject_data):
                window_data = subject_data.iloc[start:end]
                coefs = fit_glm_for_drift(window_data)
                if coefs is not None:
                    method2_coefficients[window_name] = coefs
        
        # Calculate Method 2 drift (consecutive non-overlapping windows)
        method2_drifts = []
        method2_contrast_drifts = []
        method2_prior_drifts = []
        method2_wsls_drifts = []
        
        window_order = ['baseline', 'post1', 'post2', 'post3']
        for i in range(1, len(window_order)):
            prev_name = window_order[i-1]
            curr_name = window_order[i]
            
            if prev_name in method2_coefficients and curr_name in method2_coefficients:
                prev = method2_coefficients[prev_name]
                curr = method2_coefficients[curr_name]
                
                prev_vec = np.array([prev['contrast'], prev['prior'], prev['wsls']])
                curr_vec = np.array([curr['contrast'], curr['prior'], curr['wsls']])
                
                drift = np.linalg.norm(curr_vec - prev_vec)
                method2_drifts.append(drift)
                method2_contrast_drifts.append(abs(curr['contrast'] - prev['contrast']))
                method2_prior_drifts.append(abs(curr['prior'] - prev['prior']))
                method2_wsls_drifts.append(abs(curr['wsls'] - prev['wsls']))
        
        method2_mean_drift = np.mean(method2_drifts) if method2_drifts else np.nan
        method2_total_drift = np.sum(method2_drifts) if method2_drifts else np.nan
        
        # ====================================================================
        # METHOD 3: Fixed baseline comparison
        # ====================================================================
        if 'baseline' in method2_coefficients:
            baseline_vec = np.array([
                method2_coefficients['baseline']['contrast'],
                method2_coefficients['baseline']['prior'],
                method2_coefficients['baseline']['wsls']
            ])
            
            method3_drifts = []
            method3_contrast_drifts = []
            method3_prior_drifts = []
            method3_wsls_drifts = []
            
            for window_name in ['post1', 'post2', 'post3']:
                if window_name in method2_coefficients:
                    curr = method2_coefficients[window_name]
                    curr_vec = np.array([curr['contrast'], curr['prior'], curr['wsls']])
                    
                    drift = np.linalg.norm(curr_vec - baseline_vec)
                    method3_drifts.append(drift)
                    method3_contrast_drifts.append(abs(curr['contrast'] - method2_coefficients['baseline']['contrast']))
                    method3_prior_drifts.append(abs(curr['prior'] - method2_coefficients['baseline']['prior']))
                    method3_wsls_drifts.append(abs(curr['wsls'] - method2_coefficients['baseline']['wsls']))
            
            method3_mean_drift = np.mean(method3_drifts) if method3_drifts else np.nan
            method3_total_drift = np.sum(method3_drifts) if method3_drifts else np.nan
        else:
            method3_mean_drift = np.nan
            method3_total_drift = np.nan
            method3_contrast_drifts = []
            method3_prior_drifts = []
            method3_wsls_drifts = []
        
        # ====================================================================
        # Zero-contrast accuracy
        # ====================================================================
        zero_trials = subject_data[np.abs(subject_data['contrastLR']) < 0.01]
        if len(zero_trials) > 0 and 'feedbackType' in zero_trials.columns:
            zero_acc = (zero_trials['feedbackType'] == 1).mean()
        else:
            zero_acc = np.nan
        
        # ====================================================================
        # Store all results
        # ====================================================================
        result = {
            'switch_id': switch_info['switch_id'],
            'subject_id': subject_id,
            'group': switch_info['group'],
            
            # Method 1: Incremental
            'method1_mean_drift': method1_mean_drift,
            'method1_total_drift': method1_total_drift,
            'method1_mean_contrast_drift': np.mean(method1_contrast_drifts) if method1_contrast_drifts else np.nan,
            'method1_mean_prior_drift': np.mean(method1_prior_drifts) if method1_prior_drifts else np.nan,
            'method1_mean_wsls_drift': np.mean(method1_wsls_drifts) if method1_wsls_drifts else np.nan,
            'method1_total_contrast_drift': np.sum(method1_contrast_drifts) if method1_contrast_drifts else np.nan,
            'method1_total_prior_drift': np.sum(method1_prior_drifts) if method1_prior_drifts else np.nan,
            'method1_total_wsls_drift': np.sum(method1_wsls_drifts) if method1_wsls_drifts else np.nan,
            
            # Method 2: Non-overlapping
            'method2_mean_drift': method2_mean_drift,
            'method2_total_drift': method2_total_drift,
            'method2_mean_contrast_drift': np.mean(method2_contrast_drifts) if method2_contrast_drifts else np.nan,
            'method2_mean_prior_drift': np.mean(method2_prior_drifts) if method2_prior_drifts else np.nan,
            'method2_mean_wsls_drift': np.mean(method2_wsls_drifts) if method2_wsls_drifts else np.nan,
            'method2_total_contrast_drift': np.sum(method2_contrast_drifts) if method2_contrast_drifts else np.nan,
            'method2_total_prior_drift': np.sum(method2_prior_drifts) if method2_prior_drifts else np.nan,
            'method2_total_wsls_drift': np.sum(method2_wsls_drifts) if method2_wsls_drifts else np.nan,
            
            # Method 3: Fixed baseline
            'method3_mean_drift': method3_mean_drift,
            'method3_total_drift': method3_total_drift,
            'method3_mean_contrast_drift': np.mean(method3_contrast_drifts) if method3_contrast_drifts else np.nan,
            'method3_mean_prior_drift': np.mean(method3_prior_drifts) if method3_prior_drifts else np.nan,
            'method3_mean_wsls_drift': np.mean(method3_wsls_drifts) if method3_wsls_drifts else np.nan,
            'method3_total_contrast_drift': np.sum(method3_contrast_drifts) if method3_contrast_drifts else np.nan,
            'method3_total_prior_drift': np.sum(method3_prior_drifts) if method3_prior_drifts else np.nan,
            'method3_total_wsls_drift': np.sum(method3_wsls_drifts) if method3_wsls_drifts else np.nan,
            
            'zero_contrast_accuracy': zero_acc
        }
        
        drift_results.append(result)
    
    return pd.DataFrame(drift_results)

def calculate_drift_statistics_comprehensive(drift_df):
    """Calculate statistics for all drift methods"""
    print("\n=== COMPREHENSIVE DRIFT STATISTICS ===")
    
    stats = {}
    
    methods = ['method1', 'method2', 'method3']
    aggregations = ['mean', 'total']
    
    for method in methods:
        print(f"\n{method.upper()} RESULTS:")
        print("-" * 80)
        
        for agg in aggregations:
            col_name = f'{method}_{agg}_drift'
            
            if col_name not in drift_df.columns:
                continue
            
            # Overall correlation with zero-contrast accuracy
            valid = drift_df.dropna(subset=[col_name, 'zero_contrast_accuracy'])
            if len(valid) > 10:
                corr, pval = pearsonr(valid[col_name], valid['zero_contrast_accuracy'])
                stats[f'{method}_{agg}_overall_corr'] = corr
                stats[f'{method}_{agg}_overall_pval'] = pval
                print(f"\n{agg.upper()} drift vs zero-acc: r = {corr:.3f}, p = {pval:.4f}")
            
            # Group-specific stats
            for group in ['ASD', 'WT']:
                group_data = drift_df[drift_df['group'] == group]
                
                # Mean and SEM
                drift_vals = group_data[col_name].dropna()
                if len(drift_vals) > 0:
                    mean_drift = drift_vals.mean()
                    std_drift = drift_vals.std()
                    sem_drift = drift_vals.sem()
                    
                    stats[f'{method}_{agg}_{group}_mean'] = mean_drift
                    stats[f'{method}_{agg}_{group}_std'] = std_drift
                    stats[f'{method}_{agg}_{group}_sem'] = sem_drift
                    
                    print(f"{group} {agg} drift: {mean_drift:.2f} ± {sem_drift:.2f} (std={std_drift:.2f})")
                
                # Group correlation
                group_valid = group_data.dropna(subset=[col_name, 'zero_contrast_accuracy'])
                if len(group_valid) > 10:
                    corr, pval = pearsonr(group_valid[col_name], group_valid['zero_contrast_accuracy'])
                    stats[f'{method}_{agg}_{group}_corr'] = corr
                    stats[f'{method}_{agg}_{group}_pval'] = pval
            
            # Cohen's d
            asd_drift = drift_df[drift_df['group'] == 'ASD'][col_name].dropna()
            wt_drift = drift_df[drift_df['group'] == 'WT'][col_name].dropna()
            
            if len(asd_drift) > 0 and len(wt_drift) > 0:
                pooled_std = np.sqrt(((len(asd_drift) - 1) * asd_drift.std()**2 + 
                                      (len(wt_drift) - 1) * wt_drift.std()**2) / 
                                    (len(asd_drift) + len(wt_drift) - 2))
                cohens_d = (wt_drift.mean() - asd_drift.mean()) / pooled_std
                stats[f'{method}_{agg}_cohens_d'] = cohens_d
                
                stat, pval = mannwhitneyu(asd_drift, wt_drift)
                stats[f'{method}_{agg}_pval'] = pval
                
                print(f"Cohen's d: {cohens_d:.2f}, p = {pval:.4f}")
            
            # Coefficient-specific drifts
            for coef_type in ['contrast', 'prior', 'wsls']:
                coef_col = f'{method}_{agg}_{coef_type}_drift'
                
                if coef_col in drift_df.columns:
                    valid_coef = drift_df.dropna(subset=[coef_col, 'zero_contrast_accuracy'])
                    
                    if len(valid_coef) > 10:
                        corr, pval = pearsonr(valid_coef[coef_col], valid_coef['zero_contrast_accuracy'])
                        stats[f'{method}_{agg}_{coef_type}_corr'] = corr
                        stats[f'{method}_{agg}_{coef_type}_pval'] = pval
                    
                    # Group means
                    for group in ['ASD', 'WT']:
                        group_coef = drift_df[drift_df['group'] == group][coef_col].dropna()
                        if len(group_coef) > 0:
                            stats[f'{method}_{agg}_{coef_type}_{group}_mean'] = group_coef.mean()
                            stats[f'{method}_{agg}_{coef_type}_{group}_sem'] = group_coef.sem()
    
    # Print coefficient-specific correlations summary
    print("\n" + "="*80)
    print("COEFFICIENT-SPECIFIC CORRELATIONS (METHOD 2, MEAN):")
    print("-"*80)
    for coef_type in ['contrast', 'prior', 'wsls']:
        key = f'method2_mean_{coef_type}_corr'
        if key in stats:
            print(f"{coef_type} drift vs zero-acc: r = {stats[key]:.2f}")
    
    return stats

def plot_drift_comparison(drift_df, output_dir='D:/Neurofn'):
    """Create comprehensive comparison plots for all drift methods"""
    print("\n=== GENERATING DRIFT COMPARISON PLOTS ===")
    
    methods = [
        ('method1_mean_drift', 'Method 1: Incremental (Mean)', 'coral'),
        ('method1_total_drift', 'Method 1: Incremental (Sum)', 'darkred'),
        ('method2_mean_drift', 'Method 2: Non-overlapping (Mean)', 'steelblue'),
        ('method2_total_drift', 'Method 2: Non-overlapping (Sum)', 'darkblue'),
        ('method3_mean_drift', 'Method 3: Fixed Baseline (Mean)', 'green'),
        ('method3_total_drift', 'Method 3: Fixed Baseline (Sum)', 'darkgreen')
    ]
    
    # Figure 1: Method comparison - 6 panels
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    for idx, (col_name, title, color) in enumerate(methods):
        ax = axes[idx]
        
        if col_name not in drift_df.columns:
            continue
        
        valid = drift_df.dropna(subset=[col_name, 'zero_contrast_accuracy'])
        asd_data = valid[valid['group'] == 'ASD']
        wt_data = valid[valid['group'] == 'WT']
        
        # Scatter plot
        if len(asd_data) > 0:
            ax.scatter(asd_data[col_name], asd_data['zero_contrast_accuracy'], 
                      c='coral', alpha=0.5, s=20, label='ASD')
        if len(wt_data) > 0:
            ax.scatter(wt_data[col_name], wt_data['zero_contrast_accuracy'], 
                      c='steelblue', alpha=0.5, s=20, label='WT')
        
        # Overall correlation
        if len(valid) > 10:
            corr, pval = pearsonr(valid[col_name], valid['zero_contrast_accuracy'])
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Drift Distance', fontweight='bold')
        ax.set_ylabel('Zero-Contrast Accuracy', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drift_method_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/drift_method_comparison.pdf")
    
    # Figure 2: Group comparison for each method
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    method_pairs = [
        ('method1_mean_drift', 'Method 1: Mean'),
        ('method2_mean_drift', 'Method 2: Mean'),
        ('method3_mean_drift', 'Method 3: Mean'),
        ('method1_total_drift', 'Method 1: Sum'),
        ('method2_total_drift', 'Method 2: Sum'),
        ('method3_total_drift', 'Method 3: Sum')
    ]
    
    for idx, (col_name, title) in enumerate(method_pairs):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        if col_name not in drift_df.columns:
            continue
        
        asd_data = drift_df[drift_df['group'] == 'ASD'][col_name].dropna()
        wt_data = drift_df[drift_df['group'] == 'WT'][col_name].dropna()
        
        if len(asd_data) > 0 and len(wt_data) > 0:
            bp = ax.boxplot([asd_data, wt_data], labels=['ASD', 'WT'], patch_artist=True)
            bp['boxes'][0].set_facecolor('coral')
            bp['boxes'][1].set_facecolor('steelblue')
            
            asd_mean, wt_mean = asd_data.mean(), wt_data.mean()
            asd_sem, wt_sem = asd_data.sem(), wt_data.sem()
            
            stat, pval = mannwhitneyu(asd_data, wt_data)
            
            text_str = f"ASD: {asd_mean:.2f}±{asd_sem:.2f}\nWT: {wt_mean:.2f}±{wt_sem:.2f}\np={pval:.4f}"
            ax.text(0.5, 0.98, text_str, transform=ax.transAxes, 
                   ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylabel('Drift Distance', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drift_group_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/drift_group_comparison.pdf")
    
    # Figure 3: Coefficient-specific drifts (Method 2 only, as example)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    coef_types = ['contrast', 'prior', 'wsls']
    
    for agg_idx, agg in enumerate(['mean', 'total']):
        for coef_idx, coef_type in enumerate(coef_types):
            ax = axes[agg_idx, coef_idx]
            
            col_name = f'method2_{agg}_{coef_type}_drift'
            
            if col_name not in drift_df.columns:
                continue
            
            valid = drift_df.dropna(subset=[col_name, 'zero_contrast_accuracy'])
            
            if len(valid) > 0:
                # Scatter plot
                asd_data = valid[valid['group'] == 'ASD']
                wt_data = valid[valid['group'] == 'WT']
                
                if len(asd_data) > 0:
                    ax.scatter(asd_data[col_name], asd_data['zero_contrast_accuracy'], 
                              c='coral', alpha=0.5, s=20, label='ASD')
                if len(wt_data) > 0:
                    ax.scatter(wt_data[col_name], wt_data['zero_contrast_accuracy'], 
                              c='steelblue', alpha=0.5, s=20, label='WT')
                
                # Correlation
                if len(valid) > 10:
                    corr, pval = pearsonr(valid[col_name], valid['zero_contrast_accuracy'])
                    ax.text(0.05, 0.95, f'r={corr:.2f}\np={pval:.4f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel(f'Δβ^{coef_type} ({agg})', fontweight='bold')
                ax.set_ylabel('Zero-Contrast Accuracy', fontweight='bold')
                ax.set_title(f'{coef_type.capitalize()} Drift ({agg})', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drift_coefficient_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/drift_coefficient_comparison.pdf")
    
    # Figure 4: Distribution comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, (col_name, title) in enumerate(method_pairs):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        if col_name not in drift_df.columns:
            continue
        
        asd_data = drift_df[drift_df['group'] == 'ASD'][col_name].dropna()
        wt_data = drift_df[drift_df['group'] == 'WT'][col_name].dropna()
        
        if len(asd_data) > 0:
            ax.hist(asd_data, bins=30, alpha=0.6, color='coral', label='ASD', density=True)
        if len(wt_data) > 0:
            ax.hist(wt_data, bins=30, alpha=0.6, color='steelblue', label='WT', density=True)
        
        ax.set_xlabel('Drift Distance', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drift_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/drift_distributions.pdf")

def save_drift_comparison_results(drift_df, stats, output_dir='D:/Neurofn'):
    """Save comprehensive drift comparison statistics"""
    
    # Save full drift data
    drift_df.to_csv(f'{output_dir}/drift_comparison_full_data.csv', index=False)
    print(f"Saved: {output_dir}/drift_comparison_full_data.csv")
    
    # Save comparison statistics
    with open(f'{output_dir}/drift_comparison_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE DRIFT ANALYSIS - METHOD COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("THREE METHODS TESTED:\n")
        f.write("-"*80 + "\n")
        f.write("Method 1: Incremental (expanding windows, adding 1 trial at a time)\n")
        f.write("Method 2: Non-overlapping (4 windows: baseline, early, mid, late)\n")
        f.write("Method 3: Fixed baseline (all post-switch windows vs baseline)\n\n")
        
        f.write("TWO AGGREGATIONS:\n")
        f.write("-"*80 + "\n")
        f.write("Mean: Average drift across time points\n")
        f.write("Total: Sum of drift across time points\n\n")
        
        methods = ['method1', 'method2', 'method3']
        aggregations = ['mean', 'total']
        
        for method in methods:
            f.write("\n" + "="*80 + "\n")
            f.write(f"{method.upper()} RESULTS:\n")
            f.write("="*80 + "\n")
            
            for agg in aggregations:
                f.write(f"\n{agg.upper()} AGGREGATION:\n")
                f.write("-"*80 + "\n")
                
                # Overall correlation
                corr_key = f'{method}_{agg}_overall_corr'
                pval_key = f'{method}_{agg}_overall_pval'
                if corr_key in stats:
                    f.write(f"Overall drift vs zero-contrast accuracy:\n")
                    f.write(f"  r = {stats[corr_key]:.3f}\n")
                    f.write(f"  p = {stats[pval_key]:.4f}\n\n")
                
                # Group means
                f.write("Group mean drift distances:\n")
                for group in ['ASD', 'WT']:
                    mean_key = f'{method}_{agg}_{group}_mean'
                    sem_key = f'{method}_{agg}_{group}_sem'
                    if mean_key in stats:
                        f.write(f"  {group}: {stats[mean_key]:.2f} ± {stats[sem_key]:.2f}\n")
                
                # Cohen's d
                d_key = f'{method}_{agg}_cohens_d'
                p_key = f'{method}_{agg}_pval'
                if d_key in stats:
                    f.write(f"\nEffect size (Cohen's d): {stats[d_key]:.2f}\n")
                    f.write(f"Group comparison p-value: {stats[p_key]:.4f}\n")
                
                # Coefficient-specific
                f.write("\nCoefficient-specific drifts:\n")
                for coef_type in ['contrast', 'prior', 'wsls']:
                    f.write(f"\n  {coef_type.upper()}:\n")
                    
                    for group in ['ASD', 'WT']:
                        mean_key = f'{method}_{agg}_{coef_type}_{group}_mean'
                        sem_key = f'{method}_{agg}_{coef_type}_{group}_sem'
                        if mean_key in stats:
                            f.write(f"    {group}: {stats[mean_key]:.2f} ± {stats[sem_key]:.2f}\n")
                    
                    corr_key = f'{method}_{agg}_{coef_type}_corr'
                    pval_key = f'{method}_{agg}_{coef_type}_pval'
                    if corr_key in stats:
                        f.write(f"    Correlation with zero-acc: r = {stats[corr_key]:.2f}, p = {stats[pval_key]:.4f}\n")
        
        # Recommendation
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATION:\n")
        f.write("-"*80 + "\n")
        f.write("Based on the results:\n")
        f.write("- Method 2 (non-overlapping) or Method 3 (fixed baseline) likely provide\n")
        f.write("  more meaningful drift magnitudes than Method 1 (incremental)\n")
        f.write("- Total drift (sum) will give larger absolute values than mean drift\n")
        f.write("- Check which method produces drift values closest to manuscript expectations\n")
        f.write("  (ASD: ~0.83, WT: ~1.12 for mean drift)\n")
    
    print(f"Saved: {output_dir}/drift_comparison_statistics.txt")

def plot_main_results(windows_df, output_dir='D:/Neurofn'):
    """Plot main results (state transitions, evolution, correlation)"""
    
    # Figure 1: State transitions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('State Transition Matrices', fontsize=14, fontweight='bold')
    
    sorted_windows = windows_df.sort_values(['subject_id', 'window_center'])
    transition_matrices = {}
    
    for group_idx, group in enumerate(['ASD', 'WT']):
        group_windows = sorted_windows[sorted_windows['group'] == group]
        transitions = []
        
        for subject_id in group_windows['subject_id'].unique():
            subject_data = group_windows[group_windows['subject_id'] == subject_id]
            states = subject_data['cluster_id'].values
            for i in range(len(states) - 1):
                transitions.append({'from_state': states[i], 'to_state': states[i + 1]})
        
        if len(transitions) > 0:
            transition_df = pd.DataFrame(transitions)
            transition_matrix = pd.crosstab(transition_df['from_state'], transition_df['to_state'], normalize='index')
            transition_matrices[group] = transition_matrix
            
            sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues', ax=axes[group_idx])
            axes[group_idx].set_title(f'{group} Transitions')
            axes[group_idx].set_xlabel('To State')
            axes[group_idx].set_ylabel('From State')
    
    # Difference matrix
    if 'ASD' in transition_matrices and 'WT' in transition_matrices:
        all_states = list(set(transition_matrices['ASD'].index) | set(transition_matrices['WT'].index) |
                         set(transition_matrices['ASD'].columns) | set(transition_matrices['WT'].columns))
        
        asd_aligned = transition_matrices['ASD'].reindex(index=all_states, columns=all_states, fill_value=0)
        wt_aligned = transition_matrices['WT'].reindex(index=all_states, columns=all_states, fill_value=0)
        diff_matrix = asd_aligned - wt_aligned
        
        sns.heatmap(diff_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[2])
        axes[2].set_title('Difference (ASD - WT)')
        axes[2].set_xlabel('To State')
        axes[2].set_ylabel('From State')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/state_transitions.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/state_transitions.pdf")
    
    # Figure 2: Quantile evolution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Strategy evolution curves
    colors_high = ['#8B0000', '#DC143C', '#FF6347', '#FFA07A']
    colors_low = ['#00008B', '#0000CD', '#4169E1', '#87CEEB']
    cluster_colors = colors_high + colors_low
    
    for group_idx, group in enumerate(['ASD', 'WT']):
        ax = axes[0, group_idx]
        group_data = windows_df[windows_df['group'] == group]
        
        cluster_props = {cid: [] for cid in range(1, 9)}
        
        for quantile in range(20):
            quantile_data = group_data[group_data['quantile'] == quantile]
            
            if len(quantile_data) > 0:
                for cid in range(1, 9):
                    prop = len(quantile_data[quantile_data['cluster_id'] == cid]) / len(quantile_data) * 100
                    cluster_props[cid].append(prop)
            else:
                for cid in range(1, 9):
                    cluster_props[cid].append(0)
        
        # Add endpoint
        for cid in range(1, 9):
            cluster_props[cid].append(cluster_props[cid][-1])
        
        x_pos = np.arange(21)
        for cid in range(1, 9):
            label = f'C{cid}'
            if cid <= 4:
                label += ' (HP)'
            else:
                label += ' (LP)'
            ax.plot(x_pos, cluster_props[cid], marker='o', markersize=3, linewidth=2, 
                   color=cluster_colors[cid-1], alpha=0.8, label=label)
        
        ax.set_xlabel('Session Progress (%)', fontsize=11)
        ax.set_ylabel('Strategy Proportion (%)', fontsize=11)
        ax.set_title(f'{group}: Strategy Evolution', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos[::4])
        ax.set_xticklabels([f'{i*5}%' for i in range(0, 21, 4)])
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 35])
    
    # Zero-contrast accuracy over time
    ax = axes[1, 0]
    for group, color in [('ASD', 'coral'), ('WT', 'steelblue')]:
        group_data = windows_df[windows_df['group'] == group]
        
        means, sems = [], []
        for quantile in range(20):
            qdata = group_data[group_data['quantile'] == quantile]
            acc = qdata['accuracy_zero_contrast'].dropna()
            means.append(acc.mean() if len(acc) > 0 else np.nan)
            sems.append(acc.sem() if len(acc) > 0 else np.nan)
        
        means.append(means[-1])
        sems.append(sems[-1])
        
        x_pos = np.arange(21)
        ax.plot(x_pos, means, 'o-', label=group, color=color, linewidth=2.5, markersize=5)
        
        valid = ~np.isnan(means)
        ax.fill_between(x_pos[valid], 
                       np.array(means)[valid] - np.array(sems)[valid],
                       np.array(means)[valid] + np.array(sems)[valid],
                       alpha=0.3, color=color)
    
    ax.set_xlabel('Session Progress (%)', fontsize=11)
    ax.set_ylabel('Zero-Contrast Accuracy', fontsize=11)
    ax.set_title('Zero-Contrast Performance Evolution', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos[::4])
    ax.set_xticklabels([f'{i*5}%' for i in range(0, 21, 4)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim([0.4, 1.0])
    
    # High-prior proportion over time
    ax = axes[1, 1]
    for group, color in [('ASD', 'coral'), ('WT', 'steelblue')]:
        group_data = windows_df[windows_df['group'] == group]
        
        props = []
        for quantile in range(20):
            qdata = group_data[group_data['quantile'] == quantile]
            if len(qdata) > 0:
                prop = qdata['is_high_prior'].mean() * 100
                props.append(prop)
            else:
                props.append(np.nan)
        
        props.append(props[-1])
        
        x_pos = np.arange(21)
        ax.plot(x_pos, props, 'o-', label=group, color=color, linewidth=2.5, markersize=5)
    
    ax.set_xlabel('Session Progress (%)', fontsize=11)
    ax.set_ylabel('High-Prior Strategy Usage (%)', fontsize=11)
    ax.set_title('High-Prior Strategy Evolution', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos[::4])
    ax.set_xticklabels([f'{i*5}%' for i in range(0, 21, 4)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/strategy_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/strategy_evolution.pdf")
    
    # Figure 3: Correlation analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # High-prior usage vs zero-contrast accuracy
    ax = axes[0]
    for group, color in [('ASD', 'coral'), ('WT', 'steelblue')]:
        group_data = windows_df[windows_df['group'] == group]
        
        props, accs = [], []
        for quantile in range(20):
            qdata = group_data[group_data['quantile'] == quantile]
            if len(qdata) > 0:
                prop = qdata['is_high_prior'].mean() * 100
                acc_data = qdata['accuracy_zero_contrast'].dropna()
                if len(acc_data) > 0:
                    props.append(prop)
                    accs.append(acc_data.mean())
        
        if len(props) > 0:
            ax.scatter(props, accs, s=100, alpha=0.7, color=color, label=group, edgecolors='black', linewidth=1.5)
            
            if len(props) > 2:
                z = np.polyfit(props, accs, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(props), max(props), 100)
                ax.plot(x_trend, p(x_trend), '--', color=color, linewidth=2, alpha=0.8)
                
                corr, p_val = pearsonr(props, accs)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                y_pos = 0.95 if group == 'ASD' else 0.85
                ax.text(0.05, y_pos, f'{group}: r={corr:.3f} {sig}',
                       transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax.set_xlabel('High-Prior Strategy Usage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Zero-Contrast Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Strategy Usage vs Performance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    # State occupancy comparison
    ax = axes[1]
    asd_data = windows_df[windows_df['group'] == 'ASD']
    wt_data = windows_df[windows_df['group'] == 'WT']
    
    cluster_ids = range(1, 9)
    asd_props = [len(asd_data[asd_data['cluster_id'] == cid]) / len(asd_data) * 100 for cid in cluster_ids]
    wt_props = [len(wt_data[wt_data['cluster_id'] == cid]) / len(wt_data) * 100 for cid in cluster_ids]
    
    x = np.arange(len(cluster_ids))
    width = 0.35
    
    ax.bar(x - width/2, asd_props, width, label='ASD', color='coral', alpha=0.7)
    ax.bar(x + width/2, wt_props, width, label='WT', color='steelblue', alpha=0.7)
    
    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Occupancy (%)', fontsize=12, fontweight='bold')
    ax.set_title('State Occupancy by Group', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in cluster_ids])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=12.5, color='gray', linestyle=':', alpha=0.5, label='Expected (12.5%)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/correlation_analysis.pdf")

def save_numerical_results(windows_df, all_stats, output_dir='D:/Neurofn'):
    """Save all numerical results"""
    
    # Save windows data
    windows_df.to_csv(f'{output_dir}/all_windows_data.csv', index=False)
    print(f"Saved: {output_dir}/all_windows_data.csv")
    
    # Save comprehensive statistics
    with open(f'{output_dir}/key_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("KEY STATISTICS FOR MANUSCRIPT\n")
        f.write("="*80 + "\n\n")
        
        # PCA and clustering
        f.write("CLUSTERING VALIDATION:\n")
        f.write("-"*80 + "\n")
        f.write(f"PCA variance explained (first 2 components): {all_stats['clustering']['pca_variance_2pc']:.1f}%\n")
        f.write(f"Expected under independence: 66.7%\n\n")
        
        f.write("Median coefficient values:\n")
        f.write(f"  β^contrast_med = {all_stats['clustering']['contrast_median']:.2f}\n")
        f.write(f"  β^prior_med = {all_stats['clustering']['prior_median']:.2f}\n")
        f.write(f"  β^WSLS_med = {all_stats['clustering']['wsls_median']:.2f}\n\n")
        
        f.write("Empirical cluster occupancies (expected 12.5% under independence):\n")
        for i in range(1, 9):
            pct = all_stats['clustering']['cluster_occupancies'][f'C{i}']
            f.write(f"  C{i}: {pct:.1f}%\n")
        
        # Group statistics
        f.write("\n" + "="*80 + "\n")
        f.write("GROUP COMPARISONS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Optimal state (C1+C2) occupancy:\n")
        f.write(f"  WT: {all_stats['group']['WT_optimal_occupancy']:.0f}%\n")
        f.write(f"  ASD: {all_stats['group']['ASD_optimal_occupancy']:.0f}%\n\n")
        
        f.write(f"Low-prior state occupancy:\n")
        f.write(f"  C6: ASD {all_stats['group']['C6_ASD']:.1f}% vs WT {all_stats['group']['C6_WT']:.0f}%\n")
        f.write(f"  C8: ASD {all_stats['group']['C8_ASD']:.1f}% vs WT {all_stats['group']['C8_WT']:.1f}%\n\n")
        
        f.write(f"Self-transition probabilities:\n")
        f.write(f"  ASD mean: {all_stats['group']['ASD_self_transition']:.2f}\n")
        f.write(f"  WT mean: {all_stats['group']['WT_self_transition']:.2f}\n\n")
        
        f.write(f"C8 self-transition:\n")
        f.write(f"  ASD: {all_stats['group']['ASD_C8_self_transition']:.2f}\n")
        f.write(f"  WT: {all_stats['group']['WT_C8_self_transition']:.2f}\n\n")
        
        f.write(f"C1 to suboptimal (C3,C4,C7,C8) transition probability:\n")
        f.write(f"  ASD: {all_stats['group']['ASD_C1_to_suboptimal']:.2f}\n")
        f.write(f"  WT: {all_stats['group']['WT_C1_to_suboptimal']:.2f}\n")
        
        # Temporal evolution
        f.write("\n" + "="*80 + "\n")
        f.write("TEMPORAL EVOLUTION:\n")
        f.write("-"*80 + "\n")
        f.write("Quantile 1 (0-5%):\n")
        f.write(f"  ASD: High-prior {all_stats['temporal']['ASD_high_prior_q1']:.0f}%, Zero-acc {all_stats['temporal']['ASD_zero_acc_q1']:.2f}\n")
        f.write(f"  WT: High-prior {all_stats['temporal']['WT_high_prior_q1']:.0f}%, Zero-acc {all_stats['temporal']['WT_zero_acc_q1']:.2f}\n\n")
        
        f.write("Quantile 20 (95-100%):\n")
        f.write(f"  ASD: High-prior {all_stats['temporal']['ASD_high_prior_q20']:.0f}%, Zero-acc {all_stats['temporal']['ASD_zero_acc_q20']:.2f}\n")
        f.write(f"  WT: High-prior {all_stats['temporal']['WT_high_prior_q20']:.0f}%, Zero-acc {all_stats['temporal']['WT_zero_acc_q20']:.2f}\n\n")
        
        f.write("High-prior vs accuracy correlation:\n")
        f.write(f"  ASD: r = {all_stats['temporal']['ASD_hp_acc_correlation']:.2f}, p = {all_stats['temporal']['ASD_hp_acc_pval']:.3f}\n")
        f.write(f"  WT: r = {all_stats['temporal']['WT_hp_acc_correlation']:.2f}, p = {all_stats['temporal']['WT_hp_acc_pval']:.3f}\n")
    
    print(f"Saved: {output_dir}/key_statistics.txt")
    
    return all_stats

def run_full_analysis_part1(df1, df2, proportion=1.0, output_dir='D:/Neurofn'):
    """Run complete analysis with comprehensive drift comparison"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BEHAVIORAL ANALYSIS - PART 1")
    print("STEP SIZE: 20 trials")
    print("="*80 + "\n")
    
    # Prepare data
    df1, df2 = df1.copy(), df2.copy()
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
    
    # Subsample if needed
    if proportion < 1.0:
        subjects = df_combined['subject_id'].unique()
        n_select = max(1, int(len(subjects) * proportion))
        np.random.seed(42)
        selected = np.random.choice(subjects, size=n_select, replace=False)
        df_combined = df_combined[df_combined['subject_id'].isin(selected)]
    
    print(f"Dataset: {len(df_combined)} trials")
    print(f"ASD subjects: {df_combined[df_combined['group']=='ASD']['subject_id'].nunique()}")
    print(f"WT subjects: {df_combined[df_combined['group']=='WT']['subject_id'].nunique()}\n")
    
    # Create features
    df_features = create_enhanced_features(df_combined)
    
    # Sliding window GLM (step_size=20)
    print("Running sliding window GLM with step_size=20...")
    windows_df = sliding_window_glm(df_features, df_combined, window_size=100, step_size=20)
    
    if len(windows_df) == 0:
        print("ERROR: No windows generated!")
        return None
    
    print(f"Generated {len(windows_df)} windows")
    print(f"  ASD: {len(windows_df[windows_df['group']=='ASD'])}")
    print(f"  WT: {len(windows_df[windows_df['group']=='WT'])}")
    
    # Clustering with statistics
    windows_df, clustering_stats = three_factor_clustering(windows_df)
    
    # Assign quantiles
    windows_df = assign_quantiles(windows_df, n_quantiles=20)
    
    # Calculate group statistics
    group_stats = calculate_group_statistics(windows_df)
    
    # Calculate temporal evolution
    temporal_stats = calculate_temporal_evolution(windows_df)
    
    # Combine all statistics (without drift for now)
    all_stats = {
        'clustering': clustering_stats,
        'group': group_stats,
        'temporal': temporal_stats
    }
    
    # Drift analysis with method comparison
    valid_switches = detect_valid_block_switches(df_combined)
    print(f"\nDetected {len(valid_switches)} valid block switches")
    
    drift_df = pd.DataFrame()
    drift_stats = {}
    
    if len(valid_switches) > 0:
        # Use comprehensive drift calculation
        drift_df = calculate_drift_metrics_comprehensive(df_combined, valid_switches)
        print(f"Analyzed {len(drift_df)} switches for drift")
        
        # Calculate statistics for all methods
        drift_stats = calculate_drift_statistics_comprehensive(drift_df)
        
        # Save and plot comparisons
        save_drift_comparison_results(drift_df, drift_stats, output_dir)
        plot_drift_comparison(drift_df, output_dir)
    
    # Save numerical results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    all_stats = save_numerical_results(windows_df, all_stats, output_dir)
    
    # Plot main results
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    plot_main_results(windows_df, output_dir)
    
    print("\n" + "="*80)
    print("PART 1 COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    
    # Display key statistics in Jupyter
    print("\n" + "="*80)
    print("KEY STATISTICS SUMMARY")
    print("="*80)
    print(f"\nPCA variance (first 2 PC): {all_stats['clustering']['pca_variance_2pc']:.1f}%")
    print(f"Median coefficients: contrast={all_stats['clustering']['contrast_median']:.2f}, "
          f"prior={all_stats['clustering']['prior_median']:.2f}, "
          f"WSLS={all_stats['clustering']['wsls_median']:.2f}")
    print(f"\nOptimal occupancy: WT {all_stats['group']['WT_optimal_occupancy']:.0f}%, "
          f"ASD {all_stats['group']['ASD_optimal_occupancy']:.0f}%")
    print(f"Correlations (high-prior vs accuracy): "
          f"ASD r={all_stats['temporal']['ASD_hp_acc_correlation']:.2f} (p={all_stats['temporal']['ASD_hp_acc_pval']:.3f}), "
          f"WT r={all_stats['temporal']['WT_hp_acc_correlation']:.2f} (p={all_stats['temporal']['WT_hp_acc_pval']:.3f})")
    
    if len(drift_stats) > 0:
        print(f"\nDrift Analysis (showing Method 2 - Non-overlapping):")
        if 'method2_mean_overall_corr' in drift_stats:
            print(f"  Overall drift-accuracy correlation: r={drift_stats['method2_mean_overall_corr']:.3f}, "
                  f"p={drift_stats['method2_mean_overall_pval']:.4f}")
        if 'method2_mean_ASD_mean' in drift_stats:
            print(f"  Mean drift: ASD {drift_stats['method2_mean_ASD_mean']:.2f}±{drift_stats['method2_mean_ASD_sem']:.2f}, "
                  f"WT {drift_stats['method2_mean_WT_mean']:.2f}±{drift_stats['method2_mean_WT_sem']:.2f}")
        if 'method2_mean_cohens_d' in drift_stats:
            print(f"  Cohen's d: {drift_stats['method2_mean_cohens_d']:.2f}")
    
    return {
        'windows_df': windows_df,
        'drift_df': drift_df,
        'df_features': df_features,
        'statistics': all_stats,
        'drift_statistics': drift_stats
    }

# ============================================================================
# EXECUTE PART 1
# ============================================================================

if __name__ == "__main__":
    # Load data
    df1 = pd.read_csv("D:\\hms\\82ASD_P2_BlockAdded.csv")
    df2 = pd.read_csv("D:\\hms\\78BWM_P2_BlockAdded.csv")
    
    # Run analysis
    results = run_full_analysis_part1(df1, df2, proportion=1, output_dir='D:/Neurofn')
    
    # Access statistics
    if results is not None:
        stats = results['statistics']
        drift_stats = results['drift_statistics']
        print("\n" + "="*80)
        print("STATISTICS DICTIONARY STRUCTURE")
        print("="*80)
        print("\nAccess statistics via:")
        print("  - results['statistics']['clustering']")
        print("  - results['statistics']['group']") 
        print("  - results['statistics']['temporal']")
        print("  - results['drift_statistics'] - all drift methods")
