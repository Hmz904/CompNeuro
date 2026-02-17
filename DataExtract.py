from one.api import ONE
import numpy as np
import scipy.signal as signal
from brainbox.io.one import SpikeSortingLoader
import pandas as pd
from datetime import datetime
import gc

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          username='intbrainlab',
          password='international')
pids = one.search_insertions(atlas_acronym=['VIS'], query_type='remote')


# one = ONE(base_url='https://alyx.internationalbrainlab.org')

# insertions = one.alyx.rest('insertions', 'list', django=f'session__projects__name__icontains,{project}')

# pid_list = [ins['id'] for ins in insertions]
# print(f'Found {len(pid_list)} PIDs: {pid_list}')

# Get subject info
# subject_info = [ins['session_info']['subject'] for ins in insertions]
# print(f'Subject info: {subject_info}')

# all_pids = pid_list
# print(f'Total unique PIDs found: {len(all_pids)}')

# Analyze all pids (or limit for testing)
pids = all_pids[:]

# pids = pids[:5]
print(f"Analyzing {len(pids)} PIDs")

freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 80)
}

def compute_band_power(data, fs, freq_range):
    if data.shape[1] < fs * 0.2:
        return np.nan
    
    try:
        nperseg = min(int(fs * 0.5), data.shape[1])
        freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=1)
        freq_idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
        
        if not np.any(freq_idx):
            return np.nan
        
        band_power = np.mean(psd[:, freq_idx])
        return np.log10(band_power + 1e-12)
        
    except Exception as e:
        print(f"      Error in compute_band_power: {str(e)}")
        return np.nan

def extract_lfp_window(sr_lf, time_center, window, fs_lf, vis_channel_indices):
    try:
        t_start = time_center + window[0]
        t_end = time_center + window[1]
        
        s_start = int(t_start * fs_lf)
        s_end = int(t_end * fs_lf)
        
        if s_start < 0 or s_end > sr_lf.ns or s_start >= s_end:
            return None
        
        lfp_data = sr_lf[s_start:s_end, :-sr_lf.nsync].T
        
        if lfp_data.shape[1] == 0:
            return None
        
        lfp_data_vis = lfp_data[vis_channel_indices, :]
        
        if lfp_data_vis.shape[0] == 0:
            return None
            
        return lfp_data_vis
        
    except Exception as e:
        print(f"      Error extracting LFP window: {str(e)}")
        return None

all_results = []

for idx, pid in enumerate(pids):
    print(f"\nProcessing PID {idx+1}/{len(pids)}: {pid}")
    
    sr_lf = None
    
    try:
        sl = SpikeSortingLoader(pid=pid, one=one)
        spikes, clusters, channels = sl.load_spike_sorting()
        
        acronyms = channels["acronym"]
        vis_mask = np.array([str(acr).startswith('VIS') if acr is not None else False for acr in acronyms])
        vis_channel_indices = np.where(vis_mask)[0]
        
        print(f"  Found {len(vis_channel_indices)} VIS channels")
        
        if len(vis_channel_indices) == 0:
            print(f"  No VIS channels found, skipping...")
            continue
        
        eid = one.pid2eid(pid)
        if not eid:
            print(f"  No EID found for PID {pid}, skipping...")
            continue
            
        session_info = one.get_details(eid[0])
        subject_name = session_info['subject']
        session_date = session_info['date'].strftime('%Y-%m-%d') if hasattr(session_info['date'], 'strftime') else str(session_info['date'])
        
        print(f"  Subject: {subject_name}, Date: {session_date}")
            
        cam = one.load_object(eid[0], 'leftCamera', collection='alf')
        trials = one.load_object(eid[0], 'trials', collection='alf')
        
        n_trials = len(trials['stimOn_times'])
        print(f"  Found {n_trials} trials")
        
        print(f"  Loading LFP data...")
        try:
            sr_lf = sl.raw_electrophysiology(band="lf", stream=True)
            fs_lf = sr_lf.fs
            print(f"  LFP sampling rate: {fs_lf} Hz")
        except Exception as e:
            print(f"  Warning: Could not load LFP data: {str(e)}")
            sr_lf = None
        
        result_matrix = np.full((n_trials, 27), np.nan, dtype=object)
        
        stim_on_times = trials['stimOn_times']
        response_times = trials['response_times']
        feedback_times = trials['feedback_times']
        
        print(f"  Computing spike counts...")
        for i, stim_time in enumerate(stim_on_times):
            if not np.isnan(stim_time):
                spike_counts = np.sum((spikes['times'] >= stim_time - 0.1) & 
                                     (spikes['times'] <= stim_time + 0.1))
                result_matrix[i, 0] = spike_counts
        
        print(f"  Computing ROI motion energy...")
        for i, stim_time in enumerate(stim_on_times):
            if not np.isnan(stim_time):
                roi_values = cam['ROIMotionEnergy'][(cam['times'] >= stim_time - 0.1) & 
                                                     (cam['times'] <= stim_time + 0.1)]
                result_matrix[i, 1] = np.mean(roi_values) if len(roi_values) > 0 else np.nan
        
        print(f"  Computing pupil diameter...")
        for i, stim_time in enumerate(stim_on_times):
            if not np.isnan(stim_time):
                try:
                    pupil_diameter = (((cam['dlc']['pupil_top_r_x'] - cam['dlc']['pupil_bottom_r_x'])**2 + 
                                      (cam['dlc']['pupil_top_r_y'] - cam['dlc']['pupil_bottom_r_y'])**2)**0.5) + \
                                    (((cam['dlc']['pupil_left_r_x'] - cam['dlc']['pupil_right_r_x'])**2 + 
                                      (cam['dlc']['pupil_left_r_y'] - cam['dlc']['pupil_right_r_y'])**2)**0.5)
                    
                    diameter_values = pupil_diameter[(cam['times'] >= stim_time - 0.1) & 
                                                     (cam['times'] <= stim_time + 0.1)]
                    result_matrix[i, 2] = np.mean(diameter_values) if len(diameter_values) > 0 else np.nan
                except Exception as e:
                    result_matrix[i, 2] = np.nan
        
        result_matrix[:, 3] = trials['response_times'][:n_trials] - trials['stimOn_times'][:n_trials]
        result_matrix[:, 4] = np.nan_to_num(trials['contrastRight'], nan=0) - np.nan_to_num(trials['contrastLeft'], nan=0)
        result_matrix[:, 5] = trials["probabilityLeft"][:n_trials]
        result_matrix[:, 6] = trials['feedbackType'][:n_trials]
        result_matrix[:, 7] = trials['choice'][:n_trials]
        
        result_matrix[:, 8] = str(eid[0])
        result_matrix[:, 9] = str(pid)
        result_matrix[:, 10] = subject_name
        result_matrix[:, 11] = session_date
        
        if sr_lf is not None:
            print(f"  Computing LFP band powers for VIS channels...")
            successful_trials = 0
            
            for i in range(n_trials):
                if i % 50 == 0:
                    print(f"    Processing trial {i+1}/{n_trials}")
                
                stim_time = stim_on_times[i]
                resp_time = response_times[i]
                fb_time = feedback_times[i]
                
                if np.isnan(stim_time) or np.isnan(resp_time) or np.isnan(fb_time):
                    continue
                
                lfp_baseline = extract_lfp_window(sr_lf, stim_time, [-1, 0], fs_lf, vis_channel_indices)
                
                decision_duration = resp_time - stim_time
                if decision_duration > 0.05 and decision_duration < 5:
                    lfp_decision = extract_lfp_window(sr_lf, stim_time, [0, decision_duration], fs_lf, vis_channel_indices)
                else:
                    lfp_decision = None
                
                lfp_feedback = extract_lfp_window(sr_lf, fb_time, [0, 1], fs_lf, vis_channel_indices)
                
                col_idx = 12
                
                for band_name, freq_range in freq_bands.items():
                    if lfp_baseline is not None:
                        power = compute_band_power(lfp_baseline, fs_lf, freq_range)
                        result_matrix[i, col_idx] = power
                        if not np.isnan(power):
                            successful_trials += 1
                    col_idx += 1
                
                for band_name, freq_range in freq_bands.items():
                    if lfp_decision is not None:
                        power = compute_band_power(lfp_decision, fs_lf, freq_range)
                        result_matrix[i, col_idx] = power
                    col_idx += 1
                
                for band_name, freq_range in freq_bands.items():
                    if lfp_feedback is not None:
                        power = compute_band_power(lfp_feedback, fs_lf, freq_range)
                        result_matrix[i, col_idx] = power
                    col_idx += 1
            
            print(f"  Successfully computed LFP features for {successful_trials//5} trial-windows")
        else:
            print(f"  Skipping LFP computation (data not available)")
        
        all_results.append(result_matrix)
        
        print(f"  Successfully processed {n_trials} trials")
        
    except Exception as e:
        print(f"  Error processing PID {pid}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue
    
    finally:
        try:
            del spikes, clusters, channels, cam, trials, sl
            if sr_lf is not None:
                del sr_lf
        except:
            pass
        gc.collect()
        print(f"  Memory cleaned for PID {pid}")

if all_results:
    final_result = np.vstack(all_results)
    print(f"\nFinal result shape: {final_result.shape}")
    
    columns = ['spike_counts', 'roi_motion_energy', 'pupil_diameter', 'reaction_time', 
               'contrast_diff', 'prob_left', 'feedback_type', 'choice', 'eid', 'pid', 
               'subject_name', 'session_date',
               'delta_baseline', 'theta_baseline', 'alpha_baseline', 'beta_baseline', 'gamma_baseline',
               'delta_decision', 'theta_decision', 'alpha_decision', 'beta_decision', 'gamma_decision',
               'delta_feedback', 'theta_feedback', 'alpha_feedback', 'beta_feedback', 'gamma_feedback']
    
    df = pd.DataFrame(final_result, columns=columns)
    
    numeric_cols = ['spike_counts', 'roi_motion_energy', 'pupil_diameter', 'reaction_time', 
                   'contrast_diff', 'prob_left', 'feedback_type', 'choice',
                   'delta_baseline', 'theta_baseline', 'alpha_baseline', 'beta_baseline', 'gamma_baseline',
                   'delta_decision', 'theta_decision', 'alpha_decision', 'beta_decision', 'gamma_decision',
                   'delta_feedback', 'theta_feedback', 'alpha_feedback', 'beta_feedback', 'gamma_feedback']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    output_filename = f'VIS_WT_LFP_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nData saved to: {output_filename}")
    
    print(f"\nData summary:")
    print(f"Total trials processed: {df.shape[0]}")
    print(f"Successfully processed PIDs: {len(all_results)}")
    print(f"Average trials per PID: {df.shape[0] / len(all_results):.1f}")
    print(f"Unique subjects: {df['subject_name'].nunique()}")
    print(f"Subject names: {df['subject_name'].unique().tolist()}")
    
    lfp_cols = [col for col in columns if any(band in col for band in ['delta', 'theta', 'alpha', 'beta', 'gamma'])]
    print(f"\nLFP data availability:")
    for col in lfp_cols:
        valid_count = df[col].notna().sum()
        print(f"  {col}: {valid_count}/{len(df)} ({100*valid_count/len(df):.1f}%)")
    
    print(f"\nLFP feature statistics (log10 power):")
    print(df[lfp_cols].describe())
    
    print(f"\nFirst 5 rows of data:")
    print(df.head())
    
else:
    print("No data was successfully processed")
    final_result = None
    
del all_results
gc.collect()
print("\nProcessing complete!")
