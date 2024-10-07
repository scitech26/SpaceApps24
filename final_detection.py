import os
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import trigger_onset
from obspy.signal.filter import bandpass
import numpy as np
from scipy.signal import medfilt


MSEED_FOLDER = "C:/Users/Efe_/Desktop/mseed_denoised3"  
OUTPUT_FOLDER = "C:/Users/Efe_/Desktop/s12gradeBfinal"  


FREQMIN = 0.6  
FREQMAX = 1.0  
FILTER_ORDER = 6  


HAMPel_WINDOW_SIZE = 11  
HAMPel_N_SIGMA = 3.0


MOVING_Z_WINDOW_SIZE = 101  
MOVING_Z_THRESHOLD = 3.0


STA_WINDOW_PRIMARY = 0.5       
LTA_WINDOW_PRIMARY = 10.0     


STA_WINDOW_SECONDARY = 0.5       
LTA_WINDOW_SECONDARY = 10.0     

LTA_OFFSET = 0.0000000005     


TRIGGER_ON = 1.0                    
TRIGGER_OFF = 0.5                   


VALIDATION_FACTOR = 0.5             


BIG_EVENT_THRESHOLD = 10          
BIG_EVENT_WINDOW_SECONDS = 2000  


os.makedirs(OUTPUT_FOLDER, exist_ok=True)



def compute_sta_lta(data, sampling_rate, sta_window, lta_window):
    """
    Compute STA and LTA ratios based on absolute seismic data.
    
    Parameters:
    - data: numpy array of seismic velocity data
    - sampling_rate: Sampling rate in Hz
    - sta_window: Short-term average window in seconds
    - lta_window: Long-term average window in seconds
    
    Returns:
    - sta: numpy array of STA values
    - lta: numpy array of LTA values
    """
    nsta = int(sta_window * sampling_rate)
    nlta = int(lta_window * sampling_rate)
    nsta = max(nsta, 1)
    nlta = max(nlta, 1)
    abs_data = np.abs(data)
    sta = pd.Series(abs_data).rolling(window=nsta, min_periods=1).mean().values
    lta = pd.Series(abs_data).rolling(window=nlta, min_periods=1).mean().values + LTA_OFFSET
    return sta, lta

def hampel_filter(data, window_size=11, n_sigma=3.0):
    """
    Apply Hampel filter to remove spikes from data.
    
    Parameters:
    - data: numpy array of seismic velocity data
    - window_size: Size of the moving window (must be odd)
    - n_sigma: Number of standard deviations to use as the threshold
    
    Returns:
    - filtered_data: numpy array with spikes replaced by the median
    """
    if window_size % 2 == 0:
        window_size += 1  
    median = pd.Series(data).rolling(window=window_size, center=True).median()
    mad = 1.4826 * pd.Series(data).rolling(window=window_size, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    difference = np.abs(data - median)
    threshold = n_sigma * mad
    outliers = difference > threshold
    filtered_data = data.copy()
    filtered_data[outliers] = median[outliers]
    
   
    nan_mask = np.isnan(filtered_data)
    filtered_data[nan_mask] = data[nan_mask]
    
    return filtered_data

def moving_z_score_filter(data, window_size=101, z_thresh=3.0):
    """
    Apply moving z-score filter to identify and remove spikes.
    
    Parameters:
    - data: numpy array of seismic velocity data
    - window_size: Size of the moving window
    - z_thresh: Z-score threshold to identify spikes
    
    Returns:
    - filtered_data: numpy array with spikes replaced by the rolling mean
    """
    rolling_mean = pd.Series(data).rolling(window=window_size, center=True).mean()
    rolling_std = pd.Series(data).rolling(window=window_size, center=True).std()
    z_scores = (data - rolling_mean) / rolling_std
    outliers = np.abs(z_scores) > z_thresh
    filtered_data = data.copy()
    filtered_data[outliers] = rolling_mean[outliers]
    
    
    nan_mask = np.isnan(filtered_data)
    filtered_data[nan_mask] = data[nan_mask]
    
    return filtered_data

def identify_big_events(validated_triggers, times, threshold=10, window=2000):
    """
    Identify Big Events based on the number of validated triggers within a time window.
    
    Parameters:
    - validated_triggers: List of validated trigger tuples (start_idx, end_idx)
    - times: Numpy array of time values in seconds
    - threshold: Number of events to qualify as a Big Event
    - window: Time window in seconds to consider for Big Events
    
    Returns:
    - big_events: List of tuples indicating the start and end times of Big Events
    """
   
    event_times = [times[trigger[0]] for trigger in validated_triggers]
    event_times_sorted = sorted(event_times)

    big_events = []
    num_events = len(event_times_sorted)

    for start in range(num_events):
       
        end = start
        
        while end < num_events and (event_times_sorted[end] - event_times_sorted[start]) <= window:
            end += 1
        
        count = end - start
        if count >= threshold:
            
            big_event_start = event_times_sorted[start]
            big_event_end = event_times_sorted[end - 1]
            
            if big_events and big_event_start <= big_events[-1][1]:
                
                big_events[-1] = (big_events[-1][0], max(big_events[-1][1], big_event_end))
            else:
                big_events.append((big_event_start, big_event_end))
    return big_events




mseed_files = [f for f in os.listdir(MSEED_FOLDER) if f.lower().endswith(('.mseed', '.mse', '.msd', '.miniseed'))]

if not mseed_files:
    print(f"No MiniSEED files found in {MSEED_FOLDER}. Please check the directory and file extensions.")
    exit(1)


for mseed_file in mseed_files:
    mseed_path = os.path.join(MSEED_FOLDER, mseed_file)

    try:
        
        st = read(mseed_path)
        tr = st[0]  

        
        tr_filtered = tr.copy()
        tr_filtered.detrend('linear')  
        tr_filtered.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, corners=FILTER_ORDER, zerophase=True)

       
        velocity = tr_filtered.data
        sampling_rate = tr_filtered.stats.sampling_rate
        times = tr_filtered.times("relative")  

        
        velocity = hampel_filter(velocity, window_size=HAMPel_WINDOW_SIZE, n_sigma=HAMPel_N_SIGMA)

        
        velocity = moving_z_score_filter(velocity, window_size=MOVING_Z_WINDOW_SIZE, z_thresh=MOVING_Z_THRESHOLD)

        
        sta_primary, lta_primary = compute_sta_lta(velocity, sampling_rate, STA_WINDOW_PRIMARY, LTA_WINDOW_PRIMARY)
        sta_lta_ratio_primary = np.divide(sta_primary, lta_primary, out=np.zeros_like(sta_primary), where=lta_primary!=0)

        
        sta_secondary, lta_secondary = compute_sta_lta(velocity, sampling_rate, STA_WINDOW_SECONDARY, LTA_WINDOW_SECONDARY)
        sta_lta_ratio_secondary = np.divide(sta_secondary, lta_secondary, out=np.zeros_like(sta_secondary), where=lta_secondary!=0)

        
        triggers = trigger_onset(sta_lta_ratio_primary, TRIGGER_ON, TRIGGER_OFF)
        print(f"{mseed_file}: Detected triggers before validation: {len(triggers)}")

        
        validated_triggers = []
        non_validated_triggers = []
        for trigger in triggers:
            trigger_start_idx = trigger[0]
            
            if trigger_start_idx < len(sta_lta_ratio_secondary):
                if sta_lta_ratio_primary[trigger_start_idx] > VALIDATION_FACTOR * sta_lta_ratio_secondary[trigger_start_idx]:
                    validated_triggers.append(trigger)
                else:
                    non_validated_triggers.append(trigger)
            else:
               
                non_validated_triggers.append(trigger)

        print(f"{mseed_file}: Validated triggers: {len(validated_triggers)}")
        print(f"{mseed_file}: Non-validated triggers: {len(non_validated_triggers)}")

      
        big_events = identify_big_events(
            validated_triggers,
            times,
            threshold=BIG_EVENT_THRESHOLD,
            window=BIG_EVENT_WINDOW_SECONDS
        )
        print(f"{mseed_file}: Identified Big Events: {len(big_events)}")


        fig, ax1 = plt.subplots(figsize=(15, 7))

    
        ax1.plot(times, velocity, color='black', label='Velocity (m/s)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title(f'STA/LTA and Seismic Events for {mseed_file}')
        ax1.grid(True)


        ax2 = ax1.twinx()

       
        ax2.plot(times, sta_primary, color='orange', alpha=0.6, label=f'STA ({STA_WINDOW_PRIMARY}s)')
        ax2.plot(times, lta_primary, color='purple', alpha=0.6, label=f'LTA ({LTA_WINDOW_PRIMARY}s)')

 
        ax2.plot(times, sta_lta_ratio_primary, color='blue', alpha=0.6, label='STA/LTA Ratio')

     
        ax2.plot(times, sta_lta_ratio_secondary, color='green', alpha=0.6, linestyle=':', label='Secondary STA/LTA Ratio')

        ax2.set_ylabel('STA / LTA Metrics')

     
        for trigger in validated_triggers:
            trigger_start_rel = times[trigger[0]]
            trigger_end_rel = times[trigger[1]] if len(trigger) > 1 else trigger_start_rel
            label = 'Validated Trigger' if 'Validated Trigger' not in ax2.get_legend_handles_labels()[1] else ""
            ax2.axvspan(trigger_start_rel, trigger_end_rel, color='green', alpha=0.3, label=label)

      
        for trigger in non_validated_triggers:
            trigger_start_rel = times[trigger[0]]
            trigger_end_rel = times[trigger[1]] if len(trigger) > 1 else trigger_start_rel
            label = 'Non-validated Trigger' if 'Non-validated Trigger' not in ax2.get_legend_handles_labels()[1] else ""
            ax2.axvspan(trigger_start_rel, trigger_end_rel, color='yellow', alpha=0.3, label=label)

      
        for idx, (big_start, big_end) in enumerate(big_events):
            label = 'Big Event' if idx == 0 else ""
            ax2.axvspan(big_start, big_end, color='magenta', alpha=0.3, label=label)
           
            mid_time = (big_start + big_end) / 2
            mid_y = np.max(sta_lta_ratio_primary)
            ax2.annotate(
                'Big Event',
                xy=(mid_time, mid_y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color='magenta',
                fontsize=9,
                fontweight='bold'
            )

       
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        by_label = {}
        for line, label in zip(lines_1 + lines_2, labels_1 + labels_2):
            if label not in by_label and label != '':
                by_label[label] = line
        ax1.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.tight_layout()

        
        plot_filename = os.path.splitext(mseed_file)[0] + '_sta_lta_filtered.png'
        plt.savefig(os.path.join(OUTPUT_FOLDER, plot_filename))
        plt.close()

        print(f"{mseed_file}: Processed and saved plot.")

    except Exception as e:
        print(f"An error occurred while processing {mseed_file}: {e}")

print("All files processed.")
