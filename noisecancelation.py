import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from obspy import UTCDateTime, Trace, Stream
from obspy.io.mseed import InternalMSEEDError

input_folder = 'path_to_input_folder'      
output_folder = 'path_to_output_folder'  


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def wavelet_interpolation(signal, time_rel, wavelet='db4', level=5):
   
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    
    mask = signal != 0  

 
    len_approx = len(coeffs[0])
    time_rel_downsampled = np.linspace(time_rel[0], time_rel[-1], len_approx)
    if np.sum(mask[:len_approx]) > 1:
        interp_func_approx = interp1d(time_rel_downsampled[mask[:len_approx]], coeffs[0][mask[:len_approx]],
                                      kind='linear', fill_value="extrapolate")
        coeffs[0] = interp_func_approx(time_rel_downsampled)
    
   
    for i in range(1, len(coeffs)):
        len_detail = len(coeffs[i])
        time_rel_downsampled = np.linspace(time_rel[0], time_rel[-1], len_detail)
        if np.sum(mask[:len_detail]) > 1:
            interp_func_detail = interp1d(time_rel_downsampled[mask[:len_detail]], coeffs[i][mask[:len_detail]],
                                          kind='linear', fill_value="extrapolate")
            coeffs[i] = interp_func_detail(time_rel_downsampled)
    
    return coeffs


def bayes_shrink(coeff):
    var = np.median(np.abs(coeff)) / 0.6745  
    sigma = np.sqrt(var)
    threshold = sigma * np.sqrt(2 * np.log(len(coeff)))
    return pywt.threshold(coeff, threshold, mode='soft')  


def sure_threshold(coeff):
    n = len(coeff)
    sigma = np.median(np.abs(coeff)) / 0.6745  
    sure_threshold = sigma * np.sqrt(2 * np.log(n))  
    return pywt.threshold(coeff, sure_threshold, mode='soft')


def apply_butterworth_filter(signal, fs, cutoff_freq, order=4):
    """
    Apply a low-pass Butterworth filter to the signal.
    
    Args:
    - signal: The input signal.
    - fs: Sampling frequency (Hz).
    - cutoff_freq: Cutoff frequency for the low-pass filter (Hz).
    - order: Order of the Butterworth filter.
    
    Returns:
    - filtered_signal: The signal after applying the Butterworth filter.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def denoise_signal(velocity, time_rel, file_name):
   
    coeffs = wavelet_interpolation(velocity, time_rel)
    
    coeffs[0] = bayes_shrink(coeffs[0])
    

    for i in range(1, len(coeffs)):
        coeffs[i] = sure_threshold(coeffs[i])
    
    
    denoised_signal = pywt.waverec(coeffs, 'db4')[:len(velocity)]
    

    fs = 1 / (time_rel[1] - time_rel[0])  
    cutoff_freq = fs / 4  
    denoised_signal = apply_butterworth_filter(denoised_signal, fs, cutoff_freq)
    

    plt.figure(figsize=(10, 6))
    plt.plot(time_rel, velocity, label='Original Signal', alpha=0.5, color='blue')
    plt.plot(time_rel, denoised_signal, label='Final Denoised Signal', color='red')
    plt.title(f'Final Denoised Signal - {file_name}')
    plt.xlabel('Relative Time (sec)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'final_denoised_{file_name.replace(".csv", ".png")}'))
    plt.close()
    
    return denoised_signal

for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):  
        input_file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(input_file_path)
        
     
        print(f"Processing file: {file_name}")
        print("Available columns:", data.columns.tolist())
        
        
        data.columns = data.columns.str.strip()
        
    
        possible_velocity_columns = ['velocity(m/s)', 'Velocity (m/s)', 'velocity_m/s', 'Velocity_m/s']
        
      
        velocity_column = next((col for col in possible_velocity_columns if col in data.columns), None)
        
        if velocity_column is None:
            print(f"ERROR: No velocity column found in {file_name}. Available columns: {data.columns.tolist()}")
            continue  
        
        try:
            time_abs = data['time_abs(%Y-%m-%dT%H:%M:%S.%f)']
        except KeyError:
            print(f"ERROR: 'time_abs(%Y-%m-%dT%H:%M:%S.%f)' column not found in {file_name}. Available columns: {data.columns.tolist()}")
            continue  
        
        try:
            time_rel = data['time_rel(sec)'].values
        except KeyError:
            print(f"ERROR: 'time_rel(sec)' column not found in {file_name}. Available columns: {data.columns.tolist()}")
            continue 
        
        velocity = data[velocity_column].values
       
        denoised_velocity = denoise_signal(velocity, time_rel, file_name)
        
      
        denoised_data = pd.DataFrame({
            'time_abs(%Y-%m-%dT%H:%M:%S.%f)': time_abs,
            'time_rel(sec)': time_rel,
            'denoised_velocity(m/s)': denoised_velocity
        })
        output_csv_file = os.path.join(output_folder, f'denoised_{file_name}')
        denoised_data.to_csv(output_csv_file, index=False)

def csv_to_mseed(input_folder, output_folder):
  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.csv', '.mseed'))

            try:
           
                df = pd.read_csv(input_path)

                
                time_column = df.columns[0]  
                data_column = df.columns[-1]  

                
                times = [UTCDateTime(t) for t in df[time_column]]

                
                sampling_rate = 1 / (times[1] - times[0])

                
                trace = Trace(data=df[data_column].values)
                trace.stats.starttime = times[0]
                trace.stats.sampling_rate = sampling_rate
                trace.stats.network = 'XX'  
                trace.stats.station = 'STA'  
                trace.stats.channel = 'BHZ' 

               
                stream = Stream([trace])
                stream.write(output_path, format='MSEED')

                print(f"Converted {filename} to MiniSEED format.")

            except InternalMSEEDError as e:
                print(f"Error converting {filename}: {str(e)}")
                print("This may be due to data incompatibility with MiniSEED format.")
                print("Consider adjusting the data or using a different output format.")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    input_folder = 'path/to/input/csv/folder'
    output_folder = 'path/to/output/mseed/folder'
    csv_to_mseed(input_folder, output_folder)