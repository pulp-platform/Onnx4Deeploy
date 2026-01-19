import numpy as np
import os
import pandas as pd
import scipy.signal
from scipy import signal, fft #for scipy 1.8
from scipy import fftpack #for scipy 1.1
from scipy.signal import convolve
# notch filter
def notch_multiEEG(data,fs):
    '''
    50Hz multi channel EEG
    
    Params
    ------
    data: np array (n_ch,n_s)
        EEG data
    fs: float
        sampling frequency
    
    Return
    ------
    data_filt: np array (n_ch, n_s)
        filtered EEG data 
    '''
    
    # data = np.transpose(data[:,:])
    w0 = 50/(fs/2)
    # Create the notch filter (b, a) for the 50Hz noise
    notch_filter_50Hz = signal.iirnotch(w0, Q=30)  #30?
    # notch_filter_50Hz = signal.iirnotch(50, Q=30, fs=500)
    
    #data_filt = np.zeros(data.shape)
    for chan in range(data.shape[0]):
         
        data[chan,:]= signal.lfilter(*notch_filter_50Hz, data[chan,:])
        
    return data
def norm_freq(f, fs):
    return f * 2 / fs



# bandpass filter
def bandpass_multiEEG(data,f_low,f_high,fs):
    '''
    Bandpass multi channel EEG
    
    Params
    ------
    data: np array (n_ch,n_s)
        EEG data
    f_low: float
        lower corner frequency [Hz]
    f_high: float
        upper corner_frequency [Hz]
    fs: float
        sampling frequency
    
    Return
    ------
    data_filt: np array (n_ch, n_s)
        filtered EEG data 
    '''
    
    # data = np.transpose(data[:,:])
    low_freq = norm_freq(f_low, fs)
    high_freq = norm_freq(f_high, fs)
    
    filt_coef = scipy.signal.butter(N=10, Wn=[low_freq, high_freq], btype='bandpass', output='sos')
    
    data_filt = np.zeros(data.shape)
    for chan in range(data.shape[0]):
        data_filt[chan] = scipy.signal.sosfilt(filt_coef, data[chan])  #sosfiltfilt
           
    return data_filt

def remove_running_mean(data, window_size):
    # Ensure the window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    n = len(data)
    pad_size = window_size - 1
    # Replicate padding to consider only past data
    padded_data = np.pad(data, (pad_size, 0), 'edge')
    
    detrended_data = np.zeros_like(data)
    for i in range(n):
        current_window = padded_data[i:i + window_size]
        detrended_data[i] = data[i] - np.mean(current_window)
    return detrended_data
        
def filter_eeg_data(data, fs, window_length_ms, method='chunk'):
    '''
    Apply a specified filter method to EEG data: 'chunk' for chunk-based DC removal or 'moving_average' for a moving average filter.

    Params
    ------
    data: np.array
        EEG data (n_ch, n_s)
    fs: int
        Sampling frequency in Hz
    window_length_ms: float
        Length of the window in milliseconds
    method: str
        The method of filtering: 'chunk' or 'moving_average'

    Returns
    -------
    filtered_data: np.array
        Filtered EEG data
    '''

    if method not in ['chunk', 'moving_average', 'matlab']:
        raise ValueError("Method must be 'chunk', 'moving_average', or 'matlab'")

    # Calculate the number of samples in each window
    window_length_samples = int(fs * (window_length_ms / 1000))
    N=10
    # Create the moving average filter 'b'
    b = np.ones(N) / N

    if method == 'chunk':
        # Initialize the filtered data array
        filtered_data = np.zeros(data.shape)

        # Process each channel separately
        for chan in range(data.shape[0]):
            # Process data in chunks
            for i in range(0, data.shape[1], window_length_samples):
                end = min(i + window_length_samples, data.shape[1])
                chunk_mean = np.mean(data[chan, i:end])
                filtered_data[chan, i:end] = data[chan, i:end] - chunk_mean

    elif method == 'moving_average':
        # Initialize the filtered data array
        filtered_data = np.zeros(data.shape)

        # Apply the moving average filter
        for chan in range(data.shape[0]):
            for i in range(data.shape[1]):
                # Define the window range
                start = max(i - window_length_samples // 2, 0)
                end = min(i + window_length_samples // 2, data.shape[1])
                
                # Calculate the mean
                window_mean = np.mean(data[chan, start:end])
                
                # Subtract the mean from the current sample
                filtered_data[chan, i] = data[chan, i] - window_mean
    elif method == 'matlab':
        filtered_data = np.zeros(data.shape)
        for chan in range(data.shape[0]):
            detrended_data = remove_running_mean(data[chan], 500)
            filtered_data[chan] = convolve(detrended_data, b, mode='same')


    return filtered_data

def load_real_data_new(data_path, sampling_freq=500, window_size=2, method='chunk',zero_pad = True, notch_filter = False, bandpass_filter = False):
    task_triggers = [50, 51, 52, 53, 61, 62, 63, 64, 70, 71, 80]
    window_samples = sampling_freq * window_size
    eeg_all, labels_all = [], []

    data_files = os.listdir(data_path)
    data_files.sort()  # Ensure consistent order
    window_length_ms = 250
    fs = 500
    bp_low = 0.5
    #bp_high = 100
    bp_high = 40

    for file in data_files:
        df = pd.read_csv(os.path.join(data_path, file))
        eeg_1 = df['EEG 1'].to_numpy(dtype=np.float32)
        eeg_2 = df['EEG 2'].to_numpy(dtype=np.float32)
        eeg_3 = df['EEG 3'].to_numpy(dtype=np.float32)
        trigger = df['Trigger'].to_numpy()

        # Apply your filtering functions here
        # e.g., eeg_1 = notch_filter(eeg_1, sampling_freq)
        

        # Exclude trigger values not related to task triggers
        valid_indices = np.isin(trigger, task_triggers)

        # Group the EEG data in a format: (channels, samples)
        #eeg_1 = (eeg_1 / 1000).astype(np.float32)
        #eeg_2 = (eeg_2 / 1000).astype(np.float32)
        #eeg_3 = (eeg_3 / 1000).astype(np.float32)


        eeg_data = np.array([eeg_1, eeg_2, eeg_3], dtype=np.float32)

        if notch_filter:
            eeg_data = notch_multiEEG(eeg_data, fs)
        if bandpass_filter:
            eeg_data = bandpass_multiEEG(eeg_data,bp_low,bp_high, fs)

        eeg_data = eeg_data[:, valid_indices]
        trigger = trigger[valid_indices]

        eeg_data = filter_eeg_data(eeg_data, fs=sampling_freq, window_length_ms=window_length_ms, method=method)

        # Process each window
        segmented_data = []
        window_labels = []
        i = 0
        while i < len(trigger) - window_samples + 1:
            window_triggers = trigger[i:i + window_samples]
            unique, counts = np.unique(window_triggers, return_counts=True)
            if np.max(counts) >= 1000:  # Majority threshold
                max_index = np.argmax(counts)
                if unique[max_index] in task_triggers:
                    data_window = eeg_data[:, i:i + window_samples]
                    segmented_data.append(data_window)
                    window_labels.append(unique[max_index])
                    i += window_samples - 1  # Move to the end of the window
            i += 1

        if segmented_data:
            segmented_data = np.stack(segmented_data, axis=2)  # Stack along new dimension
            window_labels = np.array(window_labels)
            if len(eeg_all) == 0:
                eeg_all = segmented_data
                labels_all = window_labels
            else:
                eeg_all = np.concatenate((eeg_all, segmented_data), axis=2)
                labels_all = np.concatenate((labels_all, window_labels))
    # Change the labels so they are of form 0,1,2...
    label_map = [50, 51, 52, 53, 61, 62, 63, 64, 70, 71, 80]
    label_mapping = {50: 0, 51: 1, 52: 2, 53: 3, 61: 4, 62: 5, 63: 6, 64: 7, 70: 8, 71: 9, 80: 10}
    for i in range(labels_all.shape[0]):
        if(labels_all[i] in label_map):
            labels_all[i] = label_mapping[labels_all[i]]
        else:
            print("Something went wrong")
    if(zero_pad):        
        # Next we loop throuhg eeg_all and add 6 zeros to every 250 samples of data.
        # This means that the new data shape will be (channels, samples_per_window + 24, num_windows) as there will be 24 added zeros for every 2s window.
        eeg_all_new = np.zeros((eeg_all.shape[0], eeg_all.shape[1] + 24, eeg_all.shape[2]))

        for i in range(eeg_all.shape[2]):
            for j in range(0, eeg_all.shape[1], 250):
                eeg_all_new[:, j:(j + 250), i] = eeg_all[:, j:(j + 250), i]
                if j + 250 < eeg_all_new.shape[1]:
                    eeg_all_new[:, (j + 250):(j + 256), i] = 0
        return eeg_all_new, labels_all


    return eeg_all, labels_all


def load_real_data_new_new(data_path, sampling_freq=500, window_size=2, method='chunk',zero_pad = True, notch_filter = False, bandpass_filter = False):
    task_triggers = [50, 51, 52, 53, 61, 62, 63, 64, 70, 71, 80]
    window_samples = sampling_freq * window_size
    eeg_all, labels_all = [], []

    data_files = os.listdir(data_path)
    data_files.sort()  # Ensure consistent order
    window_length_ms = 250
    fs = 500
    bp_low = 0.5
    #bp_high = 100
    bp_high = 40

    for file in data_files:
        df = pd.read_csv(os.path.join(data_path, file))
        eeg_1 = df['EEG 1'].to_numpy(dtype=np.float32)
        eeg_2 = df['EEG 2'].to_numpy(dtype=np.float32)
        eeg_3 = df['EEG 3'].to_numpy(dtype=np.float32)
        trigger = df['Trigger'].to_numpy()
        ver_eeg = eeg_2 - (eeg_1 + eeg_3) / 2
        hor_eeg = eeg_1 - eeg_3

        # Apply your filtering functions here
        # e.g., eeg_1 = notch_filter(eeg_1, sampling_freq)
        

        # Exclude trigger values not related to task triggers
        valid_indices = np.isin(trigger, task_triggers)

        # Group the EEG data in a format: (channels, samples)
        #eeg_1 = (eeg_1 / 1000).astype(np.float32)
        #eeg_2 = (eeg_2 / 1000).astype(np.float32)
        #eeg_3 = (eeg_3 / 1000).astype(np.float32)


        eeg_data = np.array([ver_eeg, hor_eeg], dtype=np.float32)

        if notch_filter:
            eeg_data = notch_multiEEG(eeg_data, fs)
        if bandpass_filter:
            eeg_data = bandpass_multiEEG(eeg_data,bp_low,bp_high, fs)

        eeg_data = eeg_data[:, valid_indices]
        trigger = trigger[valid_indices]

        eeg_data = filter_eeg_data(eeg_data, fs=sampling_freq, window_length_ms=window_length_ms, method=method)

        # Process each window
        segmented_data = []
        window_labels = []
        i = 0
        while i < len(trigger) - window_samples + 1:
            window_triggers = trigger[i:i + window_samples]
            unique, counts = np.unique(window_triggers, return_counts=True)
            if np.max(counts) >= 1000:  # Majority threshold
                max_index = np.argmax(counts)
                if unique[max_index] in task_triggers:
                    data_window = eeg_data[:, i:i + window_samples]
                    segmented_data.append(data_window)
                    window_labels.append(unique[max_index])
                    i += window_samples - 1  # Move to the end of the window
            i += 1

        if segmented_data:
            segmented_data = np.stack(segmented_data, axis=2)  # Stack along new dimension
            window_labels = np.array(window_labels)
            if len(eeg_all) == 0:
                eeg_all = segmented_data
                labels_all = window_labels
            else:
                eeg_all = np.concatenate((eeg_all, segmented_data), axis=2)
                labels_all = np.concatenate((labels_all, window_labels))
    # Change the labels so they are of form 0,1,2...
    label_map = [50, 51, 52, 53, 61, 62, 63, 64, 70, 71, 80]
    label_mapping = {50: 0, 51: 1, 52: 2, 53: 3, 61: 4, 62: 5, 63: 6, 64: 7, 70: 8, 71: 9, 80: 10}
    for i in range(labels_all.shape[0]):
        if(labels_all[i] in label_map):
            labels_all[i] = label_mapping[labels_all[i]]
        else:
            print("Something went wrong")
    if(zero_pad):        
        # Next we loop throuhg eeg_all and add 6 zeros to every 250 samples of data.
        # This means that the new data shape will be (channels, samples_per_window + 24, num_windows) as there will be 24 added zeros for every 2s window.
        eeg_all_new = np.zeros((eeg_all.shape[0], eeg_all.shape[1] + 24, eeg_all.shape[2]))

        for i in range(eeg_all.shape[2]):
            for j in range(0, eeg_all.shape[1], 250):
                eeg_all_new[:, j:(j + 250), i] = eeg_all[:, j:(j + 250), i]
                if j + 250 < eeg_all_new.shape[1]:
                    eeg_all_new[:, (j + 250):(j + 256), i] = 0
        return eeg_all_new, labels_all


    return eeg_all, labels_all
