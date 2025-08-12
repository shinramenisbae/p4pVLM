import os
import pickle
import numpy as np
from scipy.signal import decimate, find_peaks
from numpy.polynomial.polynomial import Polynomial

def detrend_poly(signal, order=32):
    x = np.arange(len(signal))
    p = Polynomial.fit(x, signal, deg=order)
    trend = p(x)
    return signal - trend

def extract_pulses(signal, fs=128, pulse_len=140, avg_period=120, overlap=20):
    #get peaks of signal
    peaks, _ = find_peaks(signal, distance=avg_period)

    segments = []
    half_len = pulse_len // 2

    #For each peak we define the start and end to get a segment centered on the peak
    for peak in peaks:
        start = peak - half_len
        end = peak + half_len
        if start >= 0 and end <= len(signal):
            segments.append(signal[start:end])
        
        step = avg_period - overlap
        next_peak = peak + step

        #Slide through the signal which gives overlapping segments
        while next_peak < len(signal) - half_len:
            seg_start = next_peak - half_len
            seg_end = next_peak + half_len
            if seg_start >= 0 and seg_end <= len(signal):
                segments.append(signal[seg_start:seg_end])
            next_peak += step
    return segments

def loadAllParticipants(dataFolder):
    allData = []
    allLabels = []

    for filename in os.listdir(dataFolder):
        if filename.endswith(".dat"):
            filepath = os.path.join(dataFolder, filename)
            with open(filepath, 'rb') as file:
                participant = pickle.load(file, encoding='latin1')

            raw_ppg = participant['data'][:, 39, :] 
            labels = participant['labels']

            all_trials_processed = []
            for i in range(raw_ppg.shape[0]):
                trial = raw_ppg[i]

                # Downsample to 64 Hz
                trial_ds = decimate(trial, 2)

                # Detrend
                trial_dt = detrend_poly(trial_ds, order=32)
                all_trials_processed.append(trial_dt)
            all_trials_processed = np.array(all_trials_processed)
            personal_min = np.min(all_trials_processed)
            personal_max = np.max(all_trials_processed)

            #Stacks the trials in 1 line, uses append
            for i in range(raw_ppg.shape[0]):
                trial_dt = all_trials_processed[i]

                # Normalize using personal min and max
                trial_norm = (trial_dt - personal_min) / (personal_max - personal_min) * 1000

                # Segment pulses
                segments = extract_pulses(trial_norm, pulse_len=140, avg_period=120, overlap=20)

                valence_label = int(participant['labels'][i][0] >= 5)
                arousal_label = int(participant['labels'][i][1] >= 5)

                for seg in segments:
                    if len(seg) == 140:
                        allData.append(seg)
                        allLabels.append([valence_label, arousal_label])

    return np.array(allData), np.array(allLabels)



#for graphing data + detrending

import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, find_peaks
from numpy.polynomial.polynomial import Polynomial

def visualise_detrending(dataFolder, order=32):
    for filename in os.listdir(dataFolder):
        if filename.endswith(".dat"):
            filepath = os.path.join(dataFolder, filename)
            with open(filepath, 'rb') as file:
                participant = pickle.load(file, encoding='latin1')

            # Get first trial's raw PPG from channel 39
            raw_ppg = participant['data'][0, 39, :]
            trial_ds = decimate(raw_ppg, 2)  # Downsample to 64 Hz
            x = np.arange(len(trial_ds))

            # Fit polynomial trend and detrend
            p = Polynomial.fit(x, trial_ds, deg=order)
            trend = p(x)
            detrended = trial_ds - trend

            # Find peaks in detrended signal (at least 0.5s apart = 64 samples)
            peaks, _ = find_peaks(detrended, distance=64)

            # Plot full-length signal
            plt.figure(figsize=(18, 6))
            plt.plot(x, detrended, label='Detrended Signal', color='green', alpha=0.8)
            plt.plot(x[peaks], detrended[peaks], 'rx', label='Detected Peaks')
            plt.title(f'Full Detrended Signal with Peaks - {filename} (Trial 1)')
            plt.xlabel('Sample Index (at 128 Hz)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            break  # Only one participant for now




# visualise_detrending(dataFolder='./data')