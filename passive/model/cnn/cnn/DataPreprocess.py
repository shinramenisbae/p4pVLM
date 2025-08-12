import numpy as np
from scipy.signal import find_peaks

def detrend_ppg(ppg_signal, poly_order=10):
    x = np.arange(len(ppg_signal))
    coeffs = np.polyfit(x, ppg_signal, poly_order)
    poly_fit = np.polyval(coeffs, x)
    return ppg_signal - poly_fit

def segment_pulses(ppg_signal, pulse_len=140, overlap=20):
    peaks, _ = find_peaks(ppg_signal)
    pulses = []
    half_len = pulse_len // 2
    for peak in peaks:
        start = peak - half_len
        end = peak + half_len
        if start < 0 or end > len(ppg_signal):
            continue
        pulse = ppg_signal[start:end]
        pulses.append(pulse.reshape(-1, 1))  # shape (140, 1)
    return np.array(pulses)

def personal_normalization(pulses, person_min, person_max, alpha=1000):
    return (pulses - person_min) / (person_max - person_min) * alpha
