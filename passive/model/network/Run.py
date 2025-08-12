from CNN import EmotionCNN
from DataLoader import loadAllParticipants
import torch
from collections import defaultdict, Counter
import csv
import os
import re
import ast
import json
from scipy.signal import resample, find_peaks
import numpy as np


model = EmotionCNN()
model.load_state_dict(torch.load("emotion_cnn.pth"))
model.eval()  # set to evaluation mode

dataFolder = './input-folder' 
model_input = []

# Load data
for filename in os.listdir(dataFolder):
    if filename.endswith(".csv"):
        filepath = os.path.join(dataFolder, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

            matches = re.findall(r'\[[^\]]*\]', content)

            for match in matches:
                try:
                    numbers = ast.literal_eval(match)
                    if isinstance(numbers, list):
                        model_input.extend(numbers)
                except:
                    pass
        break #remove if not just first file

def upsample_to_64hz(x_25hz, original_rate=25, target_rate=64):
    original_len = len(x_25hz)
    duration_sec = original_len / original_rate
    target_len = int(duration_sec * target_rate)
    x_64hz = resample(x_25hz, target_len)
    return x_64hz

x_25hz = np.array(model_input)
x_64hz = upsample_to_64hz(x_25hz)



personal_min = np.min(x_64hz)
personal_max = np.max(x_64hz)
x_64hz_normalized = (x_64hz - personal_min) / (personal_max - personal_min) * 1000

# peak based
# def extract_pulses(signal, fs=128, pulse_len=140, avg_period=120):
#     peaks, _ = find_peaks(signal, distance=pulse_len)
    
#     segments = []
#     half_len = pulse_len // 2

#     for peak in peaks:
#         start = peak - half_len
#         end = peak + half_len
#         if start >= 0 and end <= len(signal):
#             segments.append(signal[start:end])
    
#     return segments

#not peak based
def extract_pulses(signal, pulse_len=140):
    segments = []
    for start in range(0, len(signal) - pulse_len + 1, pulse_len):
        segments.append(signal[start:start + pulse_len])
    return segments


# segments = extract_pulses(x_64hz_normalized, pulse_len=140, avg_period=120)
segments = extract_pulses(x_64hz_normalized, pulse_len=140)
segments_array = np.array(segments)
print(segments_array)


X_tensor = torch.tensor(segments_array, dtype=torch.float32)


with torch.no_grad():
    valence_logits, arousal_logits = model(X_tensor)

    valence_preds = torch.argmax(valence_logits, dim=1)
    arousal_preds = torch.argmax(arousal_logits, dim=1)

for i in range(len(segments)):
    print(f"Segment {i+1}: Valence = {valence_preds[i].item()}, Arousal = {arousal_preds[i].item()}")