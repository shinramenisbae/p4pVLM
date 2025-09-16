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
import requests
import time
from datetime import datetime, timedelta


model = EmotionCNN()
model.load_state_dict(torch.load("emotion_cnn.pth"))
model.eval()  # set to evaluation mode


# FastAPI base URL
BASE_URL = "http://130.216.217.53:8000"
# email = os.getenv("API_USERNAME")
# password = os.getenv("API_PASSWORD")
# user_id = os.getenv("API_ID")
#LMFAO ENV 
email = "nlia656@aucklanduni.ac.nz"
password = "nlia656"
user_id = "11"

# SET THIS TO THE DAY AND TIME OF THE EXPERIMENT
#eg 2025-09-12T02:27:20
#start_date = "2025-09-12"
# AND SET THIS TO DAY AFTER TO BE SAFE
# end_date = "2025-09-13"


COLUMNS = [
    "timestamp", "ppg_gr"
]

CHUNK_DURATION = 12
last_timestamp = datetime.now() - timedelta(seconds=CHUNK_DURATION) #small buffer
last_timestamp_str = last_timestamp.strftime("%Y-%m-%dT%I:%M:%S")


data = []  # Store fetched data

def login():
    if not email or not password:
        print("Error: Email and password are required!")
        return None

    login_data = {"username": email, "password": password}
    response = requests.post(f"{BASE_URL}/login-researcher", data=login_data)
    if response.status_code == 200:
        token = response.json().get("access_token")
        print("Login successful")
        return token
    else:
        print(f"Login Failed: {response.text}")
        return None

def clean_ppg_values(rows):
    cleaned = []
    for row in rows:
        raw_value = row.get("ppg_gr")
        if raw_value is None:
            continue
        values = []
        if isinstance(raw_value, list):
            values = raw_value
        elif isinstance(raw_value, str):
            try:
                numbers = ast.literal_eval(raw_value)
                if isinstance(numbers, list):
                    values = numbers
            except Exception:
                pass
        
        
        filtered_values = [v for v in values if 0 <= v]
        cleaned.extend(filtered_values)
    return cleaned



def fetch_data(token, last_timestamp):
    if not token:
        print("Error: You must login first!")
        return [], last_timestamp

    
    next_timestamp = last_timestamp + timedelta(seconds=CHUNK_DURATION)
    next_timestamp_str = next_timestamp.strftime("%Y-%m-%dT%I:%M:%S")
    last_timestamp_query = (last_timestamp - timedelta(seconds=CHUNK_DURATION)).strftime("%Y-%m-%dT%I:%M:%S")
    next_timestamp_query = (next_timestamp - timedelta(seconds=CHUNK_DURATION)).strftime("%Y-%m-%dT%I:%M:%S")
    params = {
        "columns": COLUMNS,
        "user_id": user_id,
        "start_date": last_timestamp_query,
        "end_date": next_timestamp_query
    }

    response = requests.get(f"{BASE_URL}/research/sensor-data",
                            headers={"Authorization": f"Bearer {token}"},
                            params=params)

    if response.status_code != 200:
        print(f"Error fetching data: {response.text}")
        return [], last_timestamp

    data = response.json()
    if not data:
        return [], last_timestamp

    new_ppg = clean_ppg_values(data)
    #need to reverse the data is newest first
    data = list(reversed(data))
    #take the "oldest" time (first after we reverse it)
    new_last_timestamp = datetime.fromisoformat(data[-1]["timestamp"])
    print(f"Fetched {len(data)} rows from API")
    print(f"Between time {last_timestamp_query} and {next_timestamp_query}")
    # print(data)
    return new_ppg, new_last_timestamp


def upsample_to_64hz(x_25hz, original_rate=25, target_rate=64):
    original_len = len(x_25hz)
    duration_sec = original_len / original_rate
    target_len = int(duration_sec * target_rate)
    x_64hz = resample(x_25hz, target_len)
    return x_64hz


def normalise_signal(signal):
    personal_min = np.min(signal)
    personal_max = np.max(signal)
    return (signal - personal_min) / (personal_max - personal_min) * 1000


def extract_pulses(signal, pulse_len=140):
    segments = []
    for start in range(0, len(signal) - pulse_len + 1, pulse_len):
        segments.append(signal[start:start + pulse_len])
    return segments


def predict_segments(segments_array):
    X_tensor = torch.tensor(segments_array, dtype=torch.float32)
    with torch.no_grad():
        valence_logits, arousal_logits = model(X_tensor)
        valence_preds = torch.argmax(valence_logits, dim=1)
        arousal_preds = torch.argmax(arousal_logits, dim=1)
    return valence_preds, arousal_preds

def is_valid_ppg(ppg_list):
    return any(v != -1 for v in ppg_list)

processed_hashes = set()
def hash_ppg(ppg_list):
    # need hash because sometimes api returns dupes of same timestamp
    return hash(tuple(ppg_list))

token = login()
if not token:
    exit(1)


#code for file
model_input = []
segmented_input = [] 


while True:
    start_loop = time.time()

    new_ppg, new_last_timestamp = fetch_data(token, last_timestamp)
    if new_ppg and is_valid_ppg(new_ppg):
        data_hash = hash_ppg(new_ppg)
        if data_hash in processed_hashes:
            print("Duplicate data")
        else:
            processed_hashes.add(data_hash)
            last_timestamp = max(new_last_timestamp, last_timestamp)

            # Stack raw PPG
            model_input.extend(new_ppg)

            # Upsample and normalise
            x_64hz = upsample_to_64hz(np.array(new_ppg))
            x_64hz_norm = normalise_signal(x_64hz)

            print(f"Samples in chunk: {len(new_ppg)}")
            # Segment
            segments = extract_pulses(x_64hz_norm)

            # Predict
            segments_array = np.array(segments)
            valence_preds, arousal_preds = predict_segments(segments_array)

            # Print results
            for i in range(len(segments)):
                print(f"Segment {i+1}: Valence = {valence_preds[i].item()}, Arousal = {arousal_preds[i].item()}")
    

    else:
        print("No new data yet.")

    last_timestamp += timedelta(seconds=CHUNK_DURATION)
    last_timestamp_str = last_timestamp.strftime("%Y-%m-%dT%I:%M:%S")
    print(f"Current timestamp: {last_timestamp_str}")
    elapsed = time.time() - start_loop
    sleep_time = max(0, CHUNK_DURATION - elapsed)
    time.sleep(sleep_time)