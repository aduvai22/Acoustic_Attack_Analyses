import numpy as np
import torch
from torch.utils.data import Dataset
import os
import re  # Regular expressions for filename parsing
import scipy.signal as signal
import librosa  # For MFCC computation

# Function to convert hexadecimal to signed 16-bit integer
def hex_to_signed_16bit(hex_str):
    value = int(hex_str, 16)
    if value >= 0x8000:
        value -= 0x10000
    return value

# Function to read PES data from file
def read_pes_data(file_path):
    avg_values = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    avg_val = hex_to_signed_16bit(parts[2])
                    avg_values.append(avg_val)
                except ValueError:
                    print(f"Skipping line due to format error: {line.strip()}")
    return np.array(avg_values, dtype=np.float32)

# Feature Extraction Functions
def compute_psd(pes_signal, fs=1000):  # Assume sampling rate 1000 Hz
    f, Pxx = signal.welch(pes_signal, fs=fs, nperseg=256)
    return Pxx  # Returns power spectral density

def compute_stft(pes_signal, fs=1000):
    f, t, Zxx = signal.stft(pes_signal, fs=fs, nperseg=256)
    return np.abs(Zxx).flatten()  # Returns STFT magnitude spectrum

def compute_mfcc(pes_signal, fs=1000, n_mfcc=13):
    pes_signal = pes_signal.astype(float)  # Ensure proper dtype
    mfccs = librosa.feature.mfcc(y=pes_signal, sr=fs, n_mfcc=n_mfcc)
    return mfccs.flatten()  # Returns MFCCs as a 1D feature vector


class PESDataset(Dataset):
    def __init__(self, data_dir, max_length=500, feature_type=None):
        self.data_dir = data_dir
        self.max_length = max_length
        self.files = []
        self.labels = []
        self.feature_type = feature_type

        # Regex pattern to extract SPL value from filenames
        pattern = re.compile(r'attack_(\d+)')

        for file_name in os.listdir(data_dir):
            match = pattern.search(file_name)
            if match:
                spl_value = float(match.group(1))  # Extract SPL from filename
                file_path = os.path.join(data_dir, file_name)
                self.files.append(file_path)
                self.labels.append(spl_value)

        assert len(self.files) > 0, "No valid PES files found in directory."

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pes_data = read_pes_data(self.files[idx])

        if self.feature_type:
            # Extract features
            if self.feature_type == "PSD":
                features = compute_psd(pes_data)
            elif self.feature_type == "STFT":
                features = compute_stft(pes_data)
            elif self.feature_type == "MFCC":
                features = compute_mfcc(pes_data)
            else:
                raise ValueError("Invalid feature type!")
            
            return torch.tensor(features, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
        
        else:
            # Return raw PES data
            # Zero-padding to fixed length
            padded_data = np.zeros(self.max_length)
            length = min(len(pes_data), self.max_length)
            padded_data[:length] = pes_data[:length]

            return torch.tensor(padded_data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
