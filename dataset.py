import numpy as np
import torch
from torch.utils.data import Dataset
import os
import re  # Regular expressions for filename parsing

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

class PESDataset(Dataset):
    def __init__(self, data_dir, max_length=500):
        self.data_dir = data_dir
        self.max_length = max_length
        self.files = []
        self.labels = []

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

        # Zero-padding to fixed length
        padded_data = np.zeros(self.max_length)
        length = min(len(pes_data), self.max_length)
        padded_data[:length] = pes_data[:length]

        return torch.tensor(padded_data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
