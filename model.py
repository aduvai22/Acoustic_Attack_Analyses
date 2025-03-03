import torch
import torch.nn as nn

class SimpleDNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output SPL in dB
        )

    def forward(self, x):
        return self.model(x)

class DeepDNN(nn.Module):
    def __init__(self, input_size):
        super(DeepDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * input_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class FeatureDNN(nn.Module):
    def __init__(self, input_size):
        super(FeatureDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])  # Use last time step output

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])  # Use last time step output

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])  # Use last time step output
      
def get_model(model_name, input_size):
    if model_name == "SimpleDNN":
        return SimpleDNN(input_size)
    elif model_name == "DeepDNN":
        return DeepDNN(input_size)
    elif model_name == "CNN1D":
        return CNN1D(input_size)
    elif model_name == "FeatureDNN":
        return FeatureDNN(input_size)
    elif model_name == "RNN":
        return RNNModel(input_size)
    elif model_name == "LSTM":
        return LSTMModel(input_size)
    elif model_name == "GRU":
        return GRUModel(input_size)
    else:
        raise ValueError("Unknown model name")
