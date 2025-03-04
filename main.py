from train import train_model
from datetime import datetime

if __name__ == "__main__":
    model_name = "SimpleDNN"  # Options: "SimpleDNN", "DeepDNN", "CNN1D", "FeatureDNN", "RNN", "LSTM", "GRU"
    loss_name = "MSE"  # Options: "MSE", "Huber", "MAE"
    data_dir = "pes_data/"  # Folder containing PES files
    feature_type = "STFT" # Options: "PSD", "STFT", "MFCC", None
    epochs = 200
    batch_size = 16
    learning_rate = 0.001
    patience = 20  # Stop if no improvement for 10 epochs

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{model_name}_{current_time}.pth"  # Where to save the best model

    train_model(model_name, loss_name, data_dir, feature_type, epochs, batch_size, learning_rate, patience, save_path)
