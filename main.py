from train import train_model
from datetime import datetime

if __name__ == "__main__":
    model_name = "DeepDNN"  # Choose from: "SimpleDNN", "DeepDNN", "CNN1D"
    loss_name = "MSE"  # Choose from: "MSE", "Huber", "MAE"
    data_dir = "pes_data/"  # Folder containing PES files
    epochs = 100
    batch_size = 16
    learning_rate = 0.001
    patience = 20  # Stop if no improvement for 10 epochs

    current_time = filename1 = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{model_name}_{current_time}.pth"  # Where to save the best model

    train_model(model_name, loss_name, data_dir, epochs, batch_size, learning_rate, patience, save_path)
