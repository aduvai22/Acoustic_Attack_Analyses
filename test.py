import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os

from model import get_model
from dataset import PESDataset
from metrics import calculate_metrics

def test_model(model_name, data_dir, model_path="best_model.pth"):
    # Load dataset (same as training)
    dataset = PESDataset(data_dir)

    # Use the validation portion as the test set
    test_percentage = 0.5
    test_size = int(test_percentage * len(dataset))
    train_size = len(dataset) - test_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    input_size = dataset.max_length

    # Load the trained model
    model = get_model(model_name, input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Testing model on validation (test) set...")
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.item())
            ground_truths.append(targets.item())

    # Calculate evaluation metrics
    metrics = calculate_metrics(predictions, ground_truths)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot results
    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    plt.scatter(ground_truths, predictions, color='blue', label='Predicted vs Actual')
    
    # Line of perfect fit (y = x)
    min_spl = min(min(predictions), min(ground_truths))
    max_spl = max(max(predictions), max(ground_truths))
    plt.plot([min_spl, max_spl], [min_spl, max_spl], color='red', linestyle="--", label="Ideal Fit")

    plt.xlabel("Actual SPL (dB)")
    plt.ylabel("Predicted SPL (dB)")
    plt.title("Model Predictions vs Actual SPL")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    data_dir = "pes_data/"

    model_name = "SimpleDNN"  # Set model type
    model_path = "weights/SimpleDNN_20250303_123100.pth" # Set model path

    test_model(model_name, data_dir, model_path)
