import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

from model import get_model
from loss import get_loss_function
from dataset import PESDataset

def train_model(model_name, loss_name, data_dir, feature_type=None, epochs=50, batch_size=16, learning_rate=0.001, patience=10, save_path="best_model.pth"):

    # Create directory to save weights
    if not os.path.exists("weights"):
        os.makedirs("weights")
    save_path = os.path.join("weights", save_path)

    # Load dataset and split into training & validation sets (80% train, 20% val)
    dataset = PESDataset(data_dir, feature_type=feature_type)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # input_size = dataset.max_length  # Fixed-length input
    input_size = len(train_dataset[0][0]) 
    model = get_model(model_name, input_size)
    loss_function = get_loss_function(loss_name)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0  # Counter for early stopping

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Compute validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_function(outputs.squeeze(), targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Model saved! (Validation Loss Improved: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs of no improvement.")
            break

    print("Training completed. Best model saved as:", save_path)
