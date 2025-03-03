import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(predictions, ground_truths):
    """
    Calculate various regression metrics.

    Args:
        predictions (list): List of predicted SPL values.
        ground_truths (list): List of actual SPL values.

    Returns:
        dict: A dictionary containing MAE, RMSE, R², and mean accuracy.
    """
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    mae = mean_absolute_error(ground_truths, predictions)
    rmse = np.sqrt(mean_squared_error(ground_truths, predictions))
    r2 = r2_score(ground_truths, predictions)

    # Define accuracy as % of predictions within ±3 dB of actual
    accuracy_threshold = 5.0  # dB
    accuracy = np.mean(np.abs(predictions - ground_truths) <= accuracy_threshold) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R² Score": r2,
        "Mean Accuracy (%)": accuracy
    }
