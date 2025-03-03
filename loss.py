import torch
import torch.nn as nn

def get_loss_function(loss_name):
    if loss_name == "MSE":
        return nn.MSELoss()
    elif loss_name == "Huber":
        return nn.HuberLoss()
    elif loss_name == "MAE":
        return nn.L1Loss()
    else:
        raise ValueError("Unknown loss function")
