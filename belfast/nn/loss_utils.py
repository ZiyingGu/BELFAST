import torch
import torch.nn as nn
import numpy as np


def relative_mse(pred, target, eps=1e-6):
    """
    Compute the relative mean squared error between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values.
        eps (float, optional): Small value to avoid division by zero. Default 1e-6.

    Returns:
        torch.Tensor: Mean squared relative error.
    """
    rel_error = (pred - target) / (target + eps)
    return torch.mean(rel_error ** 2)


def weighted_mse(pred, target, weights=None):
    """
    Compute mean squared error with optional per-sample weights.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values.
        weights (torch.Tensor, optional): Sample weights of shape [N]. Default None.

    Returns:
        torch.Tensor: Weighted mean squared error.
    """
    error = (pred - target) ** 2
    if weights is not None:
        error = error * weights.unsqueeze(1)
    return torch.mean(error)


def significant_digit_loss(pred, target, digits=3, weights=None):
    """
    Compute loss based on rounding predictions and targets to a certain number of significant digits.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values.
        digits (int, optional): Number of significant digits. Default 3.
        weights (torch.Tensor, optional): Sample weights. Default None.

    Returns:
        torch.Tensor: Mean squared error after rounding to significant digits.
    """
    scale_array = np.array([0, 0, 0, 0, 0, 0, 0, 0]) + digits
    scale = torch.tensor(10 ** scale_array, dtype=pred.dtype, device=pred.device)

    pred_scaled = torch.round(pred * scale)
    target_scaled = torch.round(target * scale)
    error = (pred_scaled - target_scaled).float() ** 2
    if weights is not None:
        error = error * weights.unsqueeze(1)
    return torch.mean(error)


def hybrid_loss(pred, target, alpha=0.8, weights=None):
    """
    Combine MSE and MAE as a hybrid loss function.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values.
        alpha (float, optional): Weight for MSE component. (1-alpha) is for MAE. Default 0.8.
        weights (torch.Tensor, optional): Sample weights. Default None.

    Returns:
        torch.Tensor: Weighted hybrid loss.
    """
    mse = (pred - target) ** 2
    mae = (pred - target).abs()
    error = alpha * mse + (1 - alpha) * mae
    if weights is not None:
        error = error * weights.unsqueeze(1)
    return torch.mean(error)


def soft_significant_digit_loss(pred, target, digits=3, weights=None):
    """
    Soft version of significant digit loss without explicit rounding.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values.
        digits (int, optional): Number of significant digits. Default 3.
        weights (torch.Tensor, optional): Sample weights. Default None.

    Returns:
        torch.Tensor: Mean squared error scaled by 10^digits.
    """
    scale = 10 ** digits
    error = ((pred - target) * scale) ** 2
    if weights is not None:
        error = error * weights.unsqueeze(1)
    return torch.mean(error)


def compute_gaussian_weights(deviation_tensor, sigma=0.0):
    """
    Compute Gaussian weights from a tensor of deviations.

    Args:
        deviation_tensor (torch.Tensor): Deviation values.
        sigma (float): Standard deviation for Gaussian weighting. If 0, uniform weights. Must be >=0.

    Returns:
        torch.Tensor: Normalized weights.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    N = len(deviation_tensor)
    if sigma == 0.0:
        weights = torch.ones_like(deviation_tensor)
    else:
        weights = torch.exp(-0.5 * (deviation_tensor / sigma) ** 2)
        weights = weights * (N / weights.sum())
    return weights


def get_loss_function(name='mse', **kwargs):
    """
    Return a callable loss function based on a string name.

    Args:
        name (str): Loss function name. Options: 'mse', 'smoothl1', 'relative', 'weighted', 'sigdigits', 'softdigits', 'hybrid'.
        **kwargs: Additional keyword arguments for specific loss functions (e.g., digits, alpha).

    Returns:
        callable: Loss function accepting pred, target, and optional weights.
    """
    if name == 'mse':
        return lambda p, t, w=None: nn.MSELoss()(p, t) if w is None else weighted_mse(p, t, weights=w)
    elif name == 'smoothl1':
        return lambda p, t, w=None: nn.SmoothL1Loss(beta=kwargs.get('beta', 1.0))(p, t) if w is None else weighted_mse(p, t, weights=w)
    elif name == 'relative':
        return lambda p, t, w=None: relative_mse(p, t, eps=kwargs.get('eps', 1e-6))
    elif name == 'weighted':
        return lambda p, t, w=None: weighted_mse(p, t, weights=w)
    elif name == 'sigdigits':
        return lambda p, t, w=None: significant_digit_loss(p, t, digits=kwargs.get('digits', 3), weights=w)
    elif name == 'softdigits':
        return lambda p, t, w=None: soft_significant_digit_loss(p, t, digits=kwargs.get('digits', 3), weights=w)
    elif name == 'hybrid':
        return lambda p, t, w=None: hybrid_loss(p, t, alpha=kwargs.get('alpha', 0.8), weights=w)
    else:
        raise ValueError(f"Unknown loss function name: {name}")

