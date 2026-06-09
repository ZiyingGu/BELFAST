import torch
import torch.nn as nn
import os


class MLPRegressor(nn.Module):
    """
    A Multilayer Perceptron (MLP) regressor neural network.

    This model is designed for regression tasks with a configurable input dimension and activation function.
    It consists of multiple linear layers with batch normalization and activation functions applied after each layer,
    except the final output layer.

    Args:
        activation (str): The activation function to use. Supported options are 'tanh', 'softplus', 'swish',
                          'relu', and 'leaky_relu'. Default is 'swish'.
        input_dim (int): The number of input features. Default is 6 for backward compatibility.

    Architecture:
        - Input layer with size `input_dim`
        - Hidden layers: 512 -> 1024 -> 1024 -> 1024 -> 1024 -> 512 -> 256 -> 128 units
          Each hidden layer is followed by Batch Normalization and the chosen activation function,
          except the last hidden layer (128 units) which is followed only by the activation function.
        - Output layer: 8 units (no activation)

    Output:
        The output is a tensor of shape (batch_size, 8), representing the regression outputs.
    """
    def __init__(self, activation='swish', input_dim=6):
        """
        input_dim: number of input features. Default is 6 for backward compatibility.
        """
        super().__init__()

        activations = {
            'tanh': nn.Tanh(),
            'softplus': nn.Softplus(),
            'swish': nn.SiLU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU()
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from {list(activations.keys())}.")

        act_fn = activations[activation]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            act_fn,

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            act_fn,

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            act_fn,

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            act_fn,

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            act_fn,

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            act_fn,

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            act_fn,

            nn.Linear(256, 128),
            act_fn,

            nn.Linear(128, 8)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 8).
        """
        return self.net(x)


def save_model(model, path, verbose=True):
    """
    Save the state dictionary of the given model to the specified path.

    Args:
        model (nn.Module): The PyTorch model to save.
        path (str): The file path where the model state_dict will be saved.
        verbose (bool, optional): If True, prints a confirmation message. Default is True.

    Behavior:
        Creates the directory for the path if it does not exist.
        Saves only the model's state dictionary (weights).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    if verbose:
        print(f"✅ Model saved to {path}")


def load_model(path, activation='swish', input_dim=6, map_location=None, verbose=True):
    """
    Load a model's state dictionary from a checkpoint file and return the model instance.

    Args:
        path (str): Path to the saved model state dictionary.
        activation (str, optional): Activation function used in the model. Default is 'swish'.
        input_dim (int, optional): Number of input features expected by the model. Default is 6.
        map_location (str or torch.device, optional): Device mapping for loading the model. Default is None.
        verbose (bool, optional): If True, prints a confirmation message. Default is True.

    Returns:
        MLPRegressor: The model loaded with the saved weights.
    """
    model = MLPRegressor(activation=activation, input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location=map_location))
    if verbose:
        print(f"📦 Model loaded from {path}")
    return model



def save_full_model(model, scalers, path, metadata=None, verbose=True):
    """
    Save the full model package including model structure, weights, scalers, and optional metadata.

    Args:
        model (nn.Module): The PyTorch model to save.
        scalers (dict): Dictionary of scalers used for input/output preprocessing.
        path (str): The file path where the full model package will be saved.
        metadata (dict, optional): Additional metadata to save with the model. Default is None.
        verbose (bool, optional): If True, prints a confirmation message. Default is True.

    Behavior:
        Creates the directory for the path if it does not exist.
        Saves a checkpoint dictionary containing model state_dict, input dimension, activation,
        scalers, and metadata into a single .pt file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.net[0].in_features,
        'activation': model.net[2].__class__.__name__.lower(),
        'scalers': scalers,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)
    if verbose:
        print(f"✅ Full model (with scalers) saved to {path}")


def load_full_model(path, map_location=None, verbose=True):
    """
    Load a full model package including model structure, weights, scalers, and metadata.

    Args:
        path (str): Path to the saved full model package (.pt file).
        map_location (str or torch.device, optional): Device mapping for loading the model. Default is None.
        verbose (bool, optional): If True, prints a confirmation message. Default is True.

    Returns:
        tuple:
            - model (nn.Module): The loaded MLPRegressor model with weights.
            - scalers (dict): The scalers used during training.
            - metadata (dict): Optional metadata stored with the model.
    """
    checkpoint = torch.load(path, map_location=map_location)
    activation = checkpoint.get('activation', 'swish')
    input_dim = checkpoint.get('input_dim', 6)
    model = MLPRegressor(activation=activation, input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    if verbose:
        print(f"📦 Loaded model from {path} (input_dim={input_dim}, act={activation})")
    return model, checkpoint['scalers'], checkpoint.get('metadata', {})


