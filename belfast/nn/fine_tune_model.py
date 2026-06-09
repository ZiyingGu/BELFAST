import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Neural_Network_Model.mlp_regressor import MLPRegressor
from Neural_Network_Model.loss_utils import get_loss_function
import time

def fine_tune_model(
    model_path,
    X_train_tensor,
    Y_train_tensor,
    X_val_tensor,
    Y_val_tensor,
    activation='swish',
    loss_name='softdigits',
    loss_kwargs={'digits': 3},
    freeze_layers=None,
    epochs=50,
    batch_size=4096,
    lr=1e-6,
    per_epochs=5,
    device=None,
    verbose=True
):
    """
    Fine-tunes a pre-trained MLPRegressor model on new training data, optionally freezing certain layers.

    This function loads a pre-trained model from a given path, optionally freezes the last `freeze_layers` layers,
    and fine-tunes the model on the provided training data using the specified loss function and optimizer settings.
    The function tracks and returns training and validation losses over epochs, saving the best model (lowest validation loss).

    Args:
        model_path (str): Path to the pre-trained model weights (.pt file).
        X_train_tensor (torch.Tensor): Training input features tensor.
        Y_train_tensor (torch.Tensor): Training target values tensor.
        X_val_tensor (torch.Tensor): Validation input features tensor.
        Y_val_tensor (torch.Tensor): Validation target values tensor.
        activation (str, optional): Activation function to use in the model. Default is 'swish'.
        loss_name (str, optional): Name of the loss function to use. Default is 'softdigits'.
        loss_kwargs (dict, optional): Additional keyword arguments for the loss function. Default is {'digits': 3}.
        freeze_layers (int or None, optional): Number of last layers to freeze during fine-tuning. If None, no layers are frozen. Default is None.
        epochs (int, optional): Number of training epochs. Default is 50.
        batch_size (int, optional): Batch size for training. Default is 4096.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-6.
        per_epochs (int, optional): Frequency (in epochs) to print progress. Default is 5.
        device (torch.device or None, optional): Device to use for training ('cuda' or 'cpu'). If None, automatically selects CUDA if available. Default is None.
        verbose (bool, optional): Whether to print progress information during training. Default is True.

    Returns:
        model (MLPRegressor): The fine-tuned model.
        train_losses (list of float): List of average training losses per epoch.
        val_losses (list of float): List of validation losses per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPRegressor(activation=activation).to(device)
    model.load_state_dict(torch.load(model_path))

    # Freeze all layers except the last several layers
    if freeze_layers is not None:
        for param in list(model.parameters())[-freeze_layers:]:
            param.requires_grad = False

    criterion = get_loss_function(loss_name, **loss_kwargs)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)

    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batch_size, shuffle=True)
    num_samples = len(X_train_tensor)

    best_val_loss = float('inf')
    prev_print_time = time.time()

    train_losses = []
    val_losses = []

    print(f"🔧 Fine-tuning last layers: {epochs} epochs, {len(train_loader)} batches per epoch")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        avg_loss = epoch_loss / num_samples
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, Y_val_tensor).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"Neural_Network_Model/checkpoints/best_last4_val{val_loss:.6f}.pt")
            print(f"💾 Saved best model at epoch {epoch+1:03d} | Val Loss: {val_loss:.6f}")

        scheduler.step(val_loss)

        if verbose and ((epoch + 1) % per_epochs == 0 or epoch == 0):
            now = time.time()
            delta_time = now - prev_print_time
            prev_print_time = now
            current_lr = optimizer.param_groups[0]['lr']
            print(f"📉 Epoch {epoch+1:03d}/{epochs} | Time: {delta_time:.2f}s | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

    return model, train_losses, val_losses

