import torch
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from Neural_Network_Model.mlp_regressor import load_model


def evaluate_model_deviation(
    model_path,
    x_tensor,
    y_tensor,
    y_scaler,
    bands_list,
    input_dim=6,
    bins=800,
    xlim=None,
    figsize=None,
    alpha=0.6,
    title="Prediction Deviation",
    device=None,
    output_name=False,
):
    """
    Load a trained model, make predictions on given input tensors, compute deviations
    from the ground truth, and plot their distribution with a Gaussian fit.
    
    Args:
        model_path (str): Path to the saved model state dict.
        x_tensor (torch.Tensor): Normalized input features.
        y_tensor (torch.Tensor): Normalized ground-truth outputs.
        y_scaler (sklearn.MinMaxScaler): Scaler used for inverse-transforming outputs.
        bands_list (list of str): List of photometric bands or output labels.
        input_dim (int): Dimension of input features expected by the model.
        bins (int): Number of bins for the histogram.
        xlim (list, optional): X-axis limits for the histogram plot. Default [-0.04, 0.04].
        figsize (tuple, optional): Figure size for the plot. Default (10, 5).
        alpha (float): Transparency level for histogram bars.
        title (str): Title string for the plot.
        device (torch.device, optional): Device to run model inference. Defaults to GPU if available.
        output_name (str or bool): If provided as string, save the plot to this path; if False, do not save.

    Returns:
        np.ndarray: Array of deviations (prediction - truth) with shape same as y_tensor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if xlim is None:
        xlim = [-0.04,0.04]
    if figsize is None:
        figsize = (10, 5)

    # 1. Load model and switch to eval mode
    model = load_model(model_path, input_dim=input_dim).to(device)
    model.eval()

    # 2. Predict on input tensor
    with torch.no_grad():
        y_pred_scaled = model(x_tensor.to(device)).cpu().numpy()

    # 3. Inverse transform
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_tensor.cpu().numpy())

    # 4. Deviation computation
    deviation = y_pred - y_true

    # 5. Plot histogram
    plt.figure(figsize=figsize)
    plt.hist(deviation.flatten(), bins=bins, density=True, alpha=alpha, label="Deviation \nHistogram")

    # 6. Fit Gaussian and plot
    mean_dev = np.mean(deviation.flatten())
    # std_dev = np.std(deviation.flatten())
    std_dev = np.max([np.abs(np.percentile(deviation.flatten(), 15.865)), \
                      np.abs(np.percentile(deviation.flatten(), 84.135))])
    x_range = np.linspace(xlim[0], xlim[1], 1000)
    gauss_curve = norm.pdf(x_range, loc=mean_dev, scale=std_dev)
    plt.plot(x_range, gauss_curve, 'r-', label=f"Gaussian Fit\nmean={mean_dev:.5f}, std={std_dev:.5f}")
    
    plt.text(0.05, 0.95, r"data points near the \ntarget: mag deviation $<5\,\sigma_m$",
         transform=plt.gca().transAxes,  
         va='top', ha='left',            
         fontsize=11,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.legend()
    plt.xlabel("Prediction - Truth (mag)")
    plt.ylabel("Density")
    plt.title(f"Distribution of {title}")
    plt.grid(True)
    plt.xlim(xlim)

    if output_name != False:
        plt.savefig(output_name)
    plt.show()
    print(bands_list)
    print('Dev in each band: ', np.std(deviation, axis=0))

    return deviation



def plot_loss_by_deviation_bins(deviation_tensor, loss_tensor, bins=None, log_scale=False):
    """
    Plot the average loss values as a function of deviation bins.

    Args:
        deviation_tensor (torch.Tensor): 1D tensor of deviation values for each sample (shape [N]).
        loss_tensor (torch.Tensor): 1D tensor of loss values corresponding to each sample (shape [N]).
        bins (list or int, optional): Bin edges or number of bins. If None, defaults to [0, 1, 2, 3, 5, 10, 20, 50, 100].
        log_scale (bool): Whether to use a logarithmic scale for the Y axis.

    Returns:
        None. Displays a bar plot of average loss per deviation bin.
    """
    deviation = deviation_tensor.cpu().numpy()
    loss = loss_tensor.cpu().numpy()

    # Default bins if not specified
    if bins is None:
        bins = [0, 1, 2, 3, 5, 10, 20, 50, 100]

    bin_indices = np.digitize(deviation, bins)
    bin_means = []
    bin_counts = []

    for i in range(1, len(bins)):
        bin_loss = loss[bin_indices == i]
        if len(bin_loss) > 0:
            bin_means.append(np.mean(bin_loss))
            bin_counts.append(len(bin_loss))
        else:
            bin_means.append(0.0)
            bin_counts.append(0)

    # Format bin labels
    bin_labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]

    # Plot
    plt.figure(figsize=(10, 5), dpi=100)
    plt.bar(bin_labels, bin_means, color='steelblue')
    plt.xticks(rotation=45)
    plt.xlabel("Deviation Interval")
    plt.ylabel("Average Loss")
    plt.title("Loss Distribution Across Deviation Ranges")
    if log_scale:
        plt.yscale("log")
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()

