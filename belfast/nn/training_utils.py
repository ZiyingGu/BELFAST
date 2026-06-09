import time
import os
import logging
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from belfast.nn.mlp_regressor import MLPRegressor, save_full_model
from belfast.nn.loss_utils import get_loss_function
from belfast.nn.evaluate_model import evaluate_model_deviation, plot_loss_by_deviation_bins


def make_unique_folder(base_path):
    """
    Create a unique folder based on the given base path.
    If the folder already exists, appends a number to the folder name to make it unique.

    Args:
        base_path (str): The desired base directory path.

    Returns:
        str: The path to the created unique folder.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path

    count = 2
    while True:
        new_path = f"{base_path} ({count})"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        count += 1



def setup_logger(saving_folder, log_name="run.log"):
    """
    Set up a logger that writes to both a file and the console.

    Args:
        saving_folder (str): Directory to save the log file.
        log_name (str): Name of the log file (default is 'run.log').

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(saving_folder, exist_ok=True)
    log_path = os.path.join(saving_folder, log_name)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear old handlers to prevent duplicate output
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



class SolutionQueue:
    """
    A queue that keeps track of solution/model checkpoint files, with a maximum size.
    When the queue exceeds max_size, the oldest file is deleted from disk.
    """
    
    def __init__(self, logger, max_size):
        """
        Initialize the SolutionQueue.

        Args:
            logger (logging.Logger): Logger for logging actions.
            max_size (int): Maximum number of files to keep in the queue.
        """
        self.logger = logger
        self.max_size = max_size
        self.queue = deque() 

    def add(self, filepath):
        """
        Add a file path to the queue and delete the oldest file if the queue exceeds max_size.

        Args:
            filepath (str): Path to the file to add.
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"  File does not exist: {filepath}")

        self.queue.append(filepath)
        self.logger.info(f"    Added: {filepath}")

        if len(self.queue) > self.max_size:
            oldest = self.queue.popleft()
            try:
                os.remove(oldest)
                self.logger.info(f"    Deleted oldest file: {oldest}")
            except Exception as e:
                self.logger.info(f"    Failed to delete {oldest}: {e}")

    def get_all_files(self):
        """
        Get a list of all file paths currently in the queue.

        Returns:
            list: List of file paths.
        """
        return list(self.queue)

    def get_latest(self):
        """
        Return the most recently added file path, or None if the queue is empty.

        Returns:
            str or None: Latest file path or None if queue is empty.
        """
        if self.queue:
            return self.queue[-1]
        else:
            return None



class Training():
    """
    Class for training a neural network model for age estimation.
    Handles model initialization, training, validation, checkpointing, and evaluation.
    """
    def __init__(self, 
                 target_name,
                 bands_list,
                 age_type,
                 cluster_name,
                 sigma_deviation,
                 X_train, 
                 Y_train, 
                 X_val, 
                 Y_val,
                 X_test,
                 Y_test,
                 x_near_target,
                 y_near_target,
                 scalers,
                #  input_dim=6,
                 loss='softdigits',
                 activation='swish',
                 epochs=300, 
                 batch_size=1024*10, 
                 learning_rate=1e-3,
                 weight_decay=1e-5,
                 mode='min', 
                 checkpoints_max_size=5,
                 lr_factor=0.5, 
                 lr_manual_flag=False,
                 lr_auto_patience=10, 
                 verbose=True, 
                 min_lr=1e-7,
                 shuffle=True,
                 ):
        """
        Initialize the Training class with data, hyperparameters, and configuration.

        Args:
            target_name (str): Name of the target object.
            bands_list (list): List of feature bands used.
            age_type (str): Type of age estimation.
            cluster_name (str): Name of the cluster.
            sigma_deviation (float): Sigma deviation for evaluation.
            X_train, Y_train, X_val, Y_val, X_test, Y_test (torch.Tensor): Training, validation, and test datasets.
            x_near_target, y_near_target (torch.Tensor): Data near the target for deviation calculation.
            scalers (dict): Dictionary of scalers for input/output.
            loss (str): Loss function name.
            activation (str): Activation function name.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for optimizer.
            mode (str): Mode for learning rate scheduler ('min' or 'max').
            checkpoints_max_size (int): Maximum number of checkpoints to keep.
            lr_factor (float): Learning rate reduction factor.
            lr_manual_flag (bool): Whether to manually reduce learning rate.
            lr_auto_patience (int): Patience for automatic LR scheduler.
            verbose (bool): If True, print verbose logs.
            min_lr (float): Minimum learning rate for scheduler.
            shuffle (bool): Whether to shuffle training data.
        """
        self.target_name = target_name
        self.bands_list = bands_list 
        self.age_type = age_type
        self.cluster_name = cluster_name
        self.sigma_deviation = sigma_deviation
        self.checkpoints_max_size = checkpoints_max_size

        # ✅ Check if GPU is available and set the computation device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # self.model = MLPRegressor(input_dim=input_dim, activation=activation).to(device)
        # self.input_dim = input_dim
        self.model = MLPRegressor(activation=activation).to(device)
        self.loss = loss
        self.activation=activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mode = mode
        self.lr_factor = lr_factor
        self.lr_manual_flag = lr_manual_flag
        self.lr_auto_patience = lr_auto_patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.x_near_target = x_near_target
        self.y_near_target = y_near_target
        self.scalers = scalers
        self.shuffle = shuffle

        # Optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer = optimizer
        
        if lr_manual_flag == False:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=self.mode, factor=self.lr_factor, patience=self.lr_auto_patience, 
                verbose=self.verbose, min_lr=self.min_lr
            )
            self.scheduler = scheduler

        # Loss function (without weights)
        criterion = get_loss_function(self.loss, digits=3)
        self.criterion = criterion

        # Prepare training loader
        train_dataset = TensorDataset(self.X_train, self.Y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.train_loader = train_loader

        # Track training losses
        self.train_losses = []

        # Manually calculate the number of batches
        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / self.batch_size))

        print(f"Training parameters: {self.epochs} epochs, {num_batches} batches per epoch")
    
    def reduce_lr(self, lr_factor):
        """
        Manually reduce the learning rate by the given factor, down to the minimum learning rate.

        Args:
            lr_factor (float): Factor to multiply the current learning rate by.
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * lr_factor, self.min_lr)
            param_group['lr'] = new_lr            
            self.logger.info(f"               Manually reduced LR: {old_lr:.2e} → {new_lr:.2e}")


    def train(self, improve_threshold = 0.05, max_no_improve=20, early_stop_start_epoch=100, early_stop_lr=1e-6,
              lr_improve_threshold = 0.05, lr_patience=10):
        """
        Train the neural network model with early stopping and learning rate scheduling.

        Args:
            improve_threshold (float): Minimum relative improvement to avoid early stopping.
            max_no_improve (int): Max epochs without sufficient improvement before stopping.
            early_stop_start_epoch (int): Epoch to start checking for early stopping.
            early_stop_lr (float): Learning rate threshold for early stopping.
            lr_improve_threshold (float): Minimum improvement to avoid reducing learning rate.
            lr_patience (int): Number of epochs to wait before reducing learning rate if no improvement.
        """
        self.saving_folder = make_unique_folder(
            base_path=f"Neural_Network_Model/checkpoints_{self.age_type}_age{self.cluster_name}_{self.target_name.replace(' ', '_')}"
        )

        # set logger
        logger = setup_logger(self.saving_folder, log_name=f"{self.age_type}_age{self.cluster_name}_{self.target_name.replace(' ', '_')}_training.log")
        logger.info(f"📝 Checkpoints of your trained NN model will be saved at: \n {os.getcwd()+'/'+self.saving_folder}")
        logger.info("🚀 Starting training: ...")

        checkpoint_queue = SolutionQueue(logger=logger, max_size=self.checkpoints_max_size)

        # Initialize tracking
        train_losses, val_losses, dev_list = [], [], []
        best_val_loss = float('inf')
        prev_time = time.time()

        # Early stopping configuration
        self.early_stop_start_epoch = early_stop_start_epoch
        self.improve_threshold = improve_threshold
        self.max_no_improve = max_no_improve
        self.early_stop_lr = early_stop_lr
        no_improve_count = 0

        # For learning rate scheduler enhancement
        lr_patience_counter = 0

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_train_loss = epoch_loss / len(self.train_loader.dataset)
            train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(self.X_val)
                val_loss = self.criterion(val_preds, self.Y_val).item()
                val_losses.append(val_loss)

                y_test_pred_scaled = self.model(self.x_near_target).cpu().numpy()
            y_scaler = self.scalers['y']
            y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)
            y_test_true = y_scaler.inverse_transform(self.y_near_target.cpu().numpy())
            deviation = y_test_pred - y_test_true
            # dev_list_i = np.std(deviation.flatten())
            dev_list_i = np.max([np.abs(np.percentile(deviation.flatten(), 15.865)), \
                                 np.abs(np.percentile(deviation.flatten(), 84.135))])
            dev_list.append(dev_list_i)
            if self.lr_manual_flag == False:
                self.scheduler.step(val_loss)
            else:
                # Manually control LR reduction
                if len(val_losses) > 1:
                    current_lowest_val_losses = min(val_losses[:-1])
                    improvement = (current_lowest_val_losses - val_loss) / abs(current_lowest_val_losses + 1e-8)

                    if improvement < lr_improve_threshold:
                        lr_patience_counter += 1
                        # logger.info(f"🕒 LR Scheduler: No sufficient improvement ({improvement:.4f}) for {lr_patience_counter}/{lr_patience} epochs.")
                    else:
                        lr_patience_counter = 0  # reset on improvement

                    if lr_patience_counter >= lr_patience:
                        if self.optimizer.param_groups[0]['lr'] != self.min_lr:
                            logger.info(f"🔽 Epoch {epoch+1:03d} | Triggering manual LR reduction due to insufficient improvement.")
                            self.reduce_lr(lr_factor=self.lr_factor)
                        lr_patience_counter = 0

            # Save best model
            if epoch + 1 > self.early_stop_start_epoch and val_loss < best_val_loss:
                best_val_loss = val_loss
                check_point_name = f"{self.saving_folder}/Best_model{self.cluster_name}_{self.target_name.replace(' ', '_')}_val{val_loss:.6f}_dev{dev_list_i:.6f}.pt"
                torch.save(self.model.state_dict(), check_point_name)
                logger.info(f"💾 Saved best model at epoch {epoch+1:03d} | Val Loss: {val_loss:.6f} | Target Dev: {dev_list_i:.6f}")
                checkpoint_queue.add(check_point_name)

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                now = time.time()
                elapsed = now - prev_time
                prev_time = now
                logger.info(f"📉 Epoch {epoch+1:03d} | Time: {elapsed:.2f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} "
                            f"| Target Dev: {dev_list_i:.6f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping based on relative improvement
            if epoch + 1 > self.early_stop_start_epoch and self.optimizer.param_groups[0]['lr'] < self.early_stop_lr:
                relative_improvement = (best_val_loss - val_loss) / best_val_loss
                if relative_improvement < improve_threshold and self.optimizer.param_groups[0]['lr'] == self.min_lr:
                    if no_improve_count == 0:
                        logger.info(f"🛑 Reaching the lowest limit of the learning rate.")
                        logger.info(f"📍 Starting to check the improvemrnt for early stopping.")
                    no_improve_count += 1
                    if (no_improve_count) % 10 == 0:
                        logger.info(f"⚠️ No sufficient improvement: {relative_improvement:.4f} < {improve_threshold} "
                                    f"({no_improve_count}/{max_no_improve})")
                else:
                    no_improve_count = 0  # reset if improved enough

                if no_improve_count >= max_no_improve:
                    logger.info(f"🛑 Early stopping triggered at epoch {epoch+1:03d} | Best Val Loss: {best_val_loss:.6f}")
                    logger.info(f"💾 The best model checkpoint has been saved at: \n {self.checkpoint_queue.get_latest()}")
                    break

            self.logger = logger
            self.train_losses = train_losses
            self.val_losses = val_losses
            self.dev_list = dev_list
            self.checkpoint_queue = checkpoint_queue



    def plot_loss_dev(self, figsize=None, loss_ylim=None, dev_ylim=None):
        """
        Plot training and validation loss, as well as deviation, over epochs.

        Args:
            figsize (tuple): Size of the plot figure.
            loss_ylim (list): Y-axis limits for the loss plot.
            dev_ylim (list): Y-axis limits for the deviation plot.
        """
        import matplotlib.pyplot as plt

        if figsize is None:
            figsize = (10, 8)
        if loss_ylim is None:
            loss_ylim = [-0.1, 500]
        if dev_ylim is None:
            dev_ylim = [0, max(self.dev_list) * 1.2]

        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # The first subplot: Loss
        axs[0].plot(self.train_losses, label='Train Loss')
        axs[0].plot(self.val_losses, label='Validation Loss')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training vs Validation Loss')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_ylim(loss_ylim)

        # The second subplot: Dev
        axs[1].plot(self.dev_list, label=r'Deviation (1-$\sigma$ percentile)', color='purple')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Deviation')
        axs[1].set_title('Deviation over Epochs')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_ylim(dev_ylim)

        plt.tight_layout()
        plt.show()


    def run_test_set(self):
        """
        Evaluate the latest model checkpoint on the test set and print the test loss.
        """
        model_path = self.checkpoint_queue.get_latest()
        model = MLPRegressor(activation=self.activation).to(self.device)
        model.load_state_dict(torch.load(model_path))

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_output = model(self.X_test)
            test_loss = self.criterion(test_output, self.Y_test).item()
        print(f"🧪 Final test loss: {test_loss:.6f}")
 
    def evaluate(self):
        """
        Evaluate the trained model on data near the target and save the evaluation plot.
        """
        model_path = self.checkpoint_queue.get_latest()
        output_name = f"{os.getcwd()}/Neural_Network_Model/trained_model/NN_model_evaluation_{self.target_name.replace(' ', '_')}{self.cluster_name}_{self.age_type}_age_deviation{self.sigma_deviation}_input{self.X_train.shape[1]}_output{self.Y_train.shape[1]}.pdf"
        self.logger.info(f"🎯 The evaluation of your trained model is save at: \n {output_name}")
        self.dev_near_target = evaluate_model_deviation(model_path, self.x_near_target, self.y_near_target, self.scalers['y'], self.bands_list,
                                                        output_name=output_name)

    # save the full NN model (the checkpoint and the scalers)
    def save_NN_model(self):
        """
        Save the full neural network model along with scalers and metadata to disk.
        """
        NN_saved_file = f"Neural_Network_Model/trained_model/NN_model_{self.target_name.replace(' ', '_')}{self.cluster_name}_{self.age_type}_age_deviation{self.sigma_deviation}_input{self.X_train.shape[1]}_output{self.Y_train.shape[1]}.pt"
        self.logger.info(f"📦 Your trained model together scalers is saved at: \n {os.getcwd()+'/Neural_Network_Model/'+NN_saved_file}")
        save_full_model(self.model, 
                        self.scalers, 
                        NN_saved_file,
                        verbose=False,
                        metadata={f"note': '{self.X_train.shape[1]}-input and {self.Y_train.shape[1]}-output trained model for the {self.target_name} with {self.age_type} ages"})
