import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


class RadarPreprocessing(nn.Module):
    def __init__(self, detection_type: str = "range"):
        """
        Preprocessing module as described in Section III.B

        Args:
            detection_type: Either "range" or "doppler"
        """
        super().__init__()
        self.detection_type = detection_type

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (batch, N, K, 2) - real and imaginary parts stacked in last dimension
        # Step 1: Reshape input based on detection type
        if self.detection_type == "range":
            X_0 = X.transpose(1, 2)  # (batch, K, N)
        else:  # doppler
            X_0 = X  # (batch, N, K)
        # Step 2: Center features over rows
        X_mean = torch.mean(X_0, dim=1, keepdim=True)
        X_1 = X_0 - X_mean
        # Step 3: Concatenate real and imaginary terms
        if torch.is_complex(X):

            Z_0 = torch.cat([X_1.real, X_1.imag], dim=-1)  # (batch, K/N, 2*N/K)
        else:
            Z_0 = torch.cat([X_1, torch.zeros_like(X_1)], dim=-1)
        return Z_0
    

class DAFCBlock(nn.Module):
    def __init__(self, H: int, W: int, H_out: int, W_out: int):
        """
        Dimensional-Alternating Fully Connected Block

        Args:
            H: Input height
            W: Input width
            H_out: Output height
            W_out: Output width
        """
        super().__init__()

        # Row mapping layer (F transform)
        self.row_fc = nn.Linear(W, W_out)

        # Column mapping layer (G transform)
        self.col_fc = nn.Linear(H, H_out)

        # Activation functions
        self.hr = nn.Tanh()
        self.hc = nn.Tanh()

    def forward(self, Z_in: torch.Tensor) -> torch.Tensor:
        Z_in = Z_in.float()
        batch_size = Z_in.size(0)
        H = Z_in.size(1)

        # Step 1: FC transform of each row
        # Reshaping is necessary to apply nn.Linear
        Z_reshaped = Z_in.reshape(-1, Z_in.size(-1))  # (batch*H, W)

        Z_r = self.hr(self.row_fc(Z_reshaped))  # (batch*H, W_out)

        # Reshaping back to normal after applying nn.Linear
        Z_r = Z_r.reshape(batch_size, H, -1)  # (batch, H, W_out)

        # Step 2: FC transform of each column
        Z_r = Z_r.transpose(1, 2)  # (batch, W_out, H)
        Z_r = Z_r.contiguous()  # Make memory contiguous after transpose

        # Reshaping is necessary to apply nn.Linear
        Z_reshaped = Z_r.reshape(-1, Z_r.size(-1))  # (batch*W_out, H)

        Z_out = self.hc(self.col_fc(Z_reshaped))  # (batch*W_out, H_out)

        # Reshaping back to normal after applying nn.Linear
        Z_out = Z_out.reshape(batch_size, -1, Z_out.size(-1))  # (batch, W_out, H_out)

        Z_out = Z_out.transpose(1, 2)  # (batch, H_out, W_out)

        return Z_out


class DAFCRadarNet(nn.Module):
    def __init__(self, detection_type: str = "range", map_type: str = "IQ"):
        """
        DAFC-based Radar Detection Network

        Args:
            detection_type: Either "range" or "doppler"
        """
        super().__init__()

        # Parameters from Table I
        
        if map_type == "IQ":
            if detection_type == "range":
                self.input_dim = (64, 128)  # (K, 2*N)
            else:  # doppler
                self.input_dim = (64, 128)  # (N, 2*K)
        else: # RD
            if detection_type == "range":
                self.input_dim = (64, 128)  # (K, 2*N)
            else:  # doppler
                self.input_dim = (64, 128)  # (N, 2*K)

        # Preprocessing
        self.preprocess = RadarPreprocessing(detection_type)

        # DAFC blocks (dimensions from Table I)
        self.dafc1 = DAFCBlock(self.input_dim[0], self.input_dim[1], 128, 1024)
        self.dafc2 = DAFCBlock(128, 1024, 16, 256)
        self.dafc3 = DAFCBlock(16, 256, 4, 128)

        # Final FC layer
        fc_input_size = 4 * 128
        self.fc = nn.Linear(fc_input_size, 64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Preprocessing
        Z_0 = self.preprocess(X)
        # DAFC blocks
        Z_1 = self.dafc1(Z_0)
        Z_2 = self.dafc2(Z_1)
        Z_3 = self.dafc3(Z_2)

        # Flatten and final FC layer
        Z_flat = Z_3.reshape(Z_3.size(0), -1)
        y = self.sigmoid(self.fc(Z_flat))

        return y
    

class CBBCE(nn.Module):
    def __init__(self, ratio: float, beta: float = 0.99):
        """
        Class-Balanced Binary Cross Entropy Loss

        Args:
            ratio: Ratio of positive samples in dataset
            (number of targets in the dataset / number of bins in the dataset)
            beta: Beta parameter for class balancing
        """
        super().__init__()
        self.weight1 = (1 - beta) / (1 - beta ** ratio)
        print("w1: ", self.weight1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            y_pred: Predicted values [batch, dim] (after sigmoid)
            y_true: True values [batch, dim]
        """
        _nll2 = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        # Find indices where y_true >= 0.9999
        ind1 = (y_true >= 0.9999).nonzero(as_tuple=False)  # Shape: [N, 2]

        # Gather the loss values at these indices
        _nll_subset = self.weight1 * _nll2[ind1[:, 0], ind1[:, 1]]

        # Update the original loss tensor using scatter_
        _nll2 = _nll2.index_put_((ind1[:, 0], ind1[:, 1]), _nll_subset)

        # Return the mean of the updated loss tensor
        loss = torch.mean(_nll2)
        return loss

def calculate_ratio(train_loader, detection_type):
    """Calculate ratio of positive samples in dataset from 2D labels"""
    n1 = 0  # target samples
    n_total = len(train_loader.dataset)
    n_total *= 32 if detection_type == 'range' else 63
    #for _, rd_label in train_loader:
    for _, _, _, _, _, rd_label in train_loader:
    
        # Get 1D labels by summing across appropriate dimension
        label = (rd_label.sum(dim=-1 if detection_type == "range" else -2) >= 1).float()
        # Count bins with targets
        n1 += torch.sum(label >= 0.9999)
    ratio = n1.item() / n_total
    print("ratio:", ratio, ", n1:", n1.item(), ", n_total:", n_total)
    return ratio


def plot_training_history(history: dict, detection_type: str, map_type: str):
    """Plot and save training history for a given detector type.

    Parameters
    ----------
    history : dict
        A dictionary containing keys "train_loss", "val_loss", 
        "train_accuracy", and "val_accuracy", each mapping to a list 
        of values recorded per epoch.
    detection_type : str
        A string indicating the type/category of the detector 
        (e.g., 'Target', 'Object', 'Clutter').
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{detection_type} Detector Training History", fontsize=16, fontweight='bold')

    # Define a style
    #plt.style.use('seaborn')

    # Plot Loss
    ax1.plot(history["train_loss"], label="Training Loss", color="steelblue", linewidth=1)
    ax1.plot(history["val_loss"], label="Validation Loss", color="firebrick", linewidth=1)
    ax1.set_title("Loss per Epoch", fontsize=14)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot Accuracy
    ax2.plot(history["train_accuracy"], label="Training Accuracy", color="darkgreen", linewidth=1)
    ax2.plot(history["val_accuracy"], label="Validation Accuracy", color="darkorange", linewidth=1)
    ax2.set_title("Accuracy per Epoch", fontsize=14)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    if map_type == "RD":
        plt.savefig(f"{detection_type}_RD_training_history.png", dpi=300)
    else:
        plt.savefig(f"{detection_type}_IQ_training_history.png", dpi=300)
    plt.close()


def train_model(model, criterion, train_loader, val_loader, detection_type, map_type, epochs=300, learning_rate=1e-3,
                weight_decay=5e-4):
    """
    Train range or Doppler detector

    Args:
        model: Neural network model
        criterion: Loss function
        train_loader: Training data loader
        val_loader: Validation data loader
        detection_type: "range" or "doppler"
        map_type: "IQ" or "RD"
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
    """

    # Initialize #wandb
    #wandb.init(
    #     project="radar_detection",  # replace with your project name
    #     name=f"{detection_type}_{map_type}_run",
    #     config={
    #         "detection_type": detection_type,
    #         "map_type": map_type,
    #         "epochs": epochs,
    #         "learning_rate": learning_rate,
    #         "weight_decay": weight_decay,
    #         "optimizer": "Adam"
    #     },
    #     tags=[detection_type, map_type]
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.905, patience=5)
    # Check the last computed learning rate
    print(f"Last computed LR: {scheduler.get_last_lr()[0]}")

    # Training history
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")

    if map_type == "IQ":
        print('Training IQ network')
        for epoch in epoch_pbar:
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for _, _, X, _, _, rd_label in train_loader:
                X = X.to(device)
                rd_label = rd_label.to(device)

                # Preprocess input and get 1D label
                label = (rd_label.sum(dim=-1 if detection_type == "range" else -2) >= 1).float()

                # Forward pass
                outputs = model(X)
                loss = criterion(outputs, label)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == label).float().sum().item()
                train_total += label.numel()

            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for _, _, X, _, _, rd_label in val_loader:
                    X = X.to(device)
                    rd_label = rd_label.to(device)

                    # Preprocess input and get 1D label
                    label = (rd_label.sum(dim=-1 if detection_type == "range" else -2) >= 1).float()

                    outputs = model(X)
                    loss = criterion(outputs, label)

                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == label).float().sum().item()
                    val_total += label.numel()

            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = train_correct / train_total
            epoch_val_acc = val_correct / val_total

            # Update history
            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["train_accuracy"].append(epoch_train_acc)
            history["val_accuracy"].append(epoch_val_acc)

            # Log metrics to #wandb
            #wandb.log({
            #     "epoch": epoch + 1,
            #     "train_loss": epoch_train_loss,
            #     "val_loss": epoch_val_loss,
            #     "train_accuracy": epoch_train_acc,
            #     "val_accuracy": epoch_val_acc,
            #     "learning_rate": scheduler.optimizer.param_groups[0]["lr"]
            # })

            # Learning rate scheduling
            scheduler.step(epoch_val_loss)

            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

            # Print epoch results every 10 epochs
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
                print(f"Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save model
        torch.save(model.state_dict(), f"{detection_type}_64_{map_type}_IQ_model.pt")



    return history


def train_range_dafc(dafc_train_loader, dafc_val_loader):
    print("\nTraining IQ Range Detector:")
    map_type = "IQ"
    detection_type = "range"
    ratio = calculate_ratio(dafc_train_loader, detection_type)
    criterion = CBBCE(ratio)
    range_model = DAFCRadarNet(detection_type, map_type)
    range_history = train_model(range_model, criterion, dafc_train_loader, dafc_val_loader, detection_type, map_type)
    plot_training_history(range_history, "Range", map_type)

def train_doppler_dafc(dafc_train_loader, dafc_val_loader):
    map_type = "IQ"
    detection_type = "doppler"
    ratio = calculate_ratio(dafc_train_loader, detection_type)
    criterion = CBBCE(ratio)
    print("\nTraining IQ Doppler Detector:")
    doppler_model = DAFCRadarNet(detection_type)
    doppler_history = train_model(doppler_model, criterion, dafc_train_loader, dafc_val_loader, detection_type, map_type)
    plot_training_history(doppler_history, "Doppler", map_type)