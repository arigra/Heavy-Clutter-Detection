import torch

def calculate_ratio(train_loader, detection_type):
    """Calculate ratio of positive samples in dataset from 2D labels

    Args:
        train_loader: DataLoader for the training set
        detection_type: "range" or "doppler"
    
    Returns:
        ratio: The fraction of bins that contain targets
    """
    n1 = 0  # Count of bins with targets
    n_total = len(train_loader.dataset) * 64  # Adjust if your range-doppler map size is different

    for _, _, _, _, _, rd_label in train_loader:
        # rd_label is [B, dR, dV]
        # If detection_type == "range", sum over Doppler dimension (-1)
        # If detection_type == "doppler", sum over Range dimension (-2)
        
        if detection_type == "range":
            # Sum over Doppler dimension (last dimension)
            label = (rd_label.sum(dim=-1) >= 1).float()
        else:
            # Sum over Range dimension (second last dimension)
            label = (rd_label.sum(dim=-2) >= 1).float()

        # Count how many bins are positive
        n1 += torch.sum(label >= 0.9999)

    ratio = n1.item() / n_total
    print("ratio:", ratio, ", n1:", n1.item(), ", n_total:", n_total)
    return ratio


class CBBCE(nn.Module):
    def __init__(self, ratio: float, beta: float = 0.99):
        """
        Class-Balanced Binary Cross Entropy Loss for logits.
        
        Args:
            ratio: Ratio of positive samples (e.g., #positives / total_samples).
            beta: Beta parameter for class balancing.
        """
        super().__init__()
        self.weight1 = (1 - beta) / (1 - beta ** ratio)
        print("CBBCE weight for positives: ", self.weight1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: [B, N*K] logits
            y_true: [B, N*K] target (0 or 1)
        """
        # Compute element-wise BCE with logits
        _nll2 = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        
        # Identify positive samples
        # y_true is either 0 or 1. If exact 1 may be used:
        ind1 = (y_true >= 0.9999).nonzero(as_tuple=False)  # Indices of positives
        
        # Weight the positive samples
        if ind1.numel() > 0:
            _nll_subset = self.weight1 * _nll2[ind1[:, 0], ind1[:, 1]]
            _nll2 = _nll2.index_put_((ind1[:, 0], ind1[:, 1]), _nll_subset)
        
        # Mean loss
        loss = torch.mean(_nll2)
        return loss
