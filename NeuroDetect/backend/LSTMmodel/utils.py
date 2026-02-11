"""
Utility classes and functions for LSTM fraud detection
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focal Loss applies a modulating term to the cross entropy loss in order to focus learning 
    on hard misclassified examples. It's particularly useful for imbalanced datasets.
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive vs negative examples
        gamma: Focusing parameter for modulating loss. Higher gamma reduces loss for well-classified examples
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal Loss calculation
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    
    Args:
        patience: How many epochs to wait after last time validation loss improved
        min_delta: Minimum change in monitored quantity to qualify as an improvement
        verbose: If True, prints a message for each validation loss improvement
    """
    def __init__(self, patience=10, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'  Validation loss decreased ({self.best_loss:.6f}). Saving model...')
        self.best_model = model.state_dict().copy()


def create_weighted_sampler(y_train):
    """
    Create weighted sampler for imbalanced classes
    
    This sampler oversamples the minority class (fraud) to balance the training batches
    
    Args:
        y_train: Training labels (numpy array or tensor)
        
    Returns:
        WeightedRandomSampler instance
    """
    # Convert to numpy if tensor
    if torch.is_tensor(y_train):
        y_train = y_train.cpu().numpy()
    
    # Ensure it's 1D
    if len(y_train.shape) > 1:
        y_train = y_train.flatten()
    
    # Calculate class weights
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def calculate_class_weights(y_train):
    """
    Calculate class weights for loss function
    
    Args:
        y_train: Training labels
        
    Returns:
        pos_weight: Weight for positive class (fraud)
    """
    # Convert to numpy if tensor
    if torch.is_tensor(y_train):
        y_train = y_train.cpu().numpy()
    
    # Ensure it's 1D
    if len(y_train.shape) > 1:
        y_train = y_train.flatten()
    
    # Calculate positive class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    
    if n_pos == 0:
        return torch.tensor([1.0])
    
    pos_weight = torch.tensor([n_neg / n_pos])
    
    return pos_weight
