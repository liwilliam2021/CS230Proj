"""
Utility functions for the project.
"""
import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def create_experiment_dir(base_dir, layer_num):
    """Create directory for experiment at specific layer."""
    exp_dir = os.path.join(base_dir, f'layer_{layer_num}')
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

