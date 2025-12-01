import json
import torch
from pathlib import Path

class DataNormalizer:
    """Normalizes tensors using min-max normalization parameters from a JSON file."""
    
    def __init__(self, json_path:Path):
        """
        Initialize normalizer with min-max stats from JSON file.
        
        Args:
            json_path: Path to JSON file containing 'min' and 'max' lists of size 13
        """
        with open(json_path, 'r') as f:
            stats = json.load(f)
        
        self.min_vals = torch.tensor(stats['mins'], dtype=torch.float32)
        self.max_vals = torch.tensor(stats['maxs'], dtype=torch.float32)
        self.range = self.max_vals - self.min_vals
    
    def __call__(self, tensor):
        """
        Normalize tensor of shape (13, H, W) using min-max normalization.
        
        Args:
            tensor: Tensor of shape (13, H, W)
            
        Returns:
            Normalized tensor with values in [0, 1]
        """
        # Reshape min/max to (13, 1, 1) for broadcasting
        min_vals = self.min_vals.view(13, 1, 1)
        range_vals = self.range.view(13, 1, 1)
        
        normalized = (tensor - min_vals) / range_vals
        return normalized
    
    def denormalize(self, tensor):
        """
        Reverse min-max normalization.
        
        Args:
            tensor: Normalized tensor of shape (13, H, W)
            
        Returns:
            Denormalized tensor
        """
        min_vals = self.min_vals.view(13, 1, 1)
        range_vals = self.range.view(13, 1, 1)
        
        return tensor * range_vals + min_vals