"""
Noise generation functions for synthetic heavy-tailed dataset experiments.

This module provides functions to generate different types of noise:
1. Standard Normal (light-tailed)
2. Component-wise symmetrized Pareto with p=2.5 (heavy-tailed, finite variance)
3. Component-wise symmetrized Pareto with p=1.5 (heavy-tailed, infinite variance)
"""

import torch
import numpy as np
from scipy.stats import pareto
from typing import Tuple, Union


def generate_normal_noise(shape: Union[Tuple, torch.Size], device: str = 'cpu') -> torch.Tensor:
    """
    Generate standard normal noise.
    
    Args:
        shape: Shape of the noise tensor
        device: Device to place the tensor on
        
    Returns:
        Tensor with standard normal noise
    """
    return torch.randn(shape, device=device)


def generate_pareto_noise(shape: Union[Tuple, torch.Size], p: float, device: str = 'cpu') -> torch.Tensor:
    """
    Generate component-wise symmetrized Pareto noise.
    
    Args:
        shape: Shape of the noise tensor
        p: Pareto distribution parameter (shape parameter)
        device: Device to place the tensor on
        
    Returns:
        Tensor with symmetrized Pareto noise
    """
    # Generate Pareto random variables
    pareto_rv = pareto.rvs(p, size=shape)
    pareto_tensor = torch.tensor(pareto_rv, dtype=torch.float32, device=device)
    
    # Symmetrize by randomly flipping signs
    signs = torch.randint(0, 2, shape, device=device) * 2 - 1
    symmetrized_noise = pareto_tensor * signs
    
    return symmetrized_noise


def generate_noise_for_experiment(
    shape: Union[Tuple, torch.Size], 
    noise_type: str, 
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate noise based on the experiment type.
    
    Args:
        shape: Shape of the noise tensor
        noise_type: Type of noise ('normal', 'pareto_2.5', 'pareto_1.5')
        device: Device to place the tensor on
        
    Returns:
        Tensor with the specified type of noise
    """
    if noise_type == 'normal':
        return generate_normal_noise(shape, device)
    elif noise_type == 'pareto_2.5':
        return generate_pareto_noise(shape, 2.5, device)
    elif noise_type == 'pareto_1.5':
        return generate_pareto_noise(shape, 1.5, device)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def add_noise_to_gradient(
    gradient: torch.Tensor, 
    noise_type: str, 
    noise_scale: float = 0.1,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Add noise to a gradient tensor.
    
    Args:
        gradient: Original gradient tensor
        noise_type: Type of noise to add
        noise_scale: Scale factor for the noise
        device: Device to place the noise on
        
    Returns:
        Gradient tensor with added noise
    """
    noise = generate_noise_for_experiment(gradient.shape, noise_type, device)
    noisy_gradient = gradient + noise_scale * noise
    return noisy_gradient


def get_noise_statistics(noise: torch.Tensor) -> dict:
    """
    Calculate statistics of the generated noise.
    
    Args:
        noise: Noise tensor
        
    Returns:
        Dictionary containing noise statistics
    """
    stats = {
        'mean': torch.mean(noise).item(),
        'std': torch.std(noise).item(),
        'min': torch.min(noise).item(),
        'max': torch.max(noise).item(),
        'norm': torch.norm(noise).item(),
        'shape': list(noise.shape)
    }
    
    # For matrix tensors, also compute Frobenius norm
    if noise.dim() == 2:
        stats['frobenius_norm'] = torch.norm(noise, p='fro').item()
    
    return stats
