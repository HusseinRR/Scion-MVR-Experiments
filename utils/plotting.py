"""
Plotting utilities for SCION vs SCION++ experiments.

This module provides functions to visualize convergence behavior,
algorithm comparisons, and experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import pandas as pd


def plot_convergence(
    results: Dict[str, List[float]], 
    title: str = "Convergence Comparison",
    save_path: Optional[str] = None,
    show_quantiles: bool = True,
    delta: float = 1e-4
):
    """
    Plot convergence behavior of different algorithms.
    
    Args:
        results: Dictionary with algorithm names as keys and lists of gradient norms as values
        title: Plot title
        save_path: Optional path to save the plot
        show_quantiles: Whether to show δ and 1-δ quantiles
        delta: Quantile parameter (δ=10⁻⁴ as in the paper)
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for i, (algo_name, gradient_norms) in enumerate(results.items()):
        gradient_norms = np.array(gradient_norms)
        
        # Calculate statistics
        median = np.median(gradient_norms)
        q_low = np.quantile(gradient_norms, delta)
        q_high = np.quantile(gradient_norms, 1 - delta)
        
        # Plot median line
        plt.plot([0, len(gradient_norms)-1], [median, median], 
                color=colors[i], linestyle='-', linewidth=2, 
                label=f'{algo_name} (median)')
        
        # Plot quantiles
        if show_quantiles:
            plt.fill_between([0, len(gradient_norms)-1], [q_low, q_low], [q_high, q_high], 
                           color=colors[i], alpha=0.3, 
                           label=f'{algo_name} ({delta:.0e}, {1-delta:.0e}) quantiles')
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Gradient Norm')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_algorithm_comparison(
    scion_results: List[float],
    scion_plus_results: List[float],
    noise_type: str,
    dimension: int,
    save_path: Optional[str] = None
):
    """
    Plot comparison between SCION and SCION++ for a specific experiment.
    
    Args:
        scion_results: List of gradient norms from SCION
        scion_plus_results: List of gradient norms from SCION++
        noise_type: Type of noise used
        dimension: Problem dimension
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Histogram comparison
    ax1.hist(scion_results, bins=50, alpha=0.7, label='SCION', density=True)
    ax1.hist(scion_plus_results, bins=50, alpha=0.7, label='SCION++', density=True)
    ax1.set_xlabel('Average Gradient Norm')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution Comparison ({noise_type}, d={dimension})')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    data = [scion_results, scion_plus_results]
    labels = ['SCION', 'SCION++']
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Average Gradient Norm')
    ax2.set_title(f'Box Plot Comparison ({noise_type}, d={dimension})')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_noise_comparison(
    noise_results: Dict[str, Dict[str, List[float]]],
    dimension: int,
    save_path: Optional[str] = None
):
    """
    Plot comparison across different noise types.
    
    Args:
        noise_results: Dictionary with noise types as keys and algorithm results as values
        dimension: Problem dimension
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    noise_types = list(noise_results.keys())
    algorithms = list(noise_results[noise_types[0]].keys())
    
    for i, noise_type in enumerate(noise_types):
        ax = axes[i]
        
        data = [noise_results[noise_type][algo] for algo in algorithms]
        bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Average Gradient Norm')
        ax.set_title(f'{noise_type} Noise (d={dimension})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_summary_table(
    results: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a summary table of experiment results.
    
    Args:
        results: Nested dictionary with results
        save_path: Optional path to save the table as CSV
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for noise_type, algo_results in results.items():
        for algo_name, gradient_norms in algo_results.items():
            gradient_norms = np.array(gradient_norms)
            
            summary_data.append({
                'Noise Type': noise_type,
                'Algorithm': algo_name,
                'Mean': np.mean(gradient_norms),
                'Median': np.median(gradient_norms),
                'Std': np.std(gradient_norms),
                'Min': np.min(gradient_norms),
                'Max': np.max(gradient_norms),
                'Q(δ)': np.quantile(gradient_norms, 1e-4),
                'Q(1-δ)': np.quantile(gradient_norms, 1 - 1e-4),
                'Count': len(gradient_norms)
            })
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df


def plot_learning_curves(
    learning_curves: Dict[str, List[float]],
    title: str = "Learning Curves",
    save_path: Optional[str] = None
):
    """
    Plot learning curves for different algorithms.
    
    Args:
        learning_curves: Dictionary with algorithm names as keys and learning curves as values
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(learning_curves)))
    
    for i, (algo_name, curve) in enumerate(learning_curves.items()):
        plt.plot(curve, color=colors[i], linewidth=2, label=algo_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
