#!/usr/bin/env python3
"""Main experiment runner for Gluon vs Gluon++ comparison.

This script implements synthetic experiments for comparing the Gluon
optimizer with its variance-reduced counterpart Gluon++ on the quadratic
objective ``F(X) = 1/2 ||X||_F^2``.
"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from optimizers import Gluon, GluonPlus
from utils.noise_generators import generate_noise_for_experiment, add_noise_to_gradient
from utils.plotting import (
    plot_convergence, plot_algorithm_comparison, plot_noise_comparison,
    create_summary_table, plot_learning_curves
)
from experiments.config import ExperimentConfig, get_default_config, get_hyperparameters_from_paper


class SyntheticFunction:
    """Synthetic function ``F(X) = 1/2 ||X||_F^2`` for matrix experiments."""

    def __init__(self, n: int):
        """Create the quadratic function for an ``n x n`` matrix variable."""
        self.n = n
        self.x = nn.Parameter(torch.randn(n, n, requires_grad=True))

    def forward(self):
        """Compute the function value."""
        return 0.5 * torch.norm(self.x, p='fro') ** 2

    def get_gradient(self):
        """Get the gradient of the function."""
        self.x.grad = None
        loss = self.forward()
        loss.backward()
        return self.x.grad.clone()

    def reset_parameters(self):
        """Reset parameters to random initialization."""
        self.x.data = torch.randn(self.n, self.n)


def run_single_experiment(
    config: ExperimentConfig,
    dimension: int,
    noise_type: str,
    algorithm: str,
    device: str = "cpu"
) -> List[float]:
    """
    Run a single experiment configuration.
    
    Args:
        config: Experiment configuration
        dimension: Problem dimension
        noise_type: Type of noise to add
        algorithm: Algorithm to use
        device: Device to run on
        
    Returns:
        List of average gradient norms across iterations
    """
    # Create synthetic function
    func = SyntheticFunction(dimension)
    func.x = func.x.to(device)

    # Create optimizer
    if algorithm == "Gluon":
        optimizer = Gluon([func.x], **get_hyperparameters_from_paper()[algorithm])
    elif algorithm == "Gluon++":
        optimizer = GluonPlus([func.x], **get_hyperparameters_from_paper()[algorithm])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Initialize MVR optimizer if needed
    if "++" in algorithm:
        initial_grad = func.get_gradient()
        optimizer.set_initial_grad({func.x: initial_grad})
    
    gradient_norms = []
    
    for iteration in range(config.num_iterations):
        # Get clean gradient
        clean_gradient = func.get_gradient()
        
        # Add noise to gradient
        noisy_gradient = add_noise_to_gradient(
            clean_gradient, noise_type, config.noise_scale, device
        )
        
        # Set noisy gradient
        func.x.grad = noisy_gradient
        
        # Optimizer step
        if "++" in algorithm and iteration > 0:
            # For MVR, we need both old and new gradients
            old_gradient = clean_gradient
            optimizer.step()
            new_gradient = func.get_gradient()
            optimizer.step(grad_new_dict={func.x: new_gradient}, 
                         grad_old_dict={func.x: old_gradient})
        else:
            optimizer.step()
        
        # Compute gradient norm for convergence criterion
        current_gradient = func.get_gradient()
        gradient_norm = torch.norm(current_gradient, p='fro').item()
        
        gradient_norms.append(gradient_norm)
    
    # Return average gradient norm across all iterations
    return np.mean(gradient_norms)


def run_experiment_suite(config: ExperimentConfig) -> Dict:
    """Run the complete experiment suite."""
    print("Starting experiment suite")
    print(f"Dimensions: {config.dimensions}")
    print(f"Algorithms: {config.algorithms}")
    print(f"Noise types: {config.noise_types}")
    print(f"Number of runs: {config.num_runs}")
    print(f"Number of iterations: {config.num_iterations}")
    print("-" * 50)
    
    results = {}
    
    for dimension in config.dimensions:
        print(f"\nRunning experiments for dimension {dimension}")
        results[f"d={dimension}"] = {}
        
        for noise_type in config.noise_types:
            print(f"  Noise type: {noise_type}")
            results[f"d={dimension}"][noise_type] = {}
            
            for algorithm in config.algorithms:
                print(f"    Algorithm: {algorithm}")
                
                # Run multiple times to get statistics
                run_results = []
                for run in tqdm(range(config.num_runs), desc=f"{algorithm} runs"):
                    try:
                        result = run_single_experiment(
                            config, dimension, noise_type, algorithm, config.device
                        )
                        run_results.append(result)
                    except Exception as e:
                        print(f"Error in run {run}: {e}")
                        continue
                
                results[f"d={dimension}"][noise_type][algorithm] = run_results
                
                # Print summary statistics
                if run_results:
                    run_results = np.array(run_results)
                    print(f"      Mean: {np.mean(run_results):.6f}")
                    print(f"      Median: {np.median(run_results):.6f}")
                    print(f"      Std: {np.std(run_results):.6f}")
                    print(f"      Q(δ): {np.quantile(run_results, 1e-4):.6f}")
                    print(f"      Q(1-δ): {np.quantile(run_results, 1-1e-4):.6f}")
    
    return results


def save_results(results: Dict, config: ExperimentConfig):
    """Save experiment results to files."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save raw results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for dim_key, dim_results in results.items():
            json_results[dim_key] = {}
            for noise_key, noise_results in dim_results.items():
                json_results[dim_key][noise_key] = {}
                for algo_key, algo_results in noise_results.items():
                    json_results[dim_key][noise_key][algo_key] = [
                        float(x) for x in algo_results
                    ]
        
        json.dump(json_results, f, indent=2)
    
    # Save summary table
    # Save summary tables per dimension
    for dim_key, dim_results in results.items():
        summary_file = output_dir / f"summary_{dim_key}.csv"
        create_summary_table(dim_results, str(summary_file))
    
    print(f"Results saved to {output_dir}")


def create_plots(results: Dict, config: ExperimentConfig):
    """Create and save plots for the experiment results."""
    output_dir = Path(config.output_dir)
    
    for dimension in config.dimensions:
        dim_key = f"d={dimension}"
        if dim_key not in results:
            continue
            
        for noise_type in config.noise_types:
            if noise_type not in results[dim_key]:
                continue
            
            # Algorithm comparison plot
            algo_results = results[dim_key][noise_type]
            if len(algo_results) == 2:
                algo_names = list(algo_results.keys())
                plot_algorithm_comparison(
                    algo_results[algo_names[0]],
                    algo_results[algo_names[1]],
                    noise_type,
                    dimension,
                    str(output_dir / f"comparison_d{dimension}_{noise_type}.png")
                )
    
    # Noise comparison plot
    for dimension in config.dimensions:
        dim_key = f"d={dimension}"
        if dim_key in results:
            plot_noise_comparison(
                results[dim_key],
                dimension,
                str(output_dir / f"noise_comparison_d{dimension}.png")
            )


def main():
    """Main function to run the experiments."""
    parser = argparse.ArgumentParser(description="Gluon vs Gluon++ Experiments")
    parser.add_argument("--dimension", type=int, nargs="+",
                       help="Specific matrix dimensions to test")
    parser.add_argument("--noise_type", choices=["normal", "pareto_2.5", "pareto_1.5"],
                       help="Specific noise type to test")
    parser.add_argument("--algorithm", choices=["Gluon", "Gluon++"],
                       help="Specific algorithm to test")
    parser.add_argument("--num_runs", type=int, default=100000,
                       help="Number of runs (default: 100000)")
    parser.add_argument("--num_iterations", type=int, default=100,
                       help="Number of iterations per run (default: 100)")
    parser.add_argument("--device", default="cpu", help="Device to use (default: cpu)")
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_default_config()
    
    # Override with command line arguments
    if args.dimension:
        config.dimensions = args.dimension
    if args.noise_type:
        config.noise_types = [args.noise_type]
    if args.algorithm:
        config.algorithms = [args.algorithm]
    if args.num_runs:
        config.num_runs = args.num_runs
    if args.num_iterations:
        config.num_iterations = args.num_iterations
    if args.device:
        config.device = args.device
    if args.no_plots:
        config.plot_results = False

    print("Gluon vs Gluon++ Experiments")
    print("=" * 50)
    print(f"Configuration: {config}")
    
    # Run experiments
    start_time = time.time()
    results = run_experiment_suite(config)
    end_time = time.time()
    
    print(f"\nExperiments completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    if config.save_results:
        save_results(results, config)
    
    # Create plots
    if config.plot_results:
        create_plots(results, config)
    
    print("\nExperiment suite completed successfully!")


if __name__ == "__main__":
    main()
