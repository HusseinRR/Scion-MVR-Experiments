"""
Experiment configuration for SCION vs SCION++ experiments.

This module contains the configuration settings and hyperparameters
for the synthetic heavy-tailed dataset experiments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch


@dataclass
class ExperimentConfig:
    """Configuration class for SCION vs SCION++ experiments."""
    
    # Problem settings
    problem_type: str = "vector"  # "vector" or "matrix"
    dimensions: List[int] = None  # [1, 1000] for vector, [1, 30] for matrix
    
    # Algorithm settings
    algorithms: List[str] = None  # ["SCION", "SCION++"] or ["Muon", "Muon++"]
    
    # Noise settings
    noise_types: List[str] = None  # ["normal", "pareto_2.5", "pareto_1.5"]
    noise_scale: float = 0.1
    
    # Optimization settings
    learning_rate: float = 0.1
    momentum: float = 0.9
    clipping_threshold: float = 1.0
    mvr_parameter: float = 0.5
    
    # Experiment settings
    num_runs: int = 100000  # 10^5 runs as in the paper
    num_iterations: int = 100  # T = 100 iterations
    convergence_criterion: str = "average_gradient_norm"
    
    # Device settings
    device: str = "cpu"
    
    # Output settings
    save_results: bool = True
    output_dir: str = "results"
    plot_results: bool = True
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.dimensions is None:
            if self.problem_type == "vector":
                self.dimensions = [1, 1000]
            else:
                self.dimensions = [1, 30]
        
        if self.algorithms is None:
            if self.problem_type == "vector":
                self.algorithms = ["SCION", "SCION++"]
            else:
                self.algorithms = ["Muon", "Muon++"]
        
        if self.noise_types is None:
            self.noise_types = ["normal", "pareto_2.5", "pareto_1.5"]


def get_default_config(problem_type: str = "vector") -> ExperimentConfig:
    """
    Get default configuration for a specific problem type.
    
    Args:
        problem_type: Either "vector" or "matrix"
        
    Returns:
        ExperimentConfig with default settings
    """
    if problem_type == "vector":
        return ExperimentConfig(
            problem_type="vector",
            dimensions=[1, 1000],
            algorithms=["SCION", "SCION++"],
            learning_rate=0.1,
            momentum=0.9,
            clipping_threshold=1.0,
            mvr_parameter=0.5
        )
    elif problem_type == "matrix":
        return ExperimentConfig(
            problem_type="matrix",
            dimensions=[1, 30],
            algorithms=["Muon", "Muon++"],
            learning_rate=0.1,
            momentum=0.9,
            clipping_threshold=1.0,
            mvr_parameter=0.5
        )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def get_hyperparameters_from_paper() -> Dict:
    """
    Get hyperparameters as reported in Tables 4 and 5 of the paper.
    
    Returns:
        Dictionary with hyperparameters for different algorithms
    """
    return {
        "SCION": {
            "learning_rate": 0.1,
            "momentum": 1.0,
            "clipping_threshold": 1.0,
            "norm": "Auto",
            "norm_kwargs": {},
            "scale": 1.0,
            "unconstrained": False
        },
        "SCION++": {
            "learning_rate": 0.1,
            "momentum": 0.9,
            "clipping_threshold": 1.0,
            "norm": "Auto",
            "norm_kwargs": {},
            "scale": 1.0,
            "unconstrained": False,
            "p": 0.5
        },
        "Muon": {
            "learning_rate": 0.1,
            "momentum": 1.0,
            "clipping_threshold": 1.0,
            "norm": "Auto",
            "norm_kwargs": {},
            "scale": 1.0,
            "unconstrained": False
        },
        "Muon++": {
            "learning_rate": 0.1,
            "momentum": 0.9,
            "clipping_threshold": 1.0,
            "norm": "Auto",
            "norm_kwargs": {},
            "scale": 1.0,
            "unconstrained": False,
            "p": 0.5
        }
    }


def create_optimizer_params(config: ExperimentConfig, algorithm: str) -> Dict:
    """
    Create optimizer parameters for a specific algorithm.
    
    Args:
        config: Experiment configuration
        algorithm: Algorithm name
        
    Returns:
        Dictionary with optimizer parameters
    """
    base_params = {
        "lr": config.learning_rate,
        "momentum": config.momentum,
        "unconstrained": False,
        "device": config.device,
        "norm": "Auto",
        "norm_kwargs": {},
        "scale": 1.0,
        "clipping_threshold": config.clipping_threshold
    }
    
    if "++" in algorithm:  # SCION++ or Muon++
        base_params["p"] = config.mvr_parameter
    
    return base_params
