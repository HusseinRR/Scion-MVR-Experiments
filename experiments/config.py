"""Experiment configuration for Gluon vs Gluon++ experiments.

This module stores configuration settings and hyperparameters for the
synthetic heavy-tailed experiments on ``F(X) = 1/2 ||X||_F^2``.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExperimentConfig:
    """Configuration class for Gluon experiments."""

    # Problem settings
    dimensions: List[int] = None  # e.g. [1, 30]

    # Algorithm settings
    algorithms: List[str] = None  # ["Gluon", "Gluon++"]
    
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
            self.dimensions = [1, 30]

        if self.algorithms is None:
            self.algorithms = ["Gluon", "Gluon++"]

        if self.noise_types is None:
            self.noise_types = ["normal", "pareto_2.5", "pareto_1.5"]


def get_default_config() -> ExperimentConfig:
    """Get the default configuration for Gluon experiments."""
    return ExperimentConfig(
        dimensions=[1, 30],
        algorithms=["Gluon", "Gluon++"],
        learning_rate=0.1,
        momentum=0.9,
        clipping_threshold=1.0,
        mvr_parameter=0.5
    )


def get_hyperparameters_from_paper() -> Dict:
    """
    Get hyperparameters as reported in Tables 4 and 5 of the paper.
    
    Returns:
        Dictionary with hyperparameters for different algorithms
    """
    return {
        "Gluon": {
            "lr": 0.1,
            "momentum": 1.0,
            "clipping_threshold": 1.0,
            "norm": "Auto",
            "norm_kwargs": {},
            "scale": 1.0,
            "unconstrained": False
        },
        "Gluon++": {
            "lr": 0.1,
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
    
    if "++" in algorithm:
        base_params["p"] = config.mvr_parameter
    
    return base_params
