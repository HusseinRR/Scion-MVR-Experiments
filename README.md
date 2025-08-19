# Gluon vs Gluon with Momentum Variance Reduction Experiments

This project implements synthetic heavy-tailed dataset experiments to compare the performance of Gluon and its variance-reduced variant Gluon++ on the function ``F(X) = 1/2 ||X||_F^2`` with different types of noise.

## Overview

The experiments are based on the setup from Hübler et al. [2025] and evaluate:
- **Gluon vs Gluon++**: For n=1 and n=30 matrix problems

## Noise Types

1. **Standard Normal**: Light-tailed noise
2. **Component-wise symmetrized Pareto (p=2.5)**: Heavy-tailed noise with finite variance
3. **Component-wise symmetrized Pareto (p=1.5)**: Heavy-tailed noise with infinite variance

## Project Structure

```
├── requirements.txt          # Python dependencies
├── README.md               # This file
├── main.py                 # Main experiment runner
├── optimizers/             # Optimization algorithms
│   ├── __init__.py
│   ├── gluon_base.py      # Base Gluon optimizer
│   ├── gluon.py           # Gluon optimizer
│   └── gluon_plus.py      # Gluon++ optimizer
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── noise_generators.py # Noise generation functions
│   └── plotting.py         # Visualization utilities
└── experiments/            # Experiment configurations
    ├── __init__.py
    └── config.py           # Hyperparameters and settings
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Basic Usage
```bash
python main.py
```

### Custom Configuration
```bash
python main.py --dimension 1000 --noise_type pareto_2.5 --iterations 100
```

## Results

The experiments run each algorithm 10⁵ times for T=100 iterations and use the average gradient norm across all iterations as the convergence criterion. Results are displayed as:
- Median convergence behavior
- δ and 1-δ quantiles (with δ=10⁻⁴)

## Hyperparameters

Key hyperparameters are configurable in `experiments/config.py`:
- Learning rates
- Momentum parameters
- Variance reduction parameters
- Noise distribution parameters

## Citation

This implementation follows the experimental setup from:
Hübler et al. [2025] - "Additional Experiments: Empirical Investigation of Stochastic FW with Clipping and Variance Reduction"
