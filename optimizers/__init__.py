"""Optimization algorithms package for Gluon experiments.

This package contains implementations of:
- Gluon (stochastic Frank-Wolfe for matrices)
- Gluon++ (Gluon with momentum variance reduction)
"""

from .gluon import Gluon
from .gluon_plus import GluonPlus

__all__ = ['Gluon', 'GluonPlus']
