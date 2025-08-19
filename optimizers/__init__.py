"""
Optimization algorithms package.

This package contains implementations of:
- SCION (Stochastic Frank-Wolfe)
- SCION++ (SCION with momentum variance reduction)
- Muon (Matrix version of SCION)
- Muon++ (Muon with momentum variance reduction)
"""

from .scion import SCION
from .scion_plus import SCIONPlus
from .muon import Muon
from .muon_plus import MuonPlus

__all__ = ['SCION', 'SCIONPlus', 'Muon', 'MuonPlus']
