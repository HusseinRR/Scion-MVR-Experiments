"""
Muon++ optimizer implementation.

This implements Muon with momentum variance reduction (MVR) for
better performance on heavy-tailed noise scenarios in matrix optimization.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from .muon import Muon
from .norms import norm_dict


class MuonPlus(Muon):
    """
    Muon++ optimizer with Momentum Variance Reduction (MVR).
    
    Extends Muon with momentum and variance reduction techniques
    to improve convergence on heavy-tailed noise for matrix problems.
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=1.0,
        unconstrained=False,
        device=None,
        norm: str='Auto',
        norm_kwargs: dict=None,
        scale=1.0,
        clipping_threshold: float = 1.0,
        p: float = 0.5
    ):
        """
        Initialize Muon++ optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            momentum: Momentum parameter
            unconstrained: Whether to use unconstrained updates
            device: Device to run computations on
            norm: Norm type for gradient projection
            norm_kwargs: Additional norm arguments
            scale: Scale factor for updates
            clipping_threshold: Threshold for gradient clipping
            p: MVR parameter (0 < p < 1)
        """
        super().__init__(params, lr, momentum, unconstrained, device, norm, 
                        norm_kwargs, scale, clipping_threshold)
        self.p = p
        self.is_initialized = False
        
        # Initialize gradient estimators
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['grad_estimation'] = torch.zeros_like(p.data)

    def set_initial_grad(self, initial_grad_dict: dict):
        """
        Initializes the gradient estimators with the first stochastic gradient (g_0).
        This MUST be called by the trainer once before the first step.
        """
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        if p in initial_grad_dict:
                            state['grad_estimation'] = initial_grad_dict[p].clone()
                        else:
                            state['grad_estimation'] = torch.zeros_like(p.data)
        self.is_initialized = True
            
    def step(self, closure=None, grad_new_dict: dict = None, grad_old_dict: dict = None):
        """
        Performs a single MVR optimization step for matrix parameters.
        
        Args:
            closure: Optional closure for computing loss
            grad_new_dict: Dict of {param: gradient} at the new position (X_{k+1})
            grad_old_dict: Dict of {param: gradient} at the old position (X_k)
        """
        if not self.is_initialized and grad_new_dict is not None:
            raise RuntimeError("Optimizer not initialized. Call `set_initial_grad` before the first step.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # If MVR mode (gradients provided)
        if grad_new_dict is not None and grad_old_dict is not None:
            self._mvr_step(grad_new_dict, grad_old_dict)
        else:
            # Standard Muon step
            super().step(closure)

        return loss

    def _mvr_step(self, grad_new_dict: dict, grad_old_dict: dict):
        """Internal MVR step implementation for matrix parameters."""
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                scale = group['scale']
                unconstrained = group['unconstrained']
                norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
                
                for p in group['params']:
                    if not p.requires_grad or p not in grad_new_dict:
                        continue
                        
                    state = self.state[p]
                    # This is g_k, the estimator from the previous step
                    g_k = state['grad_estimation'] 
                    
                    # 1. Update Position using g_k to get X_{k+1}
                    g_k_norm = torch.norm(g_k, p='fro')  # Use Frobenius norm for matrices
                    if g_k_norm > self.clipping_threshold:
                        g_k = g_k * (self.clipping_threshold / g_k_norm)
                    
                    if not unconstrained:
                        # For unit ball constraint, the LMO is -g_k/||g_k||_F
                        direction = -g_k / (torch.norm(g_k, p='fro') + 1e-8)
                        if torch.norm(direction, p='fro') > 1.0:
                            direction = direction / torch.norm(direction, p='fro')
                        update = scale * direction
                    else:
                        # Unconstrained: use gradient directly
                        update = -scale * g_k
                    
                    # Update parameters
                    if not unconstrained:
                        p.data.mul_(1 - lr)
                    p.data.add_(update, alpha=lr)
                    
                    # 2. Update Gradient Estimator to get g_{k+1} for the *next* step
                    g_new = grad_new_dict[p]
                    g_old = grad_old_dict[p]
                    
                    # MVR formula: g_{k+1} = grad_new + (1-p) * (g_k - grad_old)
                    g_k_plus_1 = g_new + (1 - self.p) * (g_k - g_old)
                    
                    # Update the state for the next iteration
                    state['grad_estimation'].copy_(g_k_plus_1)
    
    def get_state_dict(self):
        """Get optimizer state for saving/loading."""
        state_dict = super().get_state_dict()
        state_dict.update({
            'p': self.p,
            'is_initialized': self.is_initialized
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        super().load_state_dict(state_dict)
        self.p = state_dict.get('p', 0.5)
        self.is_initialized = state_dict.get('is_initialized', False)
