"""
SCION (Stochastic Frank-Wolfe) optimizer implementation.

This implements the basic SCION algorithm for optimizing F(x) = 1/2 ||x||²
with stochastic gradients corrupted by noise.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable


class SCION(torch.optim.Optimizer):
    """SCION optimizer implementation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv', 
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)
        scale (float, optional): Scale factor for updates (default: 1.0)
        unconstrained (bool, optional): Whether to use unconstrained updates (default: False)
    """
    
    def __init__(self, 
                 params, 
                 lr=1e-3, 
                 momentum=1.0, 
                 unconstrained=False,
                 device=None,
                 norm: str='Auto', 
                 norm_kwargs: dict=None, 
                 scale=1.0,
                 clipping_threshold: float = 1.0):
                
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = dict()
            
        self.device = device
        self.clipping_threshold = clipping_threshold
        defaults = dict(lr=lr, momentum=momentum, unconstrained=unconstrained, 
                       norm=norm, norm_kwargs=norm_kwargs, scale=scale)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                g = p.grad.data
                state = self.state[p]

                # Apply momentum if specified
                if momentum != 1.0:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1-momentum).add_(g, alpha=momentum)
                    g = buf

                # Gradient clipping for heavy-tailed noise
                g_norm = torch.norm(g)
                if g_norm > self.clipping_threshold:
                    g = g * (self.clipping_threshold / g_norm)

                # Frank-Wolfe step: project to unit ball
                if not unconstrained:
                    # For unit ball constraint, the LMO is -g/||g||
                    direction = -g / (torch.norm(g) + 1e-8)
                    if torch.norm(direction) > 1.0:
                        direction = direction / torch.norm(direction)
                    update = scale * direction
                else:
                    # Unconstrained: use gradient directly
                    update = -scale * g

                # Update parameters
                if not unconstrained:
                    p.data.mul_(1-lr)
                p.data.add_(update, alpha=lr)

        return loss

    def get_state_dict(self):
        """Get optimizer state for saving/loading."""
        return {
            'state': self.state,
            'param_groups': self.param_groups,
            'clipping_threshold': self.clipping_threshold,
            'device': self.device
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        super().load_state_dict(state_dict)
        self.clipping_threshold = state_dict.get('clipping_threshold', 1.0)
        self.device = state_dict.get('device', 'cpu')
