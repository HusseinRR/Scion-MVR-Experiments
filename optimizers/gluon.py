"""Gluon optimizer implementation for matrix optimization problems.

This is the matrix version of the Gluon algorithm for minimizing
``F(X) = 1/2 ||X||_F^2`` with stochastic gradients corrupted by noise.
"""

import torch
from .gluon_base import GluonBase
from .norms import norm_dict


class Gluon(GluonBase):
    """Gluon optimizer for matrix parameters.

    Extends :class:`GluonBase` to handle matrices using the Frobenius norm.
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
        """Initialize the Gluon optimizer.

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
        """
        super().__init__(params, lr, momentum, unconstrained, device, norm,
                         norm_kwargs, scale, clipping_threshold)

    def step(self, closure=None):
        """Perform a single optimization step for matrix parameters."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            
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
                g_norm = torch.norm(g, p='fro')  # Use Frobenius norm for matrices
                if g_norm > self.clipping_threshold:
                    g = g * (self.clipping_threshold / g_norm)

                # Frank-Wolfe step: project to unit ball in Frobenius norm
                if not unconstrained:
                    # For unit ball constraint, the LMO is -g/||g||_F
                    direction = -g / (torch.norm(g, p='fro') + 1e-8)
                    if torch.norm(direction, p='fro') > 1.0:
                        direction = direction / torch.norm(direction, p='fro')
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
        return super().get_state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        super().load_state_dict(state_dict)
