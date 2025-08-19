"""
Norm classes for gradient projection in SCION optimizers.

This module provides various norm implementations for the LMO (Linear Minimization Oracle)
used in Frank-Wolfe algorithms.
"""

import torch
import torch.nn as nn


class Norm:
    """Base class for all norm implementations."""

    def __init__(self):
        self.eps = 1e-8

    def lmo(self, g):
        """Linear Minimization Oracle - must be implemented by subclasses."""
        raise NotImplementedError

    def init(self, w):
        """Initialize weights - must be implemented by subclasses."""
        raise NotImplementedError
    
    def calculate_norm(self, p):
        """Calculate the norm of a parameter tensor."""
        raise NotImplementedError


class BiasRMS(Norm):
    """RMS normalization for bias parameters."""
    
    def lmo(self, g):
        rms_values = torch.sqrt(torch.mean(g ** 2, dim=0, keepdim=True))
        g = g / (rms_values + self.eps)
        return g
    
    def calculate_norm(self, p):
        rms_values = torch.sqrt(torch.mean(p ** 2, dim=0, keepdim=True))
        return rms_values       

    def init(self, g):
        return torch.nn.init.zeros_(g)


class SpectralConv(Norm):
    """Spectral normalization for convolutional layers."""
    
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        # Simplified version without external dependencies
        # For actual implementation, you'd need zeropower_via_newtonschulz5
        g_reshaped = g.reshape(len(g), -1)
        
        # Simple spectral normalization approximation
        u, s, v = torch.svd(g_reshaped)
        # Take the first singular vector
        g_normalized = u[:, 0:1] @ v[:, 0:1].T
        
        g = g_normalized.view(g.shape)
        out_channels, in_channels, k, _ = g.shape
        g *= (out_channels / in_channels)**0.5 / (k ** 2)
        return g

    def calculate_norm(self, p):
        norm = torch.linalg.norm(p.reshape(len(p), -1), ord=2)
        out_channels, in_channels, k, _ = p.shape
        norm /= (out_channels / in_channels)**0.5 / (k ** 2)
        return norm
    
    def init(self, w):
        w_fp = w.data.double()
        k = w.data.size(2)
        for kx in range(k):
            for ky in range(k):
                torch.nn.init.orthogonal_(w_fp[:,:,kx,ky])
        
        out_channels, in_channels, k, _ = w_fp.shape
        w_fp.mul_((out_channels / in_channels)**0.5 / (k ** 2))
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Spectral(Norm):
    """Spectral normalization for linear layers."""
    
    def __init__(self, max=False, normalized=True, steps=5, log_stats=False):
        self.max = max
        self.steps = steps
        self.normalized = normalized
        self.log_stats = log_stats

    def lmo(self, g):
        if self.log_stats:
            print(f"Head gradient stats: min: {torch.min(g)} mean: {torch.mean(g)} max: {torch.max(g)} norm: {torch.linalg.norm(g)}")
            n, m = g.shape
            print(f"ggT norm {torch.linalg.norm(g @ g.T)}")
            print(f"|ggT-I| before NS {torch.linalg.norm(g @ g.T - torch.ones((n,n), device=g.device))}")
        
        # Simplified spectral normalization
        u, s, v = torch.svd(g)
        # Take the first singular vector
        g = u[:, 0:1] @ v[:, 0:1].T
        
        if self.log_stats:
            print(f"|ggT-I| after NS {torch.linalg.norm(g @ g.T - torch.ones((n,n), device=g.device))}")
        
        d_out, d_in = g.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1, scale)
        g *= scale
        return g

    def calculate_norm(self, p):
        norm = torch.linalg.norm(p, ord=2)
        d_out, d_in = p.shape
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1, scale)
        norm *= 1 / scale
        return norm

    def init(self, w):
        w_fp = w.data.double()
        torch.nn.init.orthogonal_(w_fp)
        d_out, d_in = w_fp.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1, scale)
        w_fp.mul_(scale)
    
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Sign(Norm):
    """Sign normalization for binary-like parameters."""
    
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        d_out, d_in = g.shape
        if self.normalized:
            return (1/d_in) * torch.sign(g)    
        else:
            return torch.sign(g)

    def calculate_norm(self, p):
        d_out, d_in = p.shape
        if self.normalized:
            return d_in * torch.max(torch.abs(p))    
        else:
            return torch.max(torch.abs(p))

    def init(self, w):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            # Generate -1/fan_in or 1/fan_in uniformly at random
            d_out, d_in = w.shape
            w.data = (torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1)
            if self.normalized:
                w.data *= (1/d_in)
        return w


class Auto(Norm):
    """Automatic norm selection based on tensor dimensions."""
    
    def lmo(self, g):
        if g.ndim in [3, 4]:
            return SpectralConv().lmo(g)
        elif g.ndim == 2:
            return Spectral().lmo(g)
        elif g.ndim in [0, 1]:
            return BiasRMS().lmo(g)

    def calculate_norm(self, p):
        if p.ndim in [3, 4]:
            return SpectralConv().calculate_norm(p)
        elif p.ndim == 2:
            return Spectral().calculate_norm(p)
        elif p.ndim in [0, 1]:
            return BiasRMS().calculate_norm(p)

    def init(self, w):
        if w.ndim in [3, 4]:
            return SpectralConv().init(w)
        elif w.ndim == 2:
            return Spectral().init(w)
        elif w.ndim in [0, 1]:
            return BiasRMS().init(w)


# Dictionary mapping norm names to classes
norm_dict = {
    'Sign': Sign,
    'Spectral': Spectral,
    'SpectralConv': SpectralConv,
    'RMS': BiasRMS, 
    'Auto': Auto,
}
