"""
Distribution classes for topology-aware VAEs.

This module contains custom distributions with proper reparameterization:
- WrappedNormalDistribution: Wrapped Normal on S¹ (circle)
- KumaraswamyDistribution: Kumaraswamy on [0, 1] (bounded interval)

All distributions support exact reparameterization with informative gradients.

Author: Jilles van Hulst
"""

import numpy as np
import torch


class WrappedNormalDistribution:
    """
    Wrapped Normal distribution on S¹ (circle).
    
    This is a Gaussian wrapped around the circle:
    θ = (μ + σε) mod 2π, where ε ~ N(0,1)
    
    Key advantages over von Mises:
    - EXACT reparameterization (same as standard Gaussian!)
    - Gradients flow through both μ AND σ
    - As σ → ∞, approaches uniform on circle
    - As σ → 0, approaches point mass at μ
    
    The wrapping is achieved by centering around μ: θ_wrapped = μ + atan2(sin(θ-μ), cos(θ-μ))
    This ensures the branch cut is at μ+π (antipodal to the mean),
    placing the discontinuity where the density is lowest.
    """
    
    @staticmethod
    def sample(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Reparameterized sample: θ = μ + σε, then wrap to [μ-π, μ+π).
        
        This is the standard Gaussian reparameterization trick!
        Gradients flow through both μ and σ.
        The wrapping centers the branch cut at μ+π (antipodal to the mean),
        so that the discontinuity is far from the high-density region.
        """
        epsilon = torch.randn_like(mu)
        theta_unwrapped = mu + sigma * epsilon
        # Wrap to [μ-π, μ+π) by centering around μ before atan2
        centered = theta_unwrapped - mu
        theta = mu + torch.atan2(torch.sin(centered), torch.cos(centered))
        return theta
    
    @staticmethod
    def log_prob(theta: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, 
                 n_wraps: int = 3) -> torch.Tensor:
        """
        Log probability of wrapped normal (approximated by summing over wrappings).
        
        p(θ|μ,σ) ≈ Σₖ N(θ + 2πk | μ, σ²) for k in [-n_wraps, n_wraps]
        
        For moderate σ, n_wraps=3 is usually sufficient.
        """
        sigma = torch.clamp(sigma, min=0.01)
        
        # Sum over wrappings
        log_probs = []
        for k in range(-n_wraps, n_wraps + 1):
            theta_shifted = theta + 2 * np.pi * k
            # Gaussian log prob: -0.5 * ((x-μ)/σ)² - log(σ) - 0.5*log(2π)
            log_p = -0.5 * ((theta_shifted - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
            log_probs.append(log_p)
        
        # Log-sum-exp for numerical stability
        log_probs = torch.stack(log_probs, dim=-1)
        return torch.logsumexp(log_probs, dim=-1)
    
    @staticmethod
    def kl_divergence(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Approximate analytic KL for wrapped normal to uniform.
        
        For large σ, the wrapped normal approaches uniform, so KL → 0.
        For small σ, it behaves like a Gaussian with entropy ≈ log(σ√(2πe)).
        
        A useful approximation: KL ≈ max(0, -log(σ) - 0.5*log(2πe) + log(2π))
                                     = max(0, -log(σ) + 0.5*log(2π/e))
                                     ≈ max(0, -log(σ) + 0.42)
        
        This encourages σ → large (spreading) which pushes toward uniform.
        """
        sigma = torch.clamp(sigma, min=0.01)
        
        # Gaussian entropy = 0.5 * log(2πeσ²) = log(σ) + 0.5*log(2πe)
        gaussian_entropy = torch.log(sigma) + 0.5 * np.log(2 * np.pi * np.e)
        
        # Uniform entropy on [-π,π] = log(2π)
        uniform_entropy = np.log(2 * np.pi)
        
        # For wrapped normal, entropy is bounded by uniform_entropy
        # Approximate: H[wrapped] ≈ min(uniform_entropy, gaussian_entropy)
        # So KL ≈ uniform_entropy - min(uniform_entropy, gaussian_entropy)
        #       = max(0, uniform_entropy - gaussian_entropy)
        kl = torch.clamp(uniform_entropy - gaussian_entropy, min=0.0)
        
        return kl


class KumaraswamyDistribution:
    """
    Kumaraswamy distribution with support [0, 1].
    
    The CDF is F(x) = 1 - (1 - x^a)^b, which has a simple closed-form inverse:
    F^{-1}(u) = (1 - (1-u)^{1/b})^{1/a}
    
    This allows EXACT reparameterization via the inverse CDF trick:
    z = (1 - (1-u)^{1/b})^{1/a} where u ~ Uniform(0,1)
    
    Key properties:
    - When a = b = 1: Kumaraswamy = Uniform(0, 1)
    - Gradients flow through both a and b
    - Closed-form KL divergence to Uniform(0, 1)
    """
    
    @staticmethod
    def sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Reparameterized sample via inverse CDF.
        z = (1 - (1-u)^{1/b})^{1/a} where u ~ Uniform(0,1)
        """
        u = torch.rand_like(a)
        u = torch.clamp(u, 1e-4, 1 - 1e-4)
        a = torch.clamp(a, min=0.1, max=10.0)
        b = torch.clamp(b, min=0.1, max=10.0)
        z = (1 - (1 - u).pow(1.0 / b)).pow(1.0 / a)
        return torch.clamp(z, 1e-4, 1 - 1e-4)
    
    @staticmethod
    def log_prob(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Log probability: log p(x) = log(a) + log(b) + (a-1)log(x) + (b-1)log(1-x^a)
        """
        x = torch.clamp(x, 1e-4, 1 - 1e-4)
        a = torch.clamp(a, min=0.1, max=10.0)
        b = torch.clamp(b, min=0.1, max=10.0)
        x_a = x.pow(a)
        return (torch.log(a) + torch.log(b) + (a - 1) * torch.log(x) 
                + (b - 1) * torch.log(torch.clamp(1 - x_a, min=1e-8)))
    
    @staticmethod
    def kl_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from Kumaraswamy(a, b) to Uniform(0, 1).
        
        The entropy of Kumaraswamy(a,b) is:
        H = (1 - 1/b) + (1 - 1/a) * (γ + ψ(b) + 1/b) - log(ab)

        where γ + ψ(b) + 1/b = γ + ψ(b+1) is the b-th harmonic number.
        Since H[Uniform(0,1)] = 0 (in nats):
        KL(Kuma || Uniform) = -H[Kuma]

        This is nonnegative for every (a, b) in the clamped parameter range.

        Key property: KL = 0 when a = b = 1 (Kumaraswamy becomes Uniform)
        """
        a = torch.clamp(a, min=0.1, max=10.0)
        b = torch.clamp(b, min=0.1, max=10.0)
        
        # Euler-Mascheroni constant
        gamma = 0.5772156649
        
        # Digamma function: use torch.special.digamma for accurate computation
        psi_b = torch.special.digamma(b + 1e-8)
        
        # Harmonic-like term
        H_b = gamma + psi_b + 1.0 / (b + 1e-8)
        
        # Entropy (negated for KL)
        H = (1 - 1/b) + (1 - 1/a) * H_b - torch.log(a * b + 1e-8)
        
        # KL = -entropy (since uniform has entropy 0)
        return -H
    
    @staticmethod
    def mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean of Kumaraswamy(a, b).
        Mean = b * Gamma(1 + 1/a) * Gamma(b) / Gamma(1 + 1/a + b)
        
        For computational stability, we use the log-gamma function:
        log(mean) = log(b) + lgamma(1 + 1/a) + lgamma(b) - lgamma(1 + 1/a + b)
        """
        a = torch.clamp(a, min=0.1, max=10.0)
        b = torch.clamp(b, min=0.1, max=10.0)
        
        log_mean = (torch.log(b) + 
                   torch.lgamma(1.0 + 1.0/a) + 
                   torch.lgamma(b) - 
                   torch.lgamma(1.0 + 1.0/a + b))
        return torch.exp(log_mean)
