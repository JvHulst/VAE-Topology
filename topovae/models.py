"""
Every VAE in the paper, on one prescribed latent manifold each.

The synthetic models take a 50-dimensional observation vector; the image models
take a flattened 28×28 MNIST digit and use a CNN backbone.  Apart from that they
share one contract, which is what makes them readable side by side:

    encode(x)          → the posterior parameters for this manifold
    reparameterize(..) → a sample, differentiable through the parameters
    decode(..)         → the reconstruction
    forward(x)         → (x_recon, *posterior_params, *samples)
    loss(x, x_recon, *posterior_params, beta) → (total, recon, kl)

| Class            | Latent manifold          | Distributions              |
|------------------|--------------------------|----------------------------|
| GaussianVAE      | ℝ^d                      | Gaussian                   |
| CylinderVAE      | S¹ × [0, 1]              | WrappedNormal, Kumaraswamy |
| TorusVAE         | T² = S¹ × S¹             | WrappedNormal ×2           |
| AnnulusVAE       | S¹ × [r_min, r_max]      | WrappedNormal, Kumaraswamy |
| MobiusVAE        | (S¹ × [0, 1]) / ℤ₂       | WrappedNormal, Kumaraswamy |
| MNISTGaussianVAE | ℝ^d                      | Gaussian                   |
| MixedCircleVAE   | ℝ^k × S¹                 | Gaussian, WrappedNormal    |
| MixedTorusVAE    | ℝ^k × T²                 | Gaussian, WrappedNormal ×2 |

The KL divergence decouples across the factors of a product latent space
(Proposition 1 of the paper), so each model's ``loss`` sums a per-factor KL.

Author: Jilles van Hulst
"""

import numpy as np
import torch
import torch.nn as nn

from .distributions import WrappedNormalDistribution, KumaraswamyDistribution
from .layers import (
    build_cnn_decoder,
    build_cnn_encoder,
    build_mlp,
    cossin_to_angle,
    mobius_invariant_features,
    softplus_param,
)


# =============================================================================
# SYNTHETIC EXPERIMENTS — 50-DIMENSIONAL OBSERVATION VECTORS
# =============================================================================


class GaussianVAE(nn.Module):
    """
    Standard VAE with Gaussian encoder and N(0,1) prior.
    
    This serves as the baseline comparison for topology-aware VAEs.
    """
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = build_mlp([input_dim, hidden_dim, hidden_dim, hidden_dim // 2])
        self.decoder = build_mlp([latent_dim, hidden_dim // 2, hidden_dim, hidden_dim, input_dim])
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def loss(self, x, x_recon, mu, logvar, beta=1.0):
        recon = torch.mean(torch.sum((x - x_recon)**2, dim=1))
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))
        return recon + beta * kl, recon, kl
    
    def repelling_loss(self, z, repel_center, xi=0.3):
        """
        Compute repelling loss that pushes latent codes away from a center point.
        
        L_repel = -log(||z - center||^2 / xi^2 + 1)
        
        This creates a soft hole around the repel_center with characteristic radius xi.
        The parameter xi directly controls the scale of the hole: points within distance
        xi from the center experience strong repulsion, while points beyond ~2*xi
        experience negligible force.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            repel_center: Center point to repel from [latent_dim] or [1, latent_dim]
            xi: Characteristic radius of the hole (default: 0.3)
        """
        if repel_center.dim() == 1:
            repel_center = repel_center.unsqueeze(0)
        
        dist_sq = torch.sum((z - repel_center)**2, dim=1)
        # Normalize by xi^2 so that xi directly controls the hole scale
        repel_loss = -torch.mean(torch.log(dist_sq / (xi**2) + 1.0))
        return repel_loss


class CylinderVAE(nn.Module):
    """
    VAE for cylindrical latent space S¹ × [0, 1].
    
    Uses Wrapped Normal for the angular coordinate (proper reparameterization)
    and Kumaraswamy for the height coordinate.
    
    The encoder outputs (cos, sin) which is normalized to unit circle,
    eliminating the boundary problem at ±π.
    """
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = build_mlp([input_dim, hidden_dim, hidden_dim, hidden_dim // 2])
        self.decoder = build_mlp([3, hidden_dim // 2, hidden_dim, hidden_dim, input_dim])
        
        # Encoder heads
        self.fc_cossin = nn.Linear(hidden_dim // 2, 2)
        self.fc_log_sigma = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_a = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_b = nn.Linear(hidden_dim // 2, 1)
    
    def encode(self, x):
        h = self.encoder(x)
        mu_angle = cossin_to_angle(self.fc_cossin(h))
        sigma = softplus_param(self.fc_log_sigma(h))
        a = softplus_param(self.fc_log_a(h))
        b = softplus_param(self.fc_log_b(h))
        return mu_angle, sigma, a, b
    
    def reparameterize(self, mu_angle, sigma, a, b):
        theta = WrappedNormalDistribution.sample(mu_angle, sigma)
        h = KumaraswamyDistribution.sample(a, b)
        return theta, h
    
    def decode(self, theta, h):
        z = torch.cat([torch.cos(theta), torch.sin(theta), h], dim=1)
        return self.decoder(z)
    
    def forward(self, x):
        mu_angle, sigma, a, b = self.encode(x)
        theta, h = self.reparameterize(mu_angle, sigma, a, b)
        x_recon = self.decode(theta, h)
        return x_recon, mu_angle, sigma, a, b, theta, h
    
    def loss(self, x, x_recon, mu_angle, sigma, a, b, beta=1.0):
        recon = torch.mean(torch.sum((x - x_recon)**2, dim=1))
        kl_angle = torch.mean(WrappedNormalDistribution.kl_divergence(mu_angle, sigma))
        kl_height = torch.mean(KumaraswamyDistribution.kl_divergence(a, b))
        kl = kl_angle + kl_height
        return recon + beta * kl, recon, kl


class TorusVAE(nn.Module):
    """
    VAE for toroidal latent space T² = S¹ × S¹.
    
    Uses Wrapped Normal for both angular coordinates (proper reparameterization).
    """
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = build_mlp([input_dim, hidden_dim, hidden_dim, hidden_dim // 2])
        # Decoder receives [cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂)] = 4D
        self.decoder = build_mlp([4, hidden_dim // 2, hidden_dim, hidden_dim, input_dim])
        
        # Encoder heads
        self.fc_cossin1 = nn.Linear(hidden_dim // 2, 2)
        self.fc_cossin2 = nn.Linear(hidden_dim // 2, 2)
        self.fc_log_sigma1 = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_sigma2 = nn.Linear(hidden_dim // 2, 1)
    
    def encode(self, x):
        h = self.encoder(x)
        mu1 = cossin_to_angle(self.fc_cossin1(h))
        mu2 = cossin_to_angle(self.fc_cossin2(h))
        sigma1 = softplus_param(self.fc_log_sigma1(h))
        sigma2 = softplus_param(self.fc_log_sigma2(h))
        return mu1, sigma1, mu2, sigma2
    
    def reparameterize(self, mu1, sigma1, mu2, sigma2):
        theta1 = WrappedNormalDistribution.sample(mu1, sigma1)
        theta2 = WrappedNormalDistribution.sample(mu2, sigma2)
        return theta1, theta2
    
    def decode(self, theta1, theta2):
        z = torch.cat([torch.cos(theta1), torch.sin(theta1), 
                      torch.cos(theta2), torch.sin(theta2)], dim=1)
        return self.decoder(z)
    
    def forward(self, x):
        mu1, sigma1, mu2, sigma2 = self.encode(x)
        theta1, theta2 = self.reparameterize(mu1, sigma1, mu2, sigma2)
        x_recon = self.decode(theta1, theta2)
        return x_recon, mu1, sigma1, mu2, sigma2, theta1, theta2
    
    def loss(self, x, x_recon, mu1, sigma1, mu2, sigma2, beta=1.0):
        recon = torch.mean(torch.sum((x - x_recon)**2, dim=1))
        kl1 = torch.mean(WrappedNormalDistribution.kl_divergence(mu1, sigma1))
        kl2 = torch.mean(WrappedNormalDistribution.kl_divergence(mu2, sigma2))
        kl = kl1 + kl2
        return recon + beta * kl, recon, kl


class AnnulusVAE(nn.Module):
    """
    VAE for annular latent space with exact hole.
    
    Uses Wrapped Normal for the angular coordinate and Kumaraswamy distribution
    for the radius in [0, 1], which is then scaled to [r_min, r_max].
    
    The Kumaraswamy distribution provides:
    - Proper reparameterized sampling (differentiable)
    - Closed-form KL divergence to Uniform(0,1)
    - Bounded support [0, 1] naturally
    """
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128, 
                 r_min: float = 0.3, r_max: float = 1.0):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        
        self.encoder = build_mlp([input_dim, hidden_dim, hidden_dim, hidden_dim // 2])
        # Decoder receives Cartesian coordinates [x, y] = 2D
        self.decoder = build_mlp([2, hidden_dim // 2, hidden_dim, hidden_dim, input_dim])
        
        self.fc_cossin = nn.Linear(hidden_dim // 2, 2)
        self.fc_log_sigma = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_a_r = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_b_r = nn.Linear(hidden_dim // 2, 1)
    
    def encode(self, x):
        h = self.encoder(x)
        mu_angle = cossin_to_angle(self.fc_cossin(h))
        sigma = softplus_param(self.fc_log_sigma(h))
        a_r = softplus_param(self.fc_log_a_r(h))
        b_r = softplus_param(self.fc_log_b_r(h))
        return mu_angle, sigma, a_r, b_r
    
    def reparameterize(self, mu_angle, sigma, a_r, b_r):
        theta = WrappedNormalDistribution.sample(mu_angle, sigma)
        r_unit = KumaraswamyDistribution.sample(a_r, b_r)  # in [0, 1]
        r = self.r_min + (self.r_max - self.r_min) * r_unit
        return self.embed(theta, r), theta, r

    def embed(self, theta, r):
        """Place (θ, r) in the plane, which is what the decoder reads."""
        return torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

    def decode(self, theta, r):
        return self.decoder(self.embed(theta, r))

    def forward(self, x):
        mu_angle, sigma, a_r, b_r = self.encode(x)
        z_cart, theta, r = self.reparameterize(mu_angle, sigma, a_r, b_r)
        x_recon = self.decoder(z_cart)
        return x_recon, mu_angle, sigma, a_r, b_r, z_cart, theta, r
    
    def loss(self, x, x_recon, mu_angle, sigma, a_r, b_r, beta=1.0):
        recon = torch.mean(torch.sum((x - x_recon)**2, dim=1))
        kl_angle = torch.mean(WrappedNormalDistribution.kl_divergence(mu_angle, sigma))
        kl_r = torch.mean(KumaraswamyDistribution.kl_divergence(a_r, b_r))
        
        kl = kl_angle + kl_r
        return recon + beta * kl, recon, kl


class MobiusVAE(nn.Module):
    """
    VAE for Möbius strip using ℤ₂-INVARIANT decoder features.
    
    The Möbius strip is [0,1] × S¹ / ∼ where (h, θ) ∼ (1-h, θ+π).
    
    MATHEMATICALLY EXACT APPROACH:
    ==============================
    We use the cylinder as the covering space and build the decoder from
    features that are INVARIANT under the ℤ₂ action T(h, θ) = (1-h, θ+π).
    
    A function f(h, θ) is ℤ₂-invariant iff f(h, θ) = f(1-h, θ+π).
    
    INVARIANT FEATURES:
    1. cos(2θ): Under θ → θ+π: cos(2(θ+π)) = cos(2θ+2π) = cos(2θ)
    2. sin(2θ): Under θ → θ+π: sin(2(θ+π)) = sin(2θ+2π) = sin(2θ)
    3. (h-0.5)²: Under h → 1-h: ((1-h)-0.5)² = (0.5-h)² = (h-0.5)²
    4. cos(θ)(h-0.5): Under T: cos(θ+π)(0.5-h) = -cos(θ)·-(h-0.5) = cos(θ)(h-0.5)
    5. sin(θ)(h-0.5): Under T: sin(θ+π)(0.5-h) = -sin(θ)·-(h-0.5) = sin(θ)(h-0.5)
    
    These 5 features are a complete set of low-degree polynomial ℤ₂-invariants
    """
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128):
        super().__init__()
        
        # Extra encoder layer compared to other models (richer feature extraction)
        self.encoder = build_mlp([input_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim // 2])
        
        self.fc_cossin = nn.Linear(hidden_dim // 2, 2)
        self.fc_log_sigma = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_a = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_b = nn.Linear(hidden_dim // 2, 1)
        
        # Decoder receives 5 ℤ₂-invariant features; extra layer for expressivity
        self.decoder = build_mlp([5, hidden_dim // 2, hidden_dim, hidden_dim, hidden_dim, input_dim])
    
    def encode(self, x):
        h = self.encoder(x)
        mu_angle = cossin_to_angle(self.fc_cossin(h))
        sigma = softplus_param(self.fc_log_sigma(h))
        a = softplus_param(self.fc_log_a(h))
        b = softplus_param(self.fc_log_b(h))
        return mu_angle, sigma, a, b
    
    def reparameterize(self, mu_angle, sigma, a, b):
        theta = WrappedNormalDistribution.sample(mu_angle, sigma)
        h = KumaraswamyDistribution.sample(a, b)
        return theta, h
    
    def compute_invariant_features(self, theta, h):
        """Compute the 5 ℤ₂-invariant features for the decoder."""
        return mobius_invariant_features(theta, h, backend='torch')
    
    def decode(self, theta, h):
        features = self.compute_invariant_features(theta, h)
        return self.decoder(features)
    
    def forward(self, x):
        mu_angle, sigma, a, b = self.encode(x)
        theta, h = self.reparameterize(mu_angle, sigma, a, b)
        x_recon = self.decode(theta, h)
        return x_recon, mu_angle, sigma, a, b, theta, h
    
    def loss(self, x, x_recon, mu_angle, sigma, a, b, theta, h, beta=1.0):
        """
        Compute the Möbius VAE loss with correct KL divergence on the quotient space.
        
        The Möbius strip M is the quotient of the cylinder C by the ℤ₂ action:
            τ(θ, h) = (θ + π, 1 - h)
        
        The correct KL divergence on M is:
            KL(Q_M || U_M) = KL(Q_C || U_C) - log(2) + E[log(1 + ρ)]
        
        where ρ = q_C(τ(θ,h)) / q_C(θ,h) is the "twin ratio".
        
        This accounts for the fact that the Möbius strip has half the area of the cylinder.
        When the encoder concentrates on one branch (ρ→0), the correction is -log(2).
        When the distribution is τ-symmetric (ρ=1), the correction is 0.
        """
        recon = torch.mean(torch.sum((x - x_recon)**2, dim=1))
        
        # Standard cylinder KL terms
        kl_angle = torch.mean(WrappedNormalDistribution.kl_divergence(mu_angle, sigma))
        kl_height = torch.mean(KumaraswamyDistribution.kl_divergence(a, b))
        kl_cylinder = kl_angle + kl_height
        
        # Compute the twin ratio ρ = q_C(θ+π, 1-h) / q_C(θ, h)
        # Using log-space for numerical stability
        log_q_angle_current = WrappedNormalDistribution.log_prob(theta, mu_angle, sigma)
        log_q_angle_twin = WrappedNormalDistribution.log_prob(theta + np.pi, mu_angle, sigma)
        log_q_height_current = KumaraswamyDistribution.log_prob(h, a, b)
        log_q_height_twin = KumaraswamyDistribution.log_prob(1 - h, a, b)
        
        log_rho = (log_q_angle_twin + log_q_height_twin) - (log_q_angle_current + log_q_height_current)
        
        # Correction term: -log(2) + E[log(1 + ρ)]
        # Use log1p(exp(log_rho)) = log(1 + exp(log_rho)) for numerical stability
        # But need to handle large positive log_rho: log(1 + exp(x)) ≈ x for large x
        # correction = -np.log(2) + torch.mean(torch.log1p(torch.exp(torch.clamp(log_rho, max=20))))
        # Alternatively, we can use softplus which is numerically stable and has the same behavior:
        correction = -np.log(2) + torch.mean(torch.nn.functional.softplus(log_rho))
        
        # Full Möbius KL
        kl = kl_cylinder + correction

        # clamp the KL from below to prevent negative values due to sample-based correction.
        kl = torch.nn.functional.relu(kl)
        
        return recon + beta * kl, recon, kl


# =============================================================================
# IMAGE EXPERIMENTS — 28×28 MNIST DIGITS, CNN BACKBONE
# =============================================================================


class MNISTGaussianVAE(nn.Module):
    """
    CNN-based Gaussian VAE for MNIST images (28×28 = 784 pixels).

    Baseline model for both MNIST topology experiments. Uses the shared
    CNN backbone (_build_cnn_encoder / _build_cnn_decoder) instead of a
    flat MLP, which is more appropriate for image data.

    When latent_dim=4 and the first two pairs of dimensions are interpreted
    as (cos θ₁, sin θ₁, cos θ₂, sin θ₂), this model serves as the
    "sin/cos trick" baseline.
    """

    def __init__(self, latent_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = build_cnn_encoder(hidden_dim)           # → [batch, hidden_dim//2]
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = build_cnn_decoder(latent_dim, hidden_dim)  # → [batch, 784]

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def loss(self, x, x_recon, mu, logvar, beta=1.0):
        recon = torch.mean(torch.sum((x - x_recon) ** 2, dim=1))
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))
        return recon + beta * kl, recon, kl


class MixedCircleVAE(nn.Module):
    """
    CNN-based ℝ^k × S¹ VAE for all-digit rotated MNIST.

    Latent space: z_d ∈ ℝ^k (Gaussian) × θ ∈ S¹ (Wrapped Normal).

    The Gaussian subspace captures digit identity and within-class style.
    The circular coordinate θ captures the rotation angle.

    KL divergence decouples:
        KL = KL_Gauss(z_d) + KL_WN(θ)

    Decoder input: [z_d (k dims), cos θ, sin θ].

    Anchoring strategy: ONLY θ is anchored. The Gaussian z_d is never
    anchored directly — digit organisation emerges from reconstruction
    pressure alone. This avoids all heuristics about digit identity while
    still fixing the rotation reference frame.
    """

    def __init__(self, gaussian_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.gaussian_dim = gaussian_dim
        self.encoder = build_cnn_encoder(hidden_dim)
        # Gaussian heads
        self.fc_z_mu = nn.Linear(hidden_dim // 2, gaussian_dim)
        self.fc_z_logvar = nn.Linear(hidden_dim // 2, gaussian_dim)
        # Circle head
        self.fc_cossin = nn.Linear(hidden_dim // 2, 2)
        self.fc_log_sigma = nn.Linear(hidden_dim // 2, 1)
        # Decoder: [z_d (k), cos θ, sin θ]
        self.decoder = build_cnn_decoder(gaussian_dim + 2, hidden_dim)

    def encode(self, x):
        h = self.encoder(x)
        z_mu = self.fc_z_mu(h)
        z_logvar = self.fc_z_logvar(h)
        mu_theta = cossin_to_angle(self.fc_cossin(h))
        sigma_theta = softplus_param(self.fc_log_sigma(h))
        return z_mu, z_logvar, mu_theta, sigma_theta

    def reparameterize(self, z_mu, z_logvar, mu_theta, sigma_theta):
        std = torch.exp(0.5 * z_logvar)
        z_d = z_mu + std * torch.randn_like(std)
        theta = WrappedNormalDistribution.sample(mu_theta, sigma_theta)
        return z_d, theta

    def decode(self, z_d, theta):
        features = torch.cat([z_d, torch.cos(theta), torch.sin(theta)], dim=1)
        return self.decoder(features)

    def forward(self, x):
        z_mu, z_logvar, mu_theta, sigma_theta = self.encode(x)
        z_d, theta = self.reparameterize(z_mu, z_logvar, mu_theta, sigma_theta)
        x_recon = self.decode(z_d, theta)
        return x_recon, z_mu, z_logvar, mu_theta, sigma_theta, z_d, theta

    def loss(self, x, x_recon, z_mu, z_logvar, mu_theta, sigma_theta, beta=1.0):
        recon = torch.mean(torch.sum((x - x_recon) ** 2, dim=1))
        kl_z = -0.5 * torch.mean(torch.sum(
            1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim=1))
        kl_theta = torch.mean(WrappedNormalDistribution.kl_divergence(mu_theta, sigma_theta))
        kl = kl_z + kl_theta
        return recon + beta * kl, recon, kl


class MixedTorusVAE(nn.Module):
    """
    CNN-based ℝ^k × T² VAE for all-digit shifted MNIST.

    Latent space: z_d ∈ ℝ^k (Gaussian) × (θ₁, θ₂) ∈ T² (two Wrapped Normals).

    The Gaussian subspace z_d organises digit identity and intra-class
    variation purely through reconstruction pressure — it is never anchored.
    The toroidal subspace (θ₁, θ₂) captures the two periodic pixel shifts.

    KL divergence decouples (Proposition 1 of the paper):
        KL = KL_Gauss(z_d) + KL_WN(θ₁) + KL_WN(θ₂)

    Decoder input: [z_d (k dims), cos θ₁, sin θ₁, cos θ₂, sin θ₂].

    Anchoring: only (θ₁, θ₂) are anchored using canonical samples from
    exact integer pixel shifts. z_d is NEVER anchored.
    """

    def __init__(self, gaussian_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.gaussian_dim = gaussian_dim
        self.encoder = build_cnn_encoder(hidden_dim)
        # Gaussian heads
        self.fc_z_mu = nn.Linear(hidden_dim // 2, gaussian_dim)
        self.fc_z_logvar = nn.Linear(hidden_dim // 2, gaussian_dim)
        # Torus heads
        self.fc_cossin1 = nn.Linear(hidden_dim // 2, 2)
        self.fc_log_sigma1 = nn.Linear(hidden_dim // 2, 1)
        self.fc_cossin2 = nn.Linear(hidden_dim // 2, 2)
        self.fc_log_sigma2 = nn.Linear(hidden_dim // 2, 1)
        # Decoder: [z_d (k), cos θ₁, sin θ₁, cos θ₂, sin θ₂]
        self.decoder = build_cnn_decoder(gaussian_dim + 4, hidden_dim)

    def encode(self, x):
        h = self.encoder(x)
        z_mu = self.fc_z_mu(h)
        z_logvar = self.fc_z_logvar(h)
        mu1 = cossin_to_angle(self.fc_cossin1(h))
        sigma1 = softplus_param(self.fc_log_sigma1(h))
        mu2 = cossin_to_angle(self.fc_cossin2(h))
        sigma2 = softplus_param(self.fc_log_sigma2(h))
        return z_mu, z_logvar, mu1, sigma1, mu2, sigma2

    def reparameterize(self, z_mu, z_logvar, mu1, sigma1, mu2, sigma2):
        std = torch.exp(0.5 * z_logvar)
        z_d = z_mu + std * torch.randn_like(std)
        theta1 = WrappedNormalDistribution.sample(mu1, sigma1)
        theta2 = WrappedNormalDistribution.sample(mu2, sigma2)
        return z_d, theta1, theta2

    def decode(self, z_d, theta1, theta2):
        features = torch.cat([
            z_d,
            torch.cos(theta1), torch.sin(theta1),
            torch.cos(theta2), torch.sin(theta2),
        ], dim=1)
        return self.decoder(features)

    def forward(self, x):
        z_mu, z_logvar, mu1, sigma1, mu2, sigma2 = self.encode(x)
        z_d, theta1, theta2 = self.reparameterize(z_mu, z_logvar, mu1, sigma1, mu2, sigma2)
        x_recon = self.decode(z_d, theta1, theta2)
        return x_recon, z_mu, z_logvar, mu1, sigma1, mu2, sigma2, z_d, theta1, theta2

    def loss(self, x, x_recon, z_mu, z_logvar, mu1, sigma1, mu2, sigma2, beta=1.0):
        recon = torch.mean(torch.sum((x - x_recon) ** 2, dim=1))
        kl_z = -0.5 * torch.mean(torch.sum(
            1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim=1))
        kl1 = torch.mean(WrappedNormalDistribution.kl_divergence(mu1, sigma1))
        kl2 = torch.mean(WrappedNormalDistribution.kl_divergence(mu2, sigma2))
        kl = kl_z + kl1 + kl2
        return recon + beta * kl, recon, kl
