"""
Building blocks shared by the VAE models.

Three kinds of thing live here:

- Network builders.  ``build_mlp`` for the synthetic experiments, where an
  observation is a 50-dimensional vector, and ``build_cnn_encoder`` /
  ``build_cnn_decoder`` for the image experiments.
- Head activations.  ``cossin_to_angle`` turns an unconstrained 2-vector into
  an angle on S¹; ``softplus_param`` keeps a scale parameter positive.
- Manifold geometry.  ``mobius_invariant_features`` and
  ``canonicalize_to_mobius`` implement the ℤ₂ quotient of the Möbius strip, and
  ``embed_on_cylinder`` / ``embed_on_torus`` / ``embed_on_mobius`` place a
  latent coordinate on the corresponding surface in ℝ³.
- Manifold distances.  ``circular_distance`` wraps on S¹ and
  ``mobius_z2_distance`` minimises over the ℤ₂ orbit.  Both auto-detect
  torch or numpy input.

Author: Jilles van Hulst
"""

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# NETWORK BUILDERS
# =============================================================================


def build_mlp(dims):
    """Build an MLP: Linear → ReLU → ... → Linear (no activation on last layer)."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def build_cnn_encoder(hidden_dim: int, image_size: int = 28):
    """
    Build a CNN encoder: [batch, image_size²] → [batch, hidden_dim // 2].

    Architecture (for 28×28):
        Unflatten → Conv(1→32) → Conv(32→64) → Conv(64→128) → Flatten → Linear
    The output is hidden_dim // 2 so that existing model heads can be reused.
    """
    return nn.Sequential(
        nn.Unflatten(1, (1, image_size, image_size)),
        nn.Conv2d(1, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(1),
        nn.Linear(128 * 7 * 7, hidden_dim // 2),
        nn.ReLU(),
    )


def build_cnn_decoder(latent_input_dim: int, hidden_dim: int, image_size: int = 28):
    """
    Build a CNN decoder: [batch, latent_input_dim] → [batch, image_size²].

    Architecture (for 28×28):
        Linear → Unflatten → ConvT(128→64) → ConvT(64→32) → ConvT(32→1) → Flatten
    """
    return nn.Sequential(
        nn.Linear(latent_input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 128 * 7 * 7),
        nn.ReLU(),
        nn.Unflatten(1, (128, 7, 7)),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
        nn.Sigmoid(),
        nn.Flatten(1),
    )


# =============================================================================
# HEAD ACTIVATIONS
# =============================================================================


def cossin_to_angle(cossin_raw):
    """Normalize (cos, sin) output to unit circle and convert to angle via atan2."""
    norm = torch.sqrt(torch.sum(cossin_raw ** 2, dim=1, keepdim=True) + 1e-8)
    normalized = cossin_raw / norm
    return torch.atan2(normalized[:, 1:2], normalized[:, 0:1])


def softplus_param(raw):
    """Softplus activation with floor (for σ, a, b parameters)."""
    return torch.nn.functional.softplus(raw) + 0.01


# =============================================================================
# MÖBIUS STRIP: ℤ₂-INVARIANT COORDINATE SYSTEM
# =============================================================================


def mobius_invariant_features(theta, h, backend='auto'):
    """
    Compute 5 ℤ₂-invariant features for the Möbius strip.
    
    These features satisfy f(h, θ) = f(1-h, θ+π) exactly:
    - cos(2θ), sin(2θ): invariant under θ → θ+π
    - (h-0.5)²: invariant under h → 1-h  
    - cos(θ)(h-0.5), sin(θ)(h-0.5): both factors flip sign under the ℤ₂ action
    
    Args:
        theta: Angles in [-π, π], shape [N, 1] or [N] (torch) or 1D array (numpy)
        h: Heights in [0, 1], shape [N, 1] or [N] (torch) or 1D array (numpy)
        backend: 'torch', 'numpy', or 'auto' (auto-detect from input type)
    
    Returns:
        Tensor/array of shape [N, 5] with invariant features
    """
    if backend == 'auto':
        backend = 'torch' if isinstance(theta, torch.Tensor) else 'numpy'
    
    is_torch = (backend == 'torch')
    _cos = torch.cos if is_torch else np.cos
    _sin = torch.sin if is_torch else np.sin
    
    # Ensure column vectors
    if is_torch:
        if theta.dim() == 1: theta = theta.unsqueeze(1)
        if h.dim() == 1: h = h.unsqueeze(1)
    else:
        theta = np.atleast_1d(theta)
        h = np.atleast_1d(h)
    
    # Shared computation
    h_centered = h - 0.5
    feat1 = _cos(2 * theta)
    feat2 = _sin(2 * theta)
    feat3 = h_centered ** 2
    feat4 = _cos(theta) * h_centered
    feat5 = _sin(theta) * h_centered
    
    # Concatenate (shape handling differs)
    if is_torch:
        return torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
    else:
        return np.stack([feat1, feat2, feat3, feat4, feat5], axis=1)


def canonicalize_to_mobius(theta, h):
    """
    Canonicalize cylinder coordinates (θ, h) to the Möbius fundamental domain
    and compute embedding coordinates for 3D visualization.
    
    The Möbius strip M = (S¹ × [0,1]) / ℤ₂ with identification τ(θ, h) = (θ+π, 1-h).
    
    We choose the fundamental domain θ ∈ [0, π), h ∈ [0, 1]:
        - If θ ∈ [0, π): keep (θ, h)
        - If θ ∈ [-π, 0) (equivalently [π, 2π)): apply τ to get (θ+π, 1-h)
    
    For 3D embedding, we need to DOUBLE the canonical angle since the fundamental
    domain only covers half the angular range:
        θ_embed = 2 * θ_canonical maps [0, π) → [0, 2π)
    
    The "Möbius angle" θ_M = atan2(sin(2θ), cos(2θ)) is ℤ₂-invariant and used for coloring.
    
    Args:
        theta: Angles in [-π, π] (cylinder coordinates)
        h: Heights in [0, 1]
    
    Returns:
        theta_embed: Embedding angle in [-π, π] (ready for embed_on_mobius)
        h_canonical: Height in [0, 1]
        theta_mobius: Möbius angle in [-π, π] (ℤ₂-invariant, for coloring)
    """
    theta = np.asarray(theta).copy()
    h = np.asarray(h).copy()
    
    # Normalize theta to [0, 2π) for easier handling
    theta_normalized = np.mod(theta, 2 * np.pi)  # Now in [0, 2π)
    
    # Points in [π, 2π) need the ℤ₂ transformation to get to [0, π)
    mask = theta_normalized >= np.pi
    theta_normalized[mask] = theta_normalized[mask] - np.pi  # θ → θ - π (now in [0, π))
    h[mask] = 1 - h[mask]  # h → 1 - h
    
    # theta_normalized is now in [0, π) - the fundamental domain
    # DOUBLE the angle to map [0, π) → [0, 2π) for the Möbius embedding
    theta_doubled = 2 * theta_normalized
    
    # Map back to [-π, π] for consistency with embed_on_mobius
    theta_embed = theta_doubled - np.pi  # Now in [-π, π]
    
    # Compute Möbius angle for coloring: θ_M = 2θ (ℤ₂-invariant)
    # This is the same as 2*theta_normalized = theta_doubled, just wrapped
    theta_mobius = np.arctan2(np.sin(theta_doubled), np.cos(theta_doubled))
    
    return theta_embed, h, theta_mobius


# =============================================================================
# MANIFOLD EMBEDDINGS
# =============================================================================


def embed_on_cylinder(theta, h):
    """Embed (θ, h) coordinates onto the 3D cylinder."""
    x = np.cos(theta)
    y = np.sin(theta)
    z = h
    return x, y, z


def embed_on_torus(theta1, theta2, R=2.0, r=0.8):
    """
    Embed (θ₁, θ₂) coordinates onto the 3D torus.
    
    Args:
        theta1: Major angle (around the hole)
        theta2: Minor angle (around the tube)
        R: Major radius (default: 2.0)
        r: Minor radius (default: 0.8)
    
    Returns:
        x, y, z: 3D coordinates
    """
    x = (R + r * np.cos(theta2)) * np.cos(theta1)
    y = (R + r * np.cos(theta2)) * np.sin(theta1)
    z = r * np.sin(theta2)
    return x, y, z


def embed_on_mobius(theta, h, w=0.5):
    """
    Embed (θ, h) coordinates onto the 3D Möbius strip using the standard parametrization.
    
    IMPORTANT: This function expects CANONICALIZED coordinates where the input has
    already been mapped to the fundamental domain. Use canonicalize_to_mobius() first.
    
    The parametrization is:
        x = (1 + w*(h-0.5)*cos(θ/2)) * cos(θ)
        y = (1 + w*(h-0.5)*cos(θ/2)) * sin(θ)  
        z = w*(h-0.5)*sin(θ/2)
    
    where θ ∈ [0, 2π] goes once around the strip and h ∈ [0, 1] is the width coordinate.
    The half-angle terms cos(θ/2), sin(θ/2) create the characteristic Möbius twist.
    
    Args:
        theta: Angles - expects range [0, 2π] for full strip coverage
               (use canonicalize_to_mobius first, which outputs doubled angles)
        h: Heights in [0, 1]
        w: Width parameter for the strip (default 0.5)
    
    Returns:
        x, y, z: 3D coordinates on the Möbius strip surface
    """
    theta = np.asarray(theta)
    h = np.asarray(h)
    
    # Shift theta from [-π, π] to [0, 2π] for the standard parametrization
    u = theta + np.pi  # Now u ∈ [0, 2π]
    v = h - 0.5        # v ∈ [-0.5, 0.5]
    
    x = (1 + w * v * np.cos(u / 2)) * np.cos(u)
    y = (1 + w * v * np.cos(u / 2)) * np.sin(u)
    z = w * v * np.sin(u / 2)
    
    return x, y, z


# =============================================================================
# MANIFOLD DISTANCES
# =============================================================================


def circular_distance(theta1, theta2):
    """
    Compute circular distance between angles, handling periodicity.
    Works for both torch and numpy.
    
    Returns distance in [0, π].
    """
    is_torch = isinstance(theta1, torch.Tensor)
    _abs = torch.abs if is_torch else np.abs
    _atan2 = torch.atan2 if is_torch else np.arctan2
    _sin = torch.sin if is_torch else np.sin
    _cos = torch.cos if is_torch else np.cos
    
    diff = theta1 - theta2
    return _abs(_atan2(_sin(diff), _cos(diff)))


def mobius_z2_distance(theta1, h1, theta2, h2):
    """
    Compute ℤ₂-invariant distance on the Möbius strip.
    
    The Möbius strip has identification (h, θ) ~ (1-h, θ+π).
    Returns the minimum distance over the two representatives.
    
    Works with both PyTorch tensors and NumPy arrays (auto-detected).
    
    Args:
        theta1, h1: First set of coordinates [N, 1] or [N]
        theta2, h2: Second set of coordinates [N, 1] or [N]
    
    Returns:
        Distance tensor/array taking minimum over ℤ₂ orbit
    """
    is_torch = isinstance(theta1, torch.Tensor)
    _min = torch.minimum if is_torch else np.minimum
    _sqrt = torch.sqrt if is_torch else np.sqrt
    
    # Distance to first representative (θ₂, h₂)
    d_theta1 = circular_distance(theta1, theta2)
    dist1_sq = d_theta1**2 + (h1 - h2)**2
    
    # Distance to ℤ₂-equivalent representative (θ₂+π, 1-h₂)
    d_theta2 = circular_distance(theta1, theta2 + np.pi)
    dist2_sq = d_theta2**2 + (h1 - (1 - h2))**2
    
    return _sqrt(_min(dist1_sq, dist2_sq))
