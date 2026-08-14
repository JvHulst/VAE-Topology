"""
The four synthetic datasets.

Each one samples true coordinates uniformly on a manifold, embeds them in ℝ³ (or
ℝ⁴ for the torus), and pushes that through a fixed random smooth map into a
50-dimensional observation vector with additive noise. The observations are all
a model ever sees; the true coordinates are kept only so the figures and the
geodesic-stress metric have something to compare against.

The projection seed is fixed separately from the sampling seed, so the same
observation map is reused across runs that differ only in how many samples they
draw.

Author: Jilles van Hulst
"""

import numpy as np

from topovae.layers import mobius_invariant_features


# =============================================================================
# OBSERVATION MAP
# =============================================================================


def project_to_observations(coords, obs_dim=50, noise_std=0.1, seed=None):
    """
    Project low-dimensional coordinates to high-dimensional observation space.
    
    This applies a 3-layer neural network-like transformation with:
    - Layer 1: tanh nonlinearity
    - Layer 2: LeakyReLU nonlinearity  
    - Layer 3: tanh nonlinearity + noise
    
    This is the shared observation function used across all experiments
    to create a realistic inverse problem.
    
    Args:
        coords: Low-dimensional coordinates [N, D] where D is 2 or 3
        obs_dim: Dimension of observation space (default 50)
        noise_std: Standard deviation of observation noise (default 0.1)
        seed: Random seed for weight generation (for reproducibility)
    
    Returns:
        observations: High-dimensional observations [N, obs_dim]
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    n_samples = coords.shape[0]
    input_dim = coords.shape[1]
    
    # Layer 1: input_dim → 30
    W1 = rng.randn(30, input_dim) * 0.8
    h1 = np.tanh(coords @ W1.T)
    
    # Layer 2: 30 → 40 with LeakyReLU
    W2 = rng.randn(40, 30) * 0.5
    h2_pre = h1 @ W2.T
    h2 = np.where(h2_pre > 0, h2_pre, 0.2 * h2_pre)
    
    # Layer 3: 40 → obs_dim with tanh + noise
    W3 = rng.randn(obs_dim, 40) * 0.3
    obs = np.tanh(h2 @ W3.T) + noise_std * rng.randn(n_samples, obs_dim)
    
    return obs.astype(np.float32)


def train_test_split(data, test_ratio=0.2, seed=42, return_indices=False):
    """
    Split data dictionary into train and test sets.

    The split shuffles, so a row of ``train_data`` is at a different position
    than in the full ``data``. When you need to map a train-set position back to
    the full dataset — for example to mark the training anchors on a scatter of
    the whole latent space — pass ``return_indices=True`` to also get the index
    arrays, and index with ``train_idx[i]``.

    Args:
        data: dict with 'x' key containing observations and other coordinate keys
        test_ratio: fraction of data to use for testing
        seed: random seed for reproducibility
        return_indices: also return (train_idx, test_idx) into the full dataset

    Returns:
        train_data, test_data (and train_idx, test_idx if return_indices)
    """
    np.random.seed(seed)

    n = len(data['x'])

    indices = np.random.permutation(n)
    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_data = {}
    test_data = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            train_data[key] = val[train_idx]
            test_data[key] = val[test_idx]
        else:
            # Keep scalar values (like r_min, r_max) in both
            train_data[key] = val
            test_data[key] = val

    if return_indices:
        return train_data, test_data, train_idx, test_idx
    return train_data, test_data


# =============================================================================
# THE FOUR MANIFOLDS
# =============================================================================


def generate_cylinder_data(n_samples: int = 2000, obs_dim: int = 50, seed: int = 42) -> dict:
    """Generate data from a cylinder S^1 x [0, 1]."""
    rng = np.random.RandomState(seed)

    theta = rng.uniform(-np.pi, np.pi, n_samples)
    h = rng.uniform(0, 1, n_samples)
    coords = np.column_stack([np.cos(theta), np.sin(theta), h])
    obs = project_to_observations(coords, obs_dim=obs_dim, noise_std=0.1, seed=123)

    return {
        "theta": theta,
        "h": h,
        "x": obs,
    }


def generate_torus_data(n_samples: int = 2000, obs_dim: int = 50, seed: int = 42) -> dict:
    """Generate data from a torus T^2 = S^1 x S^1."""
    rng = np.random.RandomState(seed)

    theta1 = rng.uniform(-np.pi, np.pi, n_samples)
    theta2 = rng.uniform(-np.pi, np.pi, n_samples)

    R, r = 2.0, 0.8
    x = (R + r * np.cos(theta2)) * np.cos(theta1)
    y = (R + r * np.cos(theta2)) * np.sin(theta1)
    z = r * np.sin(theta2)

    coords = np.column_stack([x, y, z])
    obs = project_to_observations(coords, obs_dim=obs_dim, noise_std=0.1, seed=124)

    return {
        "theta1": theta1,
        "theta2": theta2,
        "x": obs,
    }


def generate_mobius_data(n_samples: int = 2000, obs_dim: int = 50, seed: int = 42, noise_std: float = 0.05):
    """Generate synthetic data on a Möbius strip."""
    rng = np.random.RandomState(seed)

    theta = rng.uniform(-np.pi, np.pi, n_samples)
    h = rng.uniform(0, 1, n_samples)
    invariant_features = mobius_invariant_features(theta, h, backend="numpy")
    obs = project_to_observations(invariant_features, obs_dim=obs_dim, noise_std=noise_std, seed=126)
    theta_mobius = np.arctan2(np.sin(2 * theta), np.cos(2 * theta))

    return {
        "x": obs,
        "theta": theta.astype(np.float32),
        "h": h.astype(np.float32),
        "theta_mobius": theta_mobius.astype(np.float32),
    }


def generate_annulus_data(
    n_samples: int = 2000,
    r_min: float = 0.3,
    r_max: float = 1.0,
    obs_dim: int = 50,
    seed: int = 42,
) -> dict:
    """Generate data from an annulus with a hole."""
    rng = np.random.RandomState(seed)

    theta = rng.uniform(-np.pi, np.pi, n_samples)
    r = rng.uniform(r_min, r_max, n_samples)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    obs = project_to_observations(coords, obs_dim=obs_dim, noise_std=0.1, seed=125)

    return {
        "theta": theta,
        "r": r,
        "r_min": r_min,
        "r_max": r_max,
        "x": obs,
    }
