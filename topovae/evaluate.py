"""
Measuring a trained model, and reading/writing its run on disk.

Two things live here.  The metrics compare a learned latent space against the
true one — reconstruction RMS, KL against the prior, and geodesic stress, the
one that actually sees topology.  The run-artifact IO reads and writes the
self-describing run directory (run.json, checkpoints, latents, metrics) that
lets training and rendering be separate steps.

Distances are taken on the manifold, not in ambient coordinates: see
``circular_distance`` and ``mobius_z2_distance`` in ``layers``.

Author: Jilles van Hulst
"""

import csv
import json
import os

import numpy as np
import torch

from .layers import mobius_z2_distance
from .train import _mnist_forward_loss, get_latent_codes, get_reconstruction


# =============================================================================
# METRICS
# =============================================================================


def geodesic_distance(z1, z2, manifold):
    """
    Compute geodesic distance on the specified manifold.
    
    Args:
        z1, z2: coordinate arrays of shape [N, D]
                 Column layout depends on manifold:
                   euclidean : [x₁, x₂, ...]
                   cylinder  : [θ, h]
                   mobius    : [θ, h]  (ℤ₂ quotient applied automatically)
                   torus     : [θ₁, θ₂]
                   annulus   : [θ, r]
        manifold: one of 'euclidean', 'cylinder', 'mobius', 'torus', 'annulus'
    
    Returns:
        Array of pairwise distances [N]
    """
    if manifold == 'euclidean':
        return np.sqrt(np.sum((z1 - z2)**2, axis=1))
    
    elif manifold == 'circle':
        # S¹: geodesic = shortest arc
        return np.abs(np.angle(np.exp(1j * (z1[:, 0] - z2[:, 0]))))
    
    elif manifold == 'cylinder':
        theta1, h1 = z1[:, 0], z1[:, 1]
        theta2, h2 = z2[:, 0], z2[:, 1]
        d_theta = np.abs(np.angle(np.exp(1j * (theta1 - theta2))))
        d_h = np.abs(h1 - h2)
        return np.sqrt(d_theta**2 + d_h**2)
    
    elif manifold == 'mobius':
        theta1, h1 = z1[:, 0], z1[:, 1]
        theta2, h2 = z2[:, 0], z2[:, 1]
        return mobius_z2_distance(theta1, h1, theta2, h2)
    
    elif manifold == 'torus':
        d_t1 = np.abs(np.angle(np.exp(1j * (z1[:, 0] - z2[:, 0]))))
        d_t2 = np.abs(np.angle(np.exp(1j * (z1[:, 1] - z2[:, 1]))))
        return np.sqrt(d_t1**2 + d_t2**2)
    
    elif manifold == 'annulus':
        x1 = z1[:, 1] * np.cos(z1[:, 0])
        y1 = z1[:, 1] * np.sin(z1[:, 0])
        x2 = z2[:, 1] * np.cos(z2[:, 0])
        y2 = z2[:, 1] * np.sin(z2[:, 0])
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    else:
        raise ValueError(f"Unknown manifold: {manifold}")


# =============================================================================
# SYNTHETIC EXPERIMENTS
# =============================================================================


# Column layout for each manifold: maps manifold name → dict keys
_MANIFOLD_KEYS = {
    'euclidean': ('z',),           # single array, already [N, D]
    'circle':    ('theta',),       # S¹: single angle
    'cylinder':  ('theta', 'h'),
    'mobius':    ('theta', 'h'),
    'torus':     ('theta1', 'theta2'),
    'annulus':   ('theta', 'r'),
}


def _extract_pairs(data_dict, idx1, idx2, manifold):
    """
    Extract coordinate-pair arrays from a keyed dictionary.
    
    Works for both true_factors dicts and latent-code dicts returned by
    get_latent_codes(), because both use the same key names.
    
    Returns:
        z1, z2: arrays of shape [N, D]
    """
    keys = _MANIFOLD_KEYS[manifold]
    if len(keys) == 1 and keys[0] == 'z':
        # Euclidean: 'z' is already an [N, D] array
        return data_dict['z'][idx1], data_dict['z'][idx2]
    else:
        z1 = np.column_stack([data_dict[k][idx1] for k in keys])
        z2 = np.column_stack([data_dict[k][idx2] for k in keys])
        return z1, z2


def _get_model_manifold(model):
    """Map a VAE model instance to its latent manifold name."""
    from .models import (GaussianVAE, CylinderVAE, TorusVAE, AnnulusVAE,
                         MobiusVAE)
    _MAP = [
        (GaussianVAE,    'euclidean'),
        (CylinderVAE,    'cylinder'),
        (MobiusVAE,      'mobius'),
        (TorusVAE,       'torus'),
        (AnnulusVAE,     'annulus'),
    ]
    for cls, name in _MAP:
        if isinstance(model, cls):
            return name
    raise ValueError(f"Unknown model type: {type(model)}")


def _eval_with_kl(model, x_data):
    """
    Single forward pass returning both reconstruction and KL divergence.
    
    Avoids the redundancy of calling get_reconstruction() (one forward pass)
    and then computing KL (a second forward pass on the same data).
    
    Args:
        model: Trained VAE model (any type)
        x_data: Input data (numpy array)
    
    Returns:
        (recon, kl): reconstruction as numpy array, KL as float
    """
    from .models import (GaussianVAE, CylinderVAE, TorusVAE, AnnulusVAE,
                         MobiusVAE)
    
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    model.eval()
    
    with torch.no_grad():
        outputs = model(x_tensor)
        recon = outputs[0].numpy()
        
        # Dispatch model.loss() with the correct output subset.
        # Each model's forward() returns diagnostic extras that loss() doesn't need;
        # the slicing below selects only the arguments loss() expects.
        if isinstance(model, GaussianVAE):
            _, _, kl = model.loss(x_tensor, *outputs[:3], beta=1.0)
        elif isinstance(model, (CylinderVAE, TorusVAE, AnnulusVAE)):
            _, _, kl = model.loss(x_tensor, *outputs[:5], beta=1.0)
        elif isinstance(model, MobiusVAE):
            _, _, kl = model.loss(x_tensor, *outputs, beta=1.0)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
    
    return recon, kl.item()


def compute_all_metrics(model, x_train, x_test, beta=1.0, true_factors=None,
                        data_manifold=None, n_pairs=500):
    """
    Compute all metrics: reconstruction, KL divergence, and geodesic distortion.

    Metrics:
    - train_rms, test_rms: Reconstruction quality
    - beta: KL weight used during training
    - kl_divergence: Unweighted (beta=1.0) KL divergence on test set
    - geodesic_stress: Kruskal's stress measuring distance preservation

    Args:
        model: VAE model
        x_train: Training data
        x_test: Test data
        beta: KL weight (beta) used during training for this model
        true_factors: Optional dict with true latent factors for test data.
                     For cylinder/Möbius: {'theta': ..., 'h': ...}
                     For torus: {'theta1': ..., 'theta2': ...}
                     For annulus: {'theta': ..., 'r': ...}
                     If None, uses observation-space Euclidean distances.
        data_manifold: Topology of the true data manifold. One of
                     'euclidean', 'cylinder', 'mobius', 'torus', 'annulus'.
                     Required (together with true_factors) for correct
                     geodesic stress computation on the data side.
                     The model's latent manifold is inferred automatically.
        n_pairs: Number of point pairs for geodesic distortion

    Returns:
        dict with 'train_rms', 'test_rms', 'beta', 'kl_divergence', 'geodesic_stress'
    """
    # --- Reconstruction + KL (single forward pass per dataset) ---
    recon_train = get_reconstruction(model, x_train)
    train_rms = np.sqrt(np.mean((x_train - recon_train)**2))

    recon_test, kl_divergence = _eval_with_kl(model, x_test)
    test_rms = np.sqrt(np.mean((x_test - recon_test)**2))

    # --- Geodesic distortion (Kruskal's stress) ---
    n_test = len(x_test)
    idx1 = np.random.choice(n_test, n_pairs, replace=True)
    idx2 = np.random.choice(n_test, n_pairs, replace=True)
    
    # Input distances: geodesic on the true data manifold
    if true_factors is not None and data_manifold is not None:
        z1_true, z2_true = _extract_pairs(true_factors, idx1, idx2, data_manifold)
        d_input = geodesic_distance(z1_true, z2_true, data_manifold)
    else:
        # Fallback: Euclidean in observation space
        d_input = np.sqrt(np.sum((x_test[idx1] - x_test[idx2])**2, axis=1))
    
    # Latent distances: geodesic on the model's latent manifold
    model_manifold = _get_model_manifold(model)
    z_all = get_latent_codes(model, x_test)
    z1_lat, z2_lat = _extract_pairs(z_all, idx1, idx2, model_manifold)
    d_latent = geodesic_distance(z1_lat, z2_lat, model_manifold)
    
    # Normalize both to have same mean (compare shape, not scale)
    d_input_norm = d_input / (np.mean(d_input) + 1e-8)
    d_latent_norm = d_latent / (np.mean(d_latent) + 1e-8)
    
    # Kruskal's stress-1
    geodesic_stress = np.sqrt(np.sum((d_input_norm - d_latent_norm)**2) / np.sum(d_input_norm**2))
    
    return {
        'train_rms': train_rms,
        'test_rms': test_rms,
        'beta': beta,
        'kl_divergence': kl_divergence,
        'geodesic_stress': geodesic_stress
    }


def print_metrics_table(model_names, metrics_list, save_path=None):
    """
    Print a formatted table of all metrics.
    
    Args:
        model_names: List of model names
        metrics_list: List of metric dicts from compute_all_metrics
        save_path: If provided, save metrics to this CSV file
    """
    print("\n" + "="*110)
    print("RECONSTRUCTION AND TOPOLOGY METRICS")
    print("="*110)
    print(f"{'Model':<32} {'Train RMS':>12} {'Test RMS':>12} {'Beta':>12} {'KL Div':>12} {'Geo. Stress':>12}")
    print("-"*110)

    for name, metrics in zip(model_names, metrics_list):
        print(f"{name:<32} {metrics['train_rms']:>12.4f} {metrics['test_rms']:>12.4f} "
              f"{metrics['beta']:>12.4f} {metrics['kl_divergence']:>12.4f} {metrics['geodesic_stress']:>12.4f}")
    print("="*110)
    
    # Save to CSV
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("model,train_rms,test_rms,beta,kl_divergence,geodesic_stress\n")
            for name, metrics in zip(model_names, metrics_list):
                f.write(f"{name},{metrics['train_rms']:.6f},{metrics['test_rms']:.6f},{metrics['beta']:.6f},"
                        f"{metrics['kl_divergence']:.6f},{metrics['geodesic_stress']:.6f}\n")
        print(f"Saved metrics to {save_path}")


# =============================================================================
# IMAGE EXPERIMENTS
# =============================================================================


def _angular_pair_distances(angles, i, j):
    """Geodesic distance on (S¹)^n between the pairs (i, j) of a product of angles.

    Each factor contributes its shortest arc and the product metric adds the
    squares, so one array gives the circle and two give the torus.
    """
    d_squared = sum(np.angle(np.exp(1j * (a[i] - a[j]))) ** 2 for a in angles)
    return np.sqrt(d_squared)


def _latent_pair_distances(model, enc, i, j):
    """Geodesic distance between the pairs (i, j) in the model's own latent space.

    Each model is measured with the metric of the manifold it was given: the
    Euclidean norm on ℝ^ell for the Gaussian baseline, and the product metric on
    ℝ^k × (S¹)^n for the topology-aware models, whose circular factors wrap.
    """
    from .models import MNISTGaussianVAE, MixedCircleVAE, MixedTorusVAE

    if isinstance(model, MNISTGaussianVAE):
        z = enc[0].cpu().numpy()
        return np.sqrt(np.sum((z[i] - z[j]) ** 2, axis=-1))

    if isinstance(model, MixedCircleVAE):
        angle_slots = (2,)          # encode: (z_mu, z_logvar, mu_theta, sigma)
    elif isinstance(model, MixedTorusVAE):
        angle_slots = (2, 4)        # encode: (z_mu, z_logvar, mu1, s1, mu2, s2)
    else:
        raise ValueError(f"Geodesic stress not defined for {type(model)}")

    z_d = enc[0].cpu().numpy()
    angles = [enc[k].cpu().numpy().squeeze(-1) for k in angle_slots]
    d_squared = np.sum((z_d[i] - z_d[j]) ** 2, axis=-1)
    d_squared = d_squared + _angular_pair_distances(angles, i, j) ** 2
    return np.sqrt(d_squared)


def _within_group_pairs(groups, rng, n_per_group=30):
    """Index pairs drawn from within groups, never across them.

    The true latent variable is only fully known inside a group: two images from
    one group differ by the applied group element and by nothing else, so their
    distance on the data manifold is exactly the distance between those elements.
    Two images from different groups differ by an unlabelled factor as well, so
    no ground-truth distance exists for them and they are not compared.

    Every group contributes, since which groups are sampled otherwise moves the
    result by more than the differences being measured. Groups larger than
    `n_per_group` are thinned, which keeps the pair count linear in the number of
    groups rather than quadratic in their size.
    """
    idx_i, idx_j = [], []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        if len(members) > n_per_group:
            members = rng.choice(members, n_per_group, replace=False)
        a, b = np.triu_indices(len(members), k=1)
        idx_i.append(members[a])
        idx_j.append(members[b])
    return np.concatenate(idx_i), np.concatenate(idx_j)


def _encode_in_batches(model, x, batch_size):
    """Encoder outputs for every row of `x`, concatenated across batches."""
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunks.append(model.encode(x[start:start + batch_size]))
    return [torch.cat(parts) for parts in zip(*chunks)]


def _kruskal_stress(d_true, d_hat):
    """Kruskal's stress-1 over a set of pairs, each side mean-normalised.

    Normalising by the mean removes the scale difference between a latent space
    and the manifold it is compared against, leaving only the shape of the
    distance structure. One scale is used for all pairs, so a model must hold the
    same scale everywhere rather than being rescaled group by group.
    """
    d_true = d_true / d_true.mean()
    d_hat = d_hat / d_hat.mean()
    return float(np.sqrt(np.sum((d_true - d_hat) ** 2) / np.sum(d_true ** 2)))


def compute_mnist_metrics(model, train_loader, val_loader, beta, x_data=None,
                          true_angles=None, groups=None, batch_size=256):
    """
    Compute train/test reconstruction RMS, unweighted KL, and (optionally)
    geodesic stress for any of the image models.

    Args:
        model:        Trained image model.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        beta:         KL weight used during training.
        x_data:       Optional float32 [N, 784] array of images.  Required
                      together with true_angles and groups for geodesic stress.
        true_angles:  Optional list of float32 [N] arrays holding the applied
                      group element per image: one array for a circle, two for
                      a torus.
        groups:       Optional int [N] array naming the exemplar each image was
                      generated from.  Distances are only compared within one.
        batch_size:   Batch size used when encoding for the geodesic stress.

    Returns dict with keys: train_rms, test_rms, beta, kl_divergence,
    geodesic_stress.
    """
    model.eval()
    device = next(model.parameters()).device

    def _rms_and_kl(loader, compute_kl=False):
        sq_sum, n_samples_seen, kl_total, n_batches = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                _, recon_loss, kl = _mnist_forward_loss(model, x, beta=1.0)
                sq_sum += recon_loss.item() * x.size(0)
                n_samples_seen += x.size(0)
                if compute_kl:
                    kl_total += kl.item()
                    n_batches += 1
        n_pix = 784
        rms = float(np.sqrt(sq_sum / max(n_samples_seen * n_pix, 1)))
        kl_div = kl_total / max(n_batches, 1) if compute_kl else None
        return rms, kl_div

    train_rms, _ = _rms_and_kl(train_loader, compute_kl=False)
    test_rms, kl_div = _rms_and_kl(val_loader, compute_kl=True)

    # Geodesic stress: Kruskal's stress-1 between distances on the data manifold
    # and distances in the model's own latent space, over pairs of images
    # generated from the same exemplar. Within one exemplar the applied group
    # element is the only thing that differs, so the distance on the data
    # manifold is known exactly, and both sides can then be measured with the
    # true geodesic of the space they live in.
    geodesic_stress = float('nan')
    if x_data is not None and true_angles is not None and groups is not None:
        rng = np.random.RandomState(0)
        i, j = _within_group_pairs(groups, rng)

        keep = np.unique(np.concatenate([i, j]))
        position = np.full(len(groups), -1)
        position[keep] = np.arange(len(keep))

        x_keep = torch.tensor(x_data[keep], dtype=torch.float32, device=device)
        enc = _encode_in_batches(model, x_keep, batch_size)

        d_true = _angular_pair_distances(true_angles, i, j)
        d_hat = _latent_pair_distances(model, enc, position[i], position[j])
        geodesic_stress = _kruskal_stress(d_true, d_hat)

    return {
        'train_rms': train_rms,
        'test_rms': test_rms,
        'beta': float(beta),
        'kl_divergence': kl_div,
        'geodesic_stress': geodesic_stress,
    }


# =============================================================================
# RUN ARTIFACTS
# =============================================================================


def save_run_info(run_dir, info: dict) -> None:
    """Write *info* to ``<run_dir>/run.json``.  The directory is created if needed."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


def load_run_info(run_dir) -> dict:
    """Load and return ``<run_dir>/run.json`` as a dict."""
    path = os.path.join(run_dir, "run.json")
    with open(path) as f:
        return json.load(f)


def save_checkpoint_bundle(run_dir, models: dict) -> None:
    """Save one ``.pt`` file per model in ``<run_dir>/checkpoints/``.

    Args:
        run_dir: Path to the run directory.
        models:  Dict mapping a short name to a ``torch.nn.Module``.
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(ckpt_dir, f"{name}.pt")
        torch.save(model.state_dict(), path)


def load_checkpoint_bundle(run_dir, models: dict) -> None:
    """Load checkpoint weights from ``<run_dir>/checkpoints/`` into *models* in-place.

    Args:
        run_dir: Path to the run directory.
        models:  Dict mapping a short name to a ``torch.nn.Module``.
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    for name, model in models.items():
        path = os.path.join(ckpt_dir, f"{name}.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))


def save_latents(run_dir, **arrays) -> None:
    """Save named numpy arrays to ``<run_dir>/latents.npz``."""
    path = os.path.join(run_dir, "latents.npz")
    np.savez(path, **arrays)


def flatten_metrics(model_names, metrics_list) -> dict:
    """Flatten per-model metric dicts into the flat key,value form save_metrics writes.

    ``["Gaussian VAE"], [{"test_rms": 0.1}]``  →  ``{"gaussian_vae.test_rms": 0.1}``
    """
    flat = {}
    for model_name, metrics in zip(model_names, metrics_list):
        prefix = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        for key, value in metrics.items():
            flat[f"{prefix}.{key}"] = value
    return flat


def save_metrics(run_dir, metrics: dict) -> None:
    """Write scalar *metrics* to ``<run_dir>/metrics.csv`` (columns: metric, value)."""
    path = os.path.join(run_dir, "metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def get_figures_dir(run_dir) -> str:
    """Return the path to ``<run_dir>/figures/``, creating it if needed."""
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir
