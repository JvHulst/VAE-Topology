"""
The synthetic-experiment figures.

Each ``plot_*_experiment`` draws one row of panels: the true latent space, the
Gaussian VAE baseline, the topology-aware VAE, and the same VAE with anchoring.
The point of the row is the comparison — the Gaussian baseline tears the
manifold, the topology-aware model does not, and anchoring puts the result in
the same frame as the ground truth so the two can be read against each other.

The cylinder, torus and Möbius figures take ``flat=True`` for a 2-D scatter in
the intrinsic coordinates instead of the 3-D embedding. Both versions are drawn
for every run; the paper uses the 3-D one.

Author: Jilles van Hulst
"""

import matplotlib.pyplot as plt
import numpy as np

from topovae.train import get_reconstruction
from topovae.layers import canonicalize_to_mobius, embed_on_cylinder, embed_on_mobius, embed_on_torus

from .._panels import (
    _add_repel_panel,
    _plot_2d_panel,
    _plot_3d_panel,
    _plot_annulus_boundaries,
    _plot_gaussian_vae_panel,
    finalize_synthetic_row,
)
from .._style import (
    angle_to_color,
    apply_publication_style,
    save_figure,
    scale_fonts_to_page,
)

apply_publication_style()


# =============================================================================
# THE FOUR MANIFOLD FIGURES
# =============================================================================


def plot_cylinder_experiment(data, z_gaussian, z_cylinder_no_anchor, z_cylinder,
                             z_gaussian_repel=None, anchor_indices=None,
                             save_path='figures/fig_cylinder.pdf', flat=False):
    """Visualize cylinder experiment. Set flat=True for 2D (θ, h) scatter plots."""
    n_panels = 5 if z_gaussian_repel is not None else 4
    scale_fonts_to_page(4.5 * n_panels)
    fig = plt.figure(figsize=(4.5 * n_panels, 4.0 if flat else 4.5))
    colors = angle_to_color(data['theta'])

    def _cyl_panel(idx, theta, h, title, anch=None):
        if flat:
            ax = fig.add_subplot(1, n_panels, idx)
            is_true = (idx == 1)
            xl = r'$\theta$ (degrees)' if is_true else r'$z_\theta$ (degrees)'
            yl = r'$h$' if is_true else r'$z_h$'
            _plot_2d_panel(ax, np.degrees(theta), h, colors, xl, yl, title,
                           xlim=(-180, 180), ylim=(-0.05, 1.05), anchor_indices=anch)
            ax.set_aspect('auto')
        else:
            ax = fig.add_subplot(1, n_panels, idx, projection='3d')
            x, y, z = embed_on_cylinder(theta, h)
            is_true = (idx == 1)
            # No parentheses around a single symbol, matching the image
            # experiments in plotting_wip.py.
            xl = r'$\cos\theta$' if is_true else r'$\cos z_\theta$'
            yl = r'$\sin\theta$' if is_true else r'$\sin z_\theta$'
            zl = r'$h$' if is_true else r'$z_h$'
            _plot_3d_panel(ax, x, y, z, colors, xl, yl, zl, title,
                           aspect=[1, 1, 1], anchor_indices=anch)

    pi = 1  # panel index
    _cyl_panel(pi, data['theta'], data['h'], 'True Latent Space')
    pi += 1

    ax_g = fig.add_subplot(1, n_panels, pi)
    _plot_gaussian_vae_panel(ax_g, z_gaussian, colors)
    pi += 1

    if z_gaussian_repel is not None:
        _add_repel_panel(fig, n_panels, pi, z_gaussian_repel, colors)
        pi += 1

    _cyl_panel(pi, z_cylinder_no_anchor['theta'], z_cylinder_no_anchor['h'],
               'Cylinder VAE')
    pi += 1

    _cyl_panel(pi, z_cylinder['theta'], z_cylinder['h'],
               'Cylinder VAE (anchored)',
               anch=anchor_indices)

    if flat:
        plt.subplots_adjust(left=0.05, right=0.98, wspace=0.35)
    else:
        finalize_synthetic_row(fig)
    save_figure(save_path)


def plot_torus_experiment(data, z_gaussian, z_torus_no_anchor, z_torus,
                          anchor_indices=None, save_path='figures/fig_torus.pdf',
                          flat=False):
    """Visualize torus experiment. Set flat=True for 2D (θ₁, θ₂) scatter plots."""
    scale_fonts_to_page(18)
    fig = plt.figure(figsize=(18, 4.0 if flat else 4.5))
    colors = angle_to_color(data['theta1'])

    def _torus_panel(idx, t1, t2, title, anch=None):
        if flat:
            ax = fig.add_subplot(1, 4, idx)
            is_true = (idx == 1)
            xl = r'$\theta_1$ (degrees)' if is_true else r'$z_{\theta_1}$ (degrees)'
            yl = r'$\theta_2$ (degrees)' if is_true else r'$z_{\theta_2}$ (degrees)'
            _plot_2d_panel(ax, np.degrees(t1), np.degrees(t2), colors,
                           xl, yl, title, xlim=(-180, 180), ylim=(-180, 180),
                           anchor_indices=anch)
        else:
            ax = fig.add_subplot(1, 4, idx, projection='3d')
            x, y, z = embed_on_torus(t1, t2)
            _plot_3d_panel(ax, x, y, z, colors, r'$x$', r'$y$', r'$z$', title,
                           aspect=[1, 1, 0.4], anchor_indices=anch)

    _torus_panel(1, data['theta1'], data['theta2'],
                 'True Latent Space')

    ax_g = fig.add_subplot(1, 4, 2)
    _plot_gaussian_vae_panel(ax_g, z_gaussian, colors)

    _torus_panel(3, z_torus_no_anchor['theta1'], z_torus_no_anchor['theta2'],
                 'Torus VAE')
    _torus_panel(4, z_torus['theta1'], z_torus['theta2'],
                 'Torus VAE (anchored)', anch=anchor_indices)

    if flat:
        plt.subplots_adjust(left=0.05, right=0.98, wspace=0.35)
    else:
        finalize_synthetic_row(fig)
    save_figure(save_path)


def plot_annulus_experiment(data, z_gaussian, z_gaussian_repel, z_annulus_no_anchor, z_annulus,
                            anchor_indices=None, save_path='figures/fig_annulus.pdf'):
    """Visualize annulus experiment - 5 panels including repelling anchor VAE."""
    scale_fonts_to_page(22)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    
    colors = angle_to_color(data['theta'])
    r_min, r_max = data['r_min'], data['r_max']
    
    # Panel (a): True latent space
    ax = axes[0]
    x_true = data['r'] * np.cos(data['theta'])
    y_true = data['r'] * np.sin(data['theta'])
    _plot_2d_panel(ax, x_true, y_true, colors,
                   xlabel=r'$z_1$', ylabel=r'$z_2$',
                   title='True Latent Space',
                   xlim=(-1.3, 1.3), ylim=(-1.3, 1.3))
    _plot_annulus_boundaries(ax, r_min, r_max)

    # Panel (b): Gaussian VAE (no hole)
    ax = axes[1]
    _plot_gaussian_vae_panel(ax, z_gaussian, colors)

    # Panel (c): Gaussian VAE with repelling anchor (soft hole)
    ax = axes[2]
    z_repel = z_gaussian_repel['z']
    _plot_2d_panel(ax, z_repel[:, 0], z_repel[:, 1], colors,
                   xlabel=r'$z_1$', ylabel=r'$z_2$',
                   title='Gaussian VAE + repel',
                   xlim=(-3, 3), ylim=(-3, 3))
    ax.scatter([0], [0], c='red', s=100, marker='x', linewidths=2, zorder=10)

    # Panel (d): Annulus VAE without anchoring (exact hole)
    ax = axes[3]
    _plot_2d_panel(ax, z_annulus_no_anchor['z1'], z_annulus_no_anchor['z2'], colors,
                   xlabel=r'$z_1$', ylabel=r'$z_2$',
                   title='Annulus VAE',
                   xlim=(-1.3, 1.3), ylim=(-1.3, 1.3))
    _plot_annulus_boundaries(ax, r_min, r_max)

    # Panel (e): Annulus VAE with anchoring
    ax = axes[4]
    _plot_2d_panel(ax, z_annulus['z1'], z_annulus['z2'], colors,
                   xlabel=r'$z_1$', ylabel=r'$z_2$',
                   title='Annulus VAE (anchored)',
                   xlim=(-1.3, 1.3), ylim=(-1.3, 1.3),
                   anchor_indices=anchor_indices)
    _plot_annulus_boundaries(ax, r_min, r_max)
    
    # Manual spacing adjustment for all 2D panels
    plt.subplots_adjust(left=0.05, right=0.98, wspace=0.25)
    save_figure(save_path)


def plot_mobius_experiment(data, z_gaussian, z_mobius_no_anchor, z_mobius,
                           anchor_indices=None, save_path='figures/fig_mobius.pdf',
                           flat=False):
    """
    Visualize Möbius strip experiment. Set flat=True for 2D (θ, h) scatter plots.
    
    Colors are based on theta_mobius (the Möbius angle = 2θ), which is ℤ₂-invariant.
    This ensures that identified points (θ, h) and (θ+π, 1-h) get the same color.
    """
    scale_fonts_to_page(18)
    fig = plt.figure(figsize=(18, 4.0 if flat else 4.5))

    if 'theta_mobius' in data:
        color_angle = data['theta_mobius']
    else:
        color_angle = np.arctan2(np.sin(2 * data['theta']), np.cos(2 * data['theta']))
    colors = angle_to_color(color_angle)

    def _mobius_panel(idx, theta, h, title, anch=None):
        tc, hc, _ = canonicalize_to_mobius(np.asarray(theta).flatten(),
                                           np.asarray(h).flatten())
        if flat:
            ax = fig.add_subplot(1, 4, idx)
            is_true = (idx == 1)
            xl = r'$\theta$ (degrees)' if is_true else r'$z_\theta$ (degrees)'
            yl = r'$h$' if is_true else r'$z_h$'
            _plot_2d_panel(ax, np.degrees(tc), hc, colors, xl, yl, title,
                           xlim=(-180, 180), ylim=(-0.05, 1.05), anchor_indices=anch)
            ax.set_aspect('auto')
        else:
            ax = fig.add_subplot(1, 4, idx, projection='3d')
            x, y, z = embed_on_mobius(tc, hc)
            _plot_3d_panel(ax, x, y, z, colors, r'$x$', r'$y$', r'$z$', title,
                           aspect=[1, 1, 0.3], anchor_indices=anch)

    _mobius_panel(1, data['theta'], data['h'],
                  'True Latent Space')

    ax_g = fig.add_subplot(1, 4, 2)
    _plot_gaussian_vae_panel(ax_g, z_gaussian, colors)

    _mobius_panel(3, z_mobius_no_anchor['theta'], z_mobius_no_anchor['h'],
                  r'M\"obius VAE')
    _mobius_panel(4, z_mobius['theta'], z_mobius['h'],
                  r'M\"obius VAE (anchored)', anch=anchor_indices)

    if flat:
        plt.subplots_adjust(left=0.05, right=0.98, wspace=0.35)
    else:
        finalize_synthetic_row(fig)
    save_figure(save_path)


# =============================================================================
# RECONSTRUCTIONS
# =============================================================================


def plot_reconstruction_grid(data, models, model_names, n_samples=6, 
                             save_path='figures/fig_reconstructions.pdf'):
    """
    Plot a grid of original observations and their reconstructions from each VAE.
    
    Args:
        data: dict with 'x' key containing observations
        models: list of trained VAE models
        model_names: list of names for each model
        n_samples: number of samples to show
        save_path: path to save the figure
    """
    x_data = data['x']
    
    np.random.seed(42)
    n_total = len(x_data)
    sample_indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
    sample_indices = np.sort(sample_indices)
    
    x_samples = x_data[sample_indices]
    
    reconstructions = []
    rms_errors = []
    for model in models:
        x_recon = get_reconstruction(model, x_samples)
        reconstructions.append(x_recon)
        rms = np.sqrt(np.mean((x_samples - x_recon)**2))
        rms_errors.append(rms)
    
    n_rows = 1 + len(models)
    n_cols = n_samples + 1

    scale_fonts_to_page(1.5 * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows),
                             constrained_layout=True)
    
    # First row: original observations
    for j in range(n_samples):
        ax = axes[0, j]
        ax.plot(x_samples[j], 'k-', linewidth=0.8)
        ax.set_xlim(0, len(x_samples[j]) - 1)
        ax.set_ylim(x_samples.min() - 0.1, x_samples.max() + 0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel('Original')
    
    # RMS header
    ax = axes[0, n_samples]
    ax.text(0.5, 0.5, 'RMS Error', ha='center', va='center', fontweight='bold')
    ax.axis('off')
    
    # Model rows: reconstructions
    for i, (recon, name, rms) in enumerate(zip(reconstructions, model_names, rms_errors)):
        for j in range(n_samples):
            ax = axes[i + 1, j]
            ax.plot(x_samples[j], 'k--', linewidth=0.5, alpha=0.5, label='Original')
            ax.plot(recon[j], 'b-', linewidth=0.8, label='Recon')
            ax.set_xlim(0, len(recon[j]) - 1)
            ax.set_ylim(x_samples.min() - 0.1, x_samples.max() + 0.1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name)
        
        # RMS value
        ax = axes[i + 1, n_samples]
        ax.text(0.5, 0.5, f'{rms:.4f}', ha='center', va='center', fontweight='bold')
        ax.axis('off')
    
    save_figure(save_path, pad_inches=0.5)
