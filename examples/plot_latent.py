"""
A one-call look at a trained latent space, for playing around.

    from plot_latent import plot_latent
    plot_latent(model, data, save_path="latent.png")

This is the quick "did it work?" plot: a scatter of the two latent coordinates,
coloured by a true factor so you can see whether the manifold came out. It reads
whatever coordinates the model exposes, so the same call works for every model.

It is deliberately not the paper figure — those live in ``experiments/`` and draw
the full multi-panel comparison. This is one axis, no LaTeX, no styling to fuss
over. It lives beside ``quickstart.py`` rather than in the library, because a
throwaway plot is an example rather than part of the API, and because importing
it needs matplotlib while the rest of ``topovae`` does not.
"""

import matplotlib.pyplot as plt

from topovae import get_latent_codes


def plot_latent(model, data, color_by=None, save_path=None, ax=None):
    """Scatter a trained model's latent space, coloured by a true factor.

    Args:
        model: a trained VAE (any synthetic model).
        data: the dict returned by a ``generate_*_data`` function; needs ``'x'``.
        color_by: which true factor to colour by (e.g. ``'theta'``). Defaults to
            the first angular factor in ``data``.
        save_path: where to write the figure. If None, the axes are returned
            without saving so you can show or tweak them.
        ax: draw into an existing axes instead of a fresh figure.

    Returns:
        The matplotlib axes.
    """
    z = get_latent_codes(model, data["x"])

    # Pick the two coordinates to plot from whatever the model exposes.
    if "z" in z:                       # Gaussian baseline: R^d
        xs, ys, xlabel, ylabel = z["z"][:, 0], z["z"][:, 1], "$z_1$", "$z_2$"
    elif "theta1" in z:                # torus: (theta1, theta2)
        xs, ys, xlabel, ylabel = z["theta1"], z["theta2"], r"$z_{\theta_1}$", r"$z_{\theta_2}$"
    elif "z1" in z:                    # annulus: planar (z1, z2)
        xs, ys, xlabel, ylabel = z["z1"], z["z2"], "$z_1$", "$z_2$"
    else:                              # cylinder / Mobius: (theta, h)
        xs, ys, xlabel, ylabel = z["theta"], z["h"], r"$z_\theta$", "$z_h$"

    if color_by is None:
        color_by = "theta" if "theta" in data else "theta1" if "theta1" in data else None
    colors = data.get(color_by)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(xs, ys, c=colors, cmap="hsv", s=6, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{type(model).__name__} latent space")
    if colors is not None:
        plt.colorbar(scatter, ax=ax, label=color_by)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved latent-space plot to {save_path}")
    return ax
