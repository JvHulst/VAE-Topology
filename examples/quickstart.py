"""
Train a cylinder VAE end to end, in under a minute, with no data on disk.

Run it to confirm the install works and to see the shape of the library's API:

    python examples/quickstart.py

The data lives on a cylinder S¹ × [0, 1] — an angle θ and a height h — which is
then projected into a 50-dimensional observation vector, the only thing the
model sees.  A CylinderVAE gives its latent space that same S¹ × [0, 1] shape,
so after training its two latent coordinates recover θ and h up to a global
offset.

Compare against `GaussianVAE`, whose flat ℝ² latent has to tear the circle
somewhere, and the angular agreement drops.
"""

import numpy as np

from topovae import CylinderVAE, get_latent_codes, train_vae


def make_cylinder_data(n_samples=2000, obs_dim=50, seed=0):
    """Sample (θ, h) on the cylinder and project to an observation vector."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(-np.pi, np.pi, n_samples)
    h = rng.uniform(0.0, 1.0, n_samples)

    # A fixed random linear map ℝ³ → ℝ^obs_dim, shared by every sample.  The
    # embedding (cos θ, sin θ, h) is what keeps the circle a circle.
    coords = np.column_stack([np.cos(theta), np.sin(theta), h])
    basis = np.random.RandomState(1).randn(3, obs_dim)
    obs = coords @ basis + 0.1 * rng.randn(n_samples, obs_dim)
    return theta, h, obs.astype(np.float32)


def angular_agreement(a, b):
    """How tightly angle b tracks angle a, up to a constant offset and reflection.

    The resultant length of the residual: 1.0 when b matches a (in either
    orientation) plus any fixed offset, falling to 0.0 as the two become
    unrelated. Unlike a Pearson correlation of the angles, this stays valid when
    the angles cover the whole circle.
    """
    return float(max(abs(np.mean(np.exp(1j * (b - a)))),
                     abs(np.mean(np.exp(1j * (-b - a))))))


def main():
    theta, h, x = make_cylinder_data()

    print("Training a CylinderVAE (S^1 x [0, 1] latent) ...")
    model = CylinderVAE(input_dim=x.shape[1])
    train_vae(model, x, n_epochs=3000, beta=1.0, beta_anneal_epochs=500,
              print_every=500)

    z = get_latent_codes(model, x)
    r_theta = angular_agreement(theta, z["theta"])
    r_h = abs(np.corrcoef(h, z["h"])[0, 1])

    print(f"\nRecovered the circle:  angular agreement(theta, z_theta) = {r_theta:.3f}")
    print(f"Recovered the height:  |corr(h, z_h)| = {r_h:.3f}")
    print("\nBoth close to 1 means the latent space found the cylinder. "
          "The angle is recovered up to a global offset, which anchoring "
          "(see experiments/) removes.")

    # Draw the latent space if matplotlib is installed (the `experiments` extra).
    # The correlations above already tell the story, so this is optional.
    try:
        from plot_latent import plot_latent
    except ImportError:
        return
    data = {"x": x, "theta": theta, "h": h}
    plot_latent(model, data, color_by="theta", save_path="quickstart_latent.png")


if __name__ == "__main__":
    main()
