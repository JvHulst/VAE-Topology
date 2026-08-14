"""
Which samples to pin, and to what, for each synthetic manifold.

Anchoring fixes the otherwise arbitrary choice of coordinates on the latent
manifold; the loss that applies it is ``topovae.train.angular_anchor_loss``
and its bounded counterpart. Choosing the anchors is experiment specific, which
is why it lives here rather than in the library.

The recipe is the same for all four: lay out a small grid of target coordinates
covering the manifold, and for each target pick the sample nearest to it in the
true coordinates, measuring angular coordinates with circular distance so that
the wrap is respected. A target whose nearest sample is already taken is
skipped, so the returned count can be below the grid size.

Each function returns ``(anchor_indices, anchor_targets)``, where the targets
are the true coordinates of the chosen samples rather than the grid values they
were selected for. Pinning a sample to its own true coordinate is what makes the
anchored latent space comparable against the ground truth.

Author: Jilles van Hulst
"""

import numpy as np


def find_cylinder_anchors(data: dict):
    """Find anchor points that span the cylinder latent space."""
    target_thetas = [-np.pi, -np.pi / 2, 0, np.pi / 2]
    target_hs = [0.0, 1.0]

    anchor_indices = []
    anchor_thetas = []
    anchor_hs = []

    for theta_target in target_thetas:
        for h_target in target_hs:
            theta_dist = np.abs(np.angle(np.exp(1j * (data["theta"] - theta_target))))
            h_dist = np.abs(data["h"] - h_target)
            combined_dist = theta_dist + h_dist
            idx = int(np.argmin(combined_dist))
            if idx not in anchor_indices:
                anchor_indices.append(idx)
                anchor_thetas.append(data["theta"][idx])
                anchor_hs.append(data["h"][idx])

    return anchor_indices, {"theta": anchor_thetas, "h": anchor_hs}


def find_torus_anchors(data: dict, n_anchors: int = 4):
    """Find anchor points for torus runs."""
    target_theta1s = np.linspace(-np.pi, np.pi, n_anchors, endpoint=False)
    target_theta2s = np.linspace(-np.pi, np.pi, n_anchors, endpoint=False)

    anchor_indices = []
    anchor_theta1s = []
    anchor_theta2s = []

    for theta1_target in target_theta1s:
        for theta2_target in target_theta2s:
            theta1_dist = np.abs(np.angle(np.exp(1j * (data["theta1"] - theta1_target))))
            theta2_dist = np.abs(np.angle(np.exp(1j * (data["theta2"] - theta2_target))))
            combined_dist = theta1_dist + theta2_dist
            idx = int(np.argmin(combined_dist))
            if idx not in anchor_indices:
                anchor_indices.append(idx)
                anchor_theta1s.append(data["theta1"][idx])
                anchor_theta2s.append(data["theta2"][idx])

    return anchor_indices, {"theta1": anchor_theta1s, "theta2": anchor_theta2s}


def find_mobius_anchors(data: dict, n_anchors: int = 8):
    """Find anchor points for Möbius runs."""
    n_heights = 2
    n_angles = n_anchors // n_heights
    target_angles_mobius = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    target_heights = np.linspace(0, 1, n_heights + 2)[1:-1]

    anchor_indices = []
    anchor_thetas = []
    anchor_heights = []

    for height_target in target_heights:
        for theta_target in target_angles_mobius:
            theta_dist = np.abs(np.angle(np.exp(1j * (data["theta_mobius"] - theta_target))))
            height_dist = np.abs(data["h"] - height_target)
            combined_dist = theta_dist + height_dist
            idx = int(np.argmin(combined_dist))
            if idx not in anchor_indices:
                anchor_indices.append(idx)
                anchor_thetas.append(data["theta"][idx])
                anchor_heights.append(data["h"][idx])

    return anchor_indices, {"theta": anchor_thetas, "h": anchor_heights}


def find_annulus_anchors(data: dict):
    """Find anchor points for annulus runs."""
    target_thetas = [0, np.pi * 2 / 5, np.pi * 4 / 5, -np.pi * 4 / 5, -np.pi * 2 / 5]
    target_rs = [data["r_min"] + 0.1, (data["r_min"] + data["r_max"]) / 2, data["r_max"] - 0.1]

    anchor_indices = []
    anchor_thetas = []
    anchor_rs = []

    for theta_target in target_thetas:
        for r_target in target_rs:
            theta_dist = np.abs(np.angle(np.exp(1j * (data["theta"] - theta_target))))
            r_dist = np.abs(data["r"] - r_target)
            combined_dist = theta_dist + r_dist
            idx = int(np.argmin(combined_dist))
            if idx not in anchor_indices:
                anchor_indices.append(idx)
                anchor_thetas.append(data["theta"][idx])
                anchor_rs.append(data["r"][idx])

    return anchor_indices, {"theta": anchor_thetas, "r": anchor_rs}
