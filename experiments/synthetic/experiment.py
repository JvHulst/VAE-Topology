"""
The four synthetic manifold experiments, where the true latent space is known.

    python -m experiments cylinder
    python -m experiments torus --render results/torus/<run>

One experiment is one row of `SPECS` below. Everything the four have in common —
the training schedule, the four-model line-up (Gaussian baseline, an optional
repelling-anchor Gaussian, the topology-aware VAE with and without anchoring),
saving the run, and the metrics table — lives in the two functions `train` and
`render`. Everything that differs — which model, which data, which figure — is a
field in the spec. To read what makes the torus experiment the torus experiment,
read its spec; nothing else changes.

The annulus has no counterpart in the paper. It is here because the framework
reaches past what the paper reports, and it costs one row to show that.

Author: Jilles van Hulst
"""

import os
from datetime import datetime

import numpy as np
import torch

from topovae import (
    AnnulusVAE,
    CylinderVAE,
    GaussianVAE,
    MobiusVAE,
    TorusVAE,
    get_latent_codes,
    train_vae,
)
from topovae.evaluate import (
    compute_all_metrics,
    flatten_metrics,
    get_figures_dir,
    load_checkpoint_bundle,
    load_run_info,
    print_metrics_table,
    save_checkpoint_bundle,
    save_latents,
    save_metrics,
    save_run_info,
)
from experiments import SEED
from experiments.synthetic.data import (
    generate_annulus_data,
    generate_cylinder_data,
    generate_mobius_data,
    generate_torus_data,
    train_test_split,
)
from experiments.synthetic.anchors import (
    find_annulus_anchors,
    find_cylinder_anchors,
    find_mobius_anchors,
    find_torus_anchors,
)
from experiments.synthetic.plots import (
    plot_annulus_experiment,
    plot_cylinder_experiment,
    plot_mobius_experiment,
    plot_reconstruction_grid,
    plot_torus_experiment,
)

# Shared training schedule — identical across all four experiments.
N_SAMPLES = 2000
OBS_DIM = 50
HIDDEN_DIM = 128
N_EPOCHS = 8000
LR = 1e-3
ANCHOR_WEIGHT = 10.0
TEST_RATIO = 0.2
REPEL_CENTER = [0.0, 0.0]


# --- per-experiment figures --------------------------------------------------
# Each draws the scatter comparison for its manifold from the latent codes `Z`
# (keyed by model role).  The reconstruction grid is drawn generically in
# `render`, so it is not repeated here.

def _plot_cylinder(data, Z, anchor_indices, fig_dir):
    for flat, suffix in [(False, ""), (True, "_2d")]:
        plot_cylinder_experiment(
            data, Z["gaussian"], Z["topology"], Z["topology_anchor"],
            z_gaussian_repel=Z["gaussian_aux"], anchor_indices=anchor_indices,
            save_path=os.path.join(fig_dir, f"fig_cylinder{suffix}.pdf"), flat=flat)


def _plot_torus(data, Z, anchor_indices, fig_dir):
    for flat, suffix in [(False, ""), (True, "_2d")]:
        plot_torus_experiment(
            data, Z["gaussian"], Z["topology"], Z["topology_anchor"],
            anchor_indices=anchor_indices,
            save_path=os.path.join(fig_dir, f"fig_torus{suffix}.pdf"), flat=flat)


def _plot_mobius(data, Z, anchor_indices, fig_dir):
    for flat, suffix in [(False, ""), (True, "_2d")]:
        plot_mobius_experiment(
            data, Z["gaussian"], Z["topology"], Z["topology_anchor"],
            anchor_indices=anchor_indices,
            save_path=os.path.join(fig_dir, f"fig_mobius{suffix}.pdf"), flat=flat)


def _plot_annulus(data, Z, anchor_indices, fig_dir):
    plot_annulus_experiment(
        data, Z["gaussian"], Z["gaussian_aux"], Z["topology"], Z["topology_anchor"],
        anchor_indices=anchor_indices,
        save_path=os.path.join(fig_dir, "fig_annulus.pdf"))


SPECS = {
    "cylinder": dict(
        model=CylinderVAE, data=generate_cylinder_data, anchors=find_cylinder_anchors,
        display="Cylinder", factors=["theta", "h"],
        gaussian_beta=1.2, topology_beta=1.0, repel_xi=0.3, plot=_plot_cylinder),
    "torus": dict(
        model=TorusVAE, data=generate_torus_data, anchors=find_torus_anchors,
        display="Torus", factors=["theta1", "theta2"],
        gaussian_beta=1.0, topology_beta=1.0, repel_xi=None, plot=_plot_torus),
    "mobius": dict(
        model=MobiusVAE, data=generate_mobius_data, anchors=find_mobius_anchors,
        display="Mobius", factors=["theta", "h"],
        gaussian_beta=1.3, topology_beta=0.95, repel_xi=None, plot=_plot_mobius),
    "annulus": dict(
        model=AnnulusVAE, data=generate_annulus_data, anchors=find_annulus_anchors,
        display="Annulus", factors=["theta", "r"],
        gaussian_beta=0.35, topology_beta=0.2, repel_xi=0.3, plot=_plot_annulus),
}


def results_dir(name):
    """Where this experiment's runs are written."""
    return os.path.join("results", name)


def _model_roles(spec):
    """The models this experiment trains, in order, as {role: display name}."""
    roles = {"gaussian": "Gaussian VAE"}
    if spec["repel_xi"] is not None:
        roles["gaussian_aux"] = "Gaussian VAE + repel"
    roles["topology"] = f"{spec['display']} VAE"
    roles["topology_anchor"] = f"{spec['display']} VAE (anchored)"
    return roles


def _build(spec, role):
    """A fresh model for a role — Gaussian baseline or the topology-aware VAE."""
    cls = GaussianVAE if role.startswith("gaussian") else spec["model"]
    return cls(input_dim=OBS_DIM, hidden_dim=HIDDEN_DIM)


def _beta(spec, role):
    return spec["gaussian_beta"] if role.startswith("gaussian") else spec["topology_beta"]


def _metrics(spec, models, roles, train_data, test_data):
    true_factors = {f: test_data[f] for f in spec["factors"]}
    metrics = [
        compute_all_metrics(models[r], train_data["x"], test_data["x"],
                            beta=_beta(spec, r), true_factors=true_factors,
                            data_manifold=spec["manifold"])
        for r in roles
    ]
    print_metrics_table(list(roles.values()), metrics)
    return metrics


def train(name, run_name=None, seed=SEED):
    """Train one experiment. `seed` initialises the models; the data and the
    train/test split are fixed by `SEED` regardless, so repeating a run under
    different `seed` values varies optimisation alone."""
    spec = dict(SPECS[name], manifold=name)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir(name), run_name)
    os.makedirs(run_dir, exist_ok=True)
    print("\n" + "=" * 70 + f"\nEXPERIMENT: {spec['display']}\n" + "=" * 70)

    data = spec["data"](n_samples=N_SAMPLES, obs_dim=OBS_DIM, seed=SEED)
    train_data, test_data = train_test_split(data, test_ratio=TEST_RATIO)
    anchor_indices, anchor_targets = spec["anchors"](train_data)

    roles = _model_roles(spec)
    models = {}
    for role, display in roles.items():
        print(f"\n--- Training {display} ---")
        kwargs = dict(n_epochs=N_EPOCHS, lr=LR, beta=_beta(spec, role),
                      beta_anneal_epochs=N_EPOCHS // 4, x_test=test_data["x"])
        if role == "gaussian_aux":
            kwargs.update(repel_center=REPEL_CENTER, repel_xi=spec["repel_xi"])
        if role == "topology_anchor":
            kwargs.update(anchor_indices=anchor_indices, anchor_targets=anchor_targets,
                          anchor_weight=ANCHOR_WEIGHT)
        model = _build(spec, role)
        train_vae(model, train_data["x"], **kwargs)
        models[role] = model

    Z = {role: get_latent_codes(m, data["x"]) for role, m in models.items()}
    metrics = _metrics(spec, models, roles, train_data, test_data)

    save_checkpoint_bundle(run_dir, models)
    save_latents(run_dir, **{f"{role}_{k}": v
                             for role, z in Z.items() for k, v in z.items()})
    save_metrics(run_dir, flatten_metrics(list(roles.values()), metrics))
    save_run_info(run_dir, {
        "experiment": name,
        "run_name": run_name,
        "dataset": f"synthetic_{name}",
        "models": list(roles),
        "checkpoints": {r: os.path.join("checkpoints", f"{r}.pt") for r in roles},
        "metrics": "metrics.csv",
        "latents": "latents.npz",
        "figures_dir": "figures",
        "hyperparameters": {
            "n_samples": N_SAMPLES, "obs_dim": OBS_DIM, "hidden_dim": HIDDEN_DIM,
            "n_epochs": N_EPOCHS, "lr": LR, "anchor_weight": ANCHOR_WEIGHT,
            "gaussian_beta": spec["gaussian_beta"], "topology_beta": spec["topology_beta"],
            "repel_xi": spec["repel_xi"], "seed": SEED, "init_seed": seed,
        },
    })
    print(f"Saved run artifacts to: {run_dir}")
    render(name, run_dir)


def render(name, run_dir, recompute_metrics=False):
    spec = dict(SPECS[name], manifold=name)
    hp = load_run_info(run_dir)["hyperparameters"]

    data = spec["data"](n_samples=hp["n_samples"], obs_dim=hp["obs_dim"], seed=hp["seed"])
    train_data, test_data, train_idx, _ = train_test_split(data, test_ratio=TEST_RATIO,
                                                           return_indices=True)
    anchor_local, _ = spec["anchors"](train_data)
    # The anchors index the (shuffled) train split; the latent scatter is over the
    # full dataset, so map the anchor positions back through the split.
    anchor_indices = train_idx[anchor_local]

    roles = _model_roles(spec)
    models = {role: _build(spec, role) for role in roles}
    load_checkpoint_bundle(run_dir, models)

    Z = {role: get_latent_codes(m, data["x"]) for role, m in models.items()}
    save_latents(run_dir, **{f"{role}_{k}": v
                             for role, z in Z.items() for k, v in z.items()})

    fig_dir = get_figures_dir(run_dir)
    spec["plot"](data, Z, anchor_indices, fig_dir)
    plot_reconstruction_grid(
        data, list(models.values()), list(roles.values()),
        save_path=os.path.join(fig_dir, f"fig_{name}_reconstructions.pdf"))

    if recompute_metrics:
        metrics = _metrics(spec, models, roles, train_data, test_data)
        save_metrics(run_dir, flatten_metrics(list(roles.values()), metrics))
    print(f"Rendered figures to: {fig_dir}")
