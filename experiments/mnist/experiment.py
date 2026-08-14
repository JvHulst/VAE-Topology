"""
The two image experiments: rotated and cyclically shifted MNIST.

    python -m experiments rotated
    python -m experiments shifted --render results/shifted_mnist/<run>

Same shape as the synthetic driver. Everything the two experiments share — three
models (a Gaussian baseline, the topology-aware VAE with and without anchoring),
training, saving, metrics — lives in `train` and `render`. Everything that
differs is a field of `SPECS`: the model, the data, how anchors are built (angles
on S^1 for rotated, a grid of pixel shifts on T^2 for shifted), and the figures.
Hyperparameters shared by both experiments are module constants below; the ones
that differ are fields of the spec.

Only the periodic coordinate is anchored; the Gaussian style space that carries
digit identity is never pinned, which is the point the experiment makes.

Author: Jilles van Hulst
"""

import os
from datetime import datetime

import numpy as np
import torch

from topovae import (
    MNISTGaussianVAE,
    MixedCircleVAE,
    MixedTorusVAE,
    get_mnist_latent_codes,
    train_mnist_vae,
)
from topovae.evaluate import (
    compute_mnist_metrics,
    flatten_metrics,
    get_figures_dir,
    load_checkpoint_bundle,
    load_run_info,
    save_checkpoint_bundle,
    save_latents,
    save_metrics,
    save_run_info,
)
from experiments import SEED
from experiments.mnist.data import (
    make_mnist_loaders,
    make_rotated_mnist_data,
    make_shifted_mnist_data,
)
from experiments.mnist.anchors import (
    find_canonical_circle_anchors,
    find_canonical_torus_anchors,
    inspect_anchors,
)
from experiments.mnist.plots import (
    plot_rotated_decoded_ring,
    plot_rotated_main_figure,
    plot_shifted_decoded_ring,
    plot_shifted_main_figure,
)

# Shared by both image experiments.
N_EXEMPLARS_PER_DIGIT = 500
GAUSSIAN_DIM = 2
BATCH_SIZE = 256
N_EPOCHS = 50
TRAIN_RATIO = 0.8

# Style-space panel of the main figure: True splits the custom VAE's panel into
# four angular-slice quadrants (digits separate at a fixed angle); False draws the
# aggregated scatter over all angles.
SLICED_STYLE = False


# --- per-experiment anchors and figures --------------------------------------

def _rotated_anchors(train_data, digits, spec):
    """One anchor per digit at each of n evenly-spaced angles on S^1."""
    target_angles = np.linspace(-np.pi, np.pi, spec["n_anchor_angles"], endpoint=False)
    return find_canonical_circle_anchors(train_data, target_angles=target_angles,
                                         anchor_digits=digits)


def _shifted_anchors(train_data, digits, spec):
    """One anchor per digit at each point of an n x n grid of integer pixel shifts."""
    n = spec["n_anchors_per_dim"]
    steps = [int(round(28 * k / n)) % 28 for k in range(n)]
    pixel_shifts = [(dx, dy) for dx in steps for dy in steps]
    return find_canonical_torus_anchors(train_data, pixel_shifts=pixel_shifts,
                                        anchor_digits=digits)


def _rotated_figures(data, models, Z, fig_dir):
    plot_rotated_main_figure(
        data, Z["gaussian"], Z.get("topology"), Z["topology_anchor"],
        model_gauss=models["gaussian"], model_circ_noanch=models.get("topology"),
        model_circ=models["topology_anchor"], sliced_style=SLICED_STYLE,
        save_path=os.path.join(fig_dir, "fig_rotated_mnist.pdf"))
    plot_rotated_decoded_ring(
        models["topology_anchor"], Z["topology_anchor"], data, n_angle=12,
        save_path=os.path.join(fig_dir, "fig_rotated_mnist_ring.pdf"))


def _shifted_figures(data, models, Z, fig_dir):
    entries = [("Gaussian VAE", models["gaussian"], Z["gaussian"])]
    if "topology" in models:
        entries.append(("Mixed Torus VAE", models["topology"], Z["topology"]))
    entries.append(("Mixed Torus VAE (anchored)", models["topology_anchor"], Z["topology_anchor"]))
    plot_shifted_main_figure(
        data, entries, sliced_style=SLICED_STYLE,
        save_path=os.path.join(fig_dir, "fig_shifted_mnist.pdf"))
    plot_shifted_decoded_ring(
        models["topology_anchor"], Z["topology_anchor"], data, n_angle=14,
        save_path=os.path.join(fig_dir, "fig_shifted_mnist_ring.pdf"))


SPECS = {
    "rotated": dict(
        results_dir="rotated_mnist",
        model=MixedCircleVAE, display="MixedCircleVAE",
        make_data=lambda digits, n_ex, per_ex: make_rotated_mnist_data(
            digits=digits, n_exemplars_per_digit=n_ex, rotations_per_exemplar=per_ex),
        per_exemplar_key="rotations_per_exemplar",
        extra_latent=1,                       # ell = gaussian_dim + 1  (one S^1)
        coord_names=("theta",),
        anchors=_rotated_anchors, figures=_rotated_figures,
        angle_keys=("theta",),
        digits=[4, 7], per_exemplar=180,
        lr=3e-3, hidden_dim=512,
        gaussian_beta=0.02, topology_beta=0.02, anchor_weight=50.0,
        n_anchor_angles=90,
    ),
    "shifted": dict(
        results_dir="shifted_mnist",
        model=MixedTorusVAE, display="MixedTorusVAE",
        make_data=lambda digits, n_ex, per_ex: make_shifted_mnist_data(
            digits=digits, n_exemplars_per_digit=n_ex, shifts_per_exemplar=per_ex),
        per_exemplar_key="shifts_per_exemplar",
        extra_latent=2,                       # ell = gaussian_dim + 2  (T^2)
        coord_names=("theta1", "theta2"),
        anchors=_shifted_anchors, figures=_shifted_figures,
        angle_keys=("theta1", "theta2"),
        digits=[3, 4, 7], per_exemplar=100,
        lr=1e-3, hidden_dim=256,
        gaussian_beta=0.04, topology_beta=0.02, anchor_weight=100.0,
        n_anchors_per_dim=4,
    ),
}

# The three models every image experiment trains, in order.
ROLES = ["gaussian", "topology", "topology_anchor"]


def _build(spec, role, gaussian_dim=None, hidden_dim=None):
    """A fresh model for a role. `gaussian_dim`/`hidden_dim` default to the spec,
    but a saved run may override them so its checkpoints rebuild exactly."""
    gaussian_dim = GAUSSIAN_DIM if gaussian_dim is None else gaussian_dim
    hidden_dim = spec["hidden_dim"] if hidden_dim is None else hidden_dim
    if role == "gaussian":
        return MNISTGaussianVAE(latent_dim=gaussian_dim + spec["extra_latent"],
                                hidden_dim=hidden_dim)
    return spec["model"](gaussian_dim=gaussian_dim, hidden_dim=hidden_dim)


def _role_names(spec):
    gauss_latent_dim = GAUSSIAN_DIM + spec["extra_latent"]
    return {
        "gaussian": f"Gaussian VAE (ell={gauss_latent_dim})",
        "topology": f"{spec['display']} (no anch., k={GAUSSIAN_DIM})",
        "topology_anchor": f"{spec['display']} (anchored, k={GAUSSIAN_DIM})",
    }


def _betas(spec):
    return {"gaussian": spec["gaussian_beta"], "topology": spec["topology_beta"],
            "topology_anchor": spec["topology_beta"]}


def _metrics(spec, models, names, train_loader, val_loader, data):
    betas = _betas(spec)
    true_angles = [data[k] for k in spec["angle_keys"]]
    metrics = [
        compute_mnist_metrics(models[r], train_loader, val_loader, betas[r],
                              x_data=data["x"], true_angles=true_angles,
                              groups=data["exemplar"])
        for r in models
    ]
    return [names[r] for r in models], metrics


def _hyperparameters(spec):
    """The run's hyperparameters, flat, as written to run.json."""
    density_key = "n_anchor_angles" if "n_anchor_angles" in spec else "n_anchors_per_dim"
    return {
        "digits": spec["digits"],
        "n_exemplars_per_digit": N_EXEMPLARS_PER_DIGIT,
        spec["per_exemplar_key"]: spec["per_exemplar"],
        "gaussian_dim": GAUSSIAN_DIM,
        "gauss_latent_dim": GAUSSIAN_DIM + spec["extra_latent"],
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "train_ratio": TRAIN_RATIO,
        "lr": spec["lr"],
        "hidden_dim": spec["hidden_dim"],
        "gaussian_beta": spec["gaussian_beta"],
        "topology_beta": spec["topology_beta"],
        "anchor_weight": spec["anchor_weight"],
        density_key: spec[density_key],
    }


def results_dir(name):
    """Where this experiment's runs are written."""
    return os.path.join("results", SPECS[name]["results_dir"])


def train(name, run_name=None, seed=SEED):
    """Train one experiment. `seed` initialises the models; the data and the
    train/test split are fixed by `SEED` regardless, so repeating a run under
    different `seed` values varies optimisation alone."""
    spec = SPECS[name]
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir(name), run_name)
    os.makedirs(run_dir, exist_ok=True)
    print("\n" + "=" * 70 + f"\nEXPERIMENT: {name} MNIST\n" + "=" * 70)

    digits = spec["digits"]
    data = spec["make_data"](digits, N_EXEMPLARS_PER_DIGIT, spec["per_exemplar"])
    train_loader, val_loader, train_data, _ = make_mnist_loaders(
        data, batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO, seed=SEED)

    anchor_indices, anchor_targets = spec["anchors"](train_data, digits, spec)
    anchor_x = torch.tensor(train_data["x"][anchor_indices], dtype=torch.float32)
    inspect_anchors(train_data, anchor_indices, anchor_targets,
                    save_path=os.path.join(run_dir, "anchor_inspection.png"),
                    coord_names=spec["coord_names"])

    names = _role_names(spec)
    betas = _betas(spec)
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    models = {}
    for i, role in enumerate(ROLES, 1):
        print(f"\n--- Training {names[role]}  [{i}/{len(ROLES)}] ---")
        model = _build(spec, role)
        kwargs = dict(n_epochs=N_EPOCHS, lr=spec["lr"], beta=betas[role],
                      vae_progress=f"VAE {i}/{len(ROLES)}",
                      checkpoint_path=os.path.join(ckpt_dir, f"{role}.pt"))
        if role == "topology_anchor":
            kwargs.update(anchor_x=anchor_x, anchor_targets=anchor_targets,
                          anchor_weight=spec["anchor_weight"])
        train_mnist_vae(model, train_loader, val_loader, **kwargs)
        models[role] = model

    Z = {r: get_mnist_latent_codes(m, data["x"]) for r, m in models.items()}
    save_checkpoint_bundle(run_dir, models)
    save_latents(run_dir, **{f"{r}_{k}": v for r, z in Z.items() for k, v in z.items()})

    metric_names, metrics = _metrics(spec, models, names, train_loader, val_loader, data)
    save_metrics(run_dir, flatten_metrics(metric_names, metrics))
    save_run_info(run_dir, {
        "experiment": spec["results_dir"],
        "run_name": run_name,
        "dataset": f"{name}_mnist_digits{''.join(str(d) for d in digits)}",
        "seed": SEED,
        "init_seed": seed,
        "models": list(models),
        "checkpoints": {r: os.path.join("checkpoints", f"{r}.pt") for r in models},
        "metrics": "metrics.csv",
        "latents": "latents.npz",
        "figures_dir": "figures",
        "hyperparameters": _hyperparameters(spec),
    })
    print(f"Saved run artifacts to: {run_dir}")
    render(name, run_dir)


def render(name, run_dir, recompute_metrics=False):
    spec = SPECS[name]
    info = load_run_info(run_dir)
    hp = info.get("hyperparameters", {})
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    gaussian_dim = int(hp.get("gaussian_dim", GAUSSIAN_DIM))
    hidden_dim = int(hp.get("hidden_dim", spec["hidden_dim"]))
    digits = hp.get("digits", spec["digits"])
    per_ex = int(hp.get(spec["per_exemplar_key"], spec["per_exemplar"]))
    n_ex = int(hp.get("n_exemplars_per_digit", N_EXEMPLARS_PER_DIGIT))
    data = spec["make_data"](digits, n_ex, per_ex)

    roles_in_run = [r for r in ROLES if r in info.get("models", ROLES)]
    models = {r: _build(spec, r, gaussian_dim, hidden_dim) for r in roles_in_run}
    load_checkpoint_bundle(run_dir, models)
    Z = {r: get_mnist_latent_codes(m, data["x"]) for r, m in models.items()}
    save_latents(run_dir, **{f"{r}_{k}": v for r, z in Z.items() for k, v in z.items()})

    fig_dir = get_figures_dir(run_dir)
    spec["figures"](data, models, Z, fig_dir)

    if recompute_metrics:
        train_loader, val_loader, _, _ = make_mnist_loaders(
            data, batch_size=int(hp.get("batch_size", BATCH_SIZE)),
            train_ratio=float(hp.get("train_ratio", TRAIN_RATIO)), seed=SEED)
        names = _role_names(spec)
        metric_names, metrics = _metrics(spec, models, names, train_loader, val_loader, data)
        save_metrics(run_dir, flatten_metrics(metric_names, metrics))
    print(f"Rendered figures to: {fig_dir}")
