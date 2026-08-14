"""
Sweep beta across a synthetic experiment, tracing the reconstruction / KL curve.

    python -m experiments torus --betas 0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0

A single run reports one operating point per model. The sweep reports the whole
trade-off, so it can show that the topology-aware models buy their structure
without paying for it in reconstruction. That makes it a different shape of job:
thousands of cheap trainings whose metrics rows are the only output, rather than
a handful of runs each saving checkpoints, latents and figures. It reads the same
SPECS table as the single-run driver — the data, the anchors, the model class and
the true factors all come from there — but trains at the reduced setting below.

Everything lands in one self-contained run directory:

    results/kl_sweep/<run_name>/
        run.json      sweep configuration and metadata
        metrics.csv   one row per (experiment, model, beta, seed)
        figures/      drawn by synthetic/sweep_plots.py

Each finished point is appended to `metrics_partial.csv` immediately, so a crash
costs one point rather than the sweep. Reusing a `--run-name` skips the points
already recorded there. Every point is seeded from its own (seed, beta, model)
coordinates, so a resumed sweep produces exactly what an uninterrupted one would,
whenever the crash happened.

Author: Jilles van Hulst
"""

import csv
import os
from datetime import datetime

import numpy as np
import torch

from topovae import GaussianVAE, train_vae
from topovae.evaluate import compute_all_metrics, save_run_info
from experiments.synthetic.data import train_test_split
from experiments.synthetic.experiment import SPECS
from experiments.synthetic.sweep_plots import render_run

#: Betas visited when none are given, spanning the range over which the KL term
#: goes from barely constraining the encoder to dominating it.
DEFAULT_BETAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
DEFAULT_N_SEEDS = 50

# A sweep is thousands of trainings, so each point has to be cheap. These are
# smaller than the single-run settings: 800 samples and half the hidden width
# still resolve the reconstruction/KL trade-off, and lr=1e-3 is stable across the
# whole beta range and converged by 8000 epochs.
N_SAMPLES = 800
OBS_DIM = 50
HIDDEN_DIM = 64
N_EPOCHS = 8000
LR = 1e-3
ANCHOR_WEIGHT = 10.0
BETA_ANNEAL_FRACTION = 0.25

#: The models compared at every point, as suffixes on the manifold name. The
#: single-run driver also trains a repelling-anchor Gaussian, which the sweep
#: omits because it is a cylinder-only comparison.
_ROLE_SUFFIXES = ["", "+Anchor"]


def _point_seed(base_seed, seed_idx, betas, beta, model_names, model_name):
    """A unique, order-independent seed for one (replicate, beta, model) point."""
    b = betas.index(beta)
    m = model_names.index(model_name)
    return base_seed + seed_idx * 100000 + b * 100 + m


def _key(experiment, model, beta, seed):
    """Hashable identity of a training point, robust to CSV float round-trip."""
    return (str(experiment), str(model), f"{float(beta):.10g}", str(seed))


def _read_rows(path):
    """Every row of a CSV as a list of dicts, or an empty list if it is absent."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_row(partial_path, row):
    """Append one metrics row, writing the header if the file is new."""
    new = not os.path.exists(partial_path)
    with open(partial_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def _write_rows(path, rows):
    """Write rows to a CSV, sorted by the columns that identify a point."""
    rows = sorted(rows, key=lambda r: (r["experiment"], r["model"],
                                       float(r["beta"]), int(r["seed"])))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _experiment_setup(experiment, n_samples, obs_dim, hidden_dim, data_seed):
    """Data, anchors and model builders for one manifold, read from its spec."""
    spec = SPECS[experiment]
    data = spec["data"](n_samples=n_samples, obs_dim=obs_dim, seed=data_seed)
    train_data, test_data = train_test_split(data, test_ratio=0.2)
    anchor_indices, anchor_targets = spec["anchors"](train_data)

    def build(cls):
        return lambda: cls(input_dim=obs_dim, hidden_dim=hidden_dim)

    model_builders = {"Gaussian": build(GaussianVAE)}
    for suffix in _ROLE_SUFFIXES:
        model_builders[spec["display"] + suffix] = build(spec["model"])

    return train_data, test_data, anchor_indices, anchor_targets, model_builders


def _run_single_training(experiment, model_name, model, train_data, test_data,
                         beta, n_epochs, lr, beta_anneal_epochs, anchor_weight,
                         anchor_indices, anchor_targets):
    """Train one point and return its metrics."""
    train_kwargs = {"n_epochs": n_epochs, "lr": lr, "beta": beta,
                    "beta_anneal_epochs": beta_anneal_epochs,
                    "x_test": test_data["x"]}
    if model_name.endswith("+Anchor"):
        train_kwargs.update(anchor_indices=anchor_indices,
                            anchor_targets=anchor_targets,
                            anchor_weight=anchor_weight)

    train_vae(model, train_data["x"], **train_kwargs)

    true_factors = {f: test_data[f] for f in SPECS[experiment]["factors"]}
    return compute_all_metrics(model, train_data["x"], test_data["x"], beta=beta,
                               true_factors=true_factors, data_manifold=experiment)


def run(experiments, betas=None, n_seeds=DEFAULT_N_SEEDS, base_seed=1000,
        run_name=None, data_seed=42, render=True):
    """Train the beta grid and write the sweep run directory.

    Args:
        experiments: one experiment name, or a list of them.
        betas: beta values to visit. Defaults to `DEFAULT_BETAS`.
        n_seeds: replicates per (experiment, beta, model) point.
        base_seed: first replicate seed.
        run_name: name of the run directory. Reusing one resumes it.
        data_seed: seed for the synthetic dataset, held fixed across the sweep.
        render: draw the sweep figure once the grid finishes.
    """
    if isinstance(experiments, str):
        experiments = [experiments]
    betas = list(betas) if betas else list(DEFAULT_BETAS)

    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", "kl_sweep", run_name)
    os.makedirs(run_dir, exist_ok=True)
    beta_anneal_epochs = max(1, int(N_EPOCHS * BETA_ANNEAL_FRACTION))

    partial_path = os.path.join(run_dir, "metrics_partial.csv")
    done = {_key(r["experiment"], r["model"], r["beta"], r["seed"])
            for r in _read_rows(partial_path)}
    total = len(experiments) * n_seeds * len(betas) * 3   # 3 models each
    print(f"Run directory: {run_dir}")
    if done:
        print(f"Resuming: {len(done)}/{total} points already done; "
              f"skipping those and continuing.")
    else:
        print(f"Fresh sweep: {total} points to train. "
              f"Progress is written to {os.path.basename(partial_path)} after each.")

    n_done = len(done)
    for experiment in experiments:
        print("\n" + "=" * 80)
        print(f"KL SWEEP: {experiment.upper()}")
        print("=" * 80)

        train_data, test_data, anchor_indices, anchor_targets, model_builders = \
            _experiment_setup(experiment, N_SAMPLES, OBS_DIM, HIDDEN_DIM, data_seed)
        model_names = list(model_builders)

        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx
            print(f"\nSeed {seed} ({seed_idx + 1}/{n_seeds})")

            for beta in betas:
                print(f"  Beta={beta:g}")

                for model_name, build_model in model_builders.items():
                    if _key(experiment, model_name, beta, seed) in done:
                        continue

                    # Seed this point from its own coordinates so the result is
                    # independent of which points ran before it (resume-safe).
                    point_seed = _point_seed(base_seed, seed_idx, betas, beta,
                                             model_names, model_name)
                    np.random.seed(point_seed)
                    torch.manual_seed(point_seed)

                    n_done += 1
                    print(f"    [{n_done}/{total}] Training {model_name}")

                    metrics = _run_single_training(
                        experiment, model_name, build_model(), train_data, test_data,
                        beta=beta, n_epochs=N_EPOCHS, lr=LR,
                        beta_anneal_epochs=beta_anneal_epochs,
                        anchor_weight=ANCHOR_WEIGHT,
                        anchor_indices=anchor_indices, anchor_targets=anchor_targets)

                    row = {"experiment": experiment, "model": model_name,
                           "beta": beta, "seed": seed, **metrics}
                    _append_row(partial_path, row)   # durable, per point
                    done.add(_key(experiment, model_name, beta, seed))

    # Finalize from the durable partial file (covers this run + any resumed work).
    rows = _read_rows(partial_path)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    _write_rows(metrics_path, rows)
    print(f"\nSaved long-form metrics to: {metrics_path}")

    # Also write per-experiment CSVs for quick inspection
    swept = sorted({r["experiment"] for r in rows})
    for experiment in swept:
        exp_rows = [r for r in rows if r["experiment"] == experiment]
        _write_rows(os.path.join(run_dir, f"{experiment}_results.csv"), exp_rows)

    save_run_info(run_dir, {
        "experiment": "kl_sweep",
        "run_name": run_name,
        "metrics": "metrics.csv",
        "figures_dir": "figures",
        "experiments": experiments,
        "models_by_experiment": {
            exp: sorted({r["model"] for r in rows if r["experiment"] == exp})
            for exp in swept
        },
        "hyperparameters": {
            "betas": betas, "num_seeds": n_seeds, "base_seed": base_seed,
            "data_seed": data_seed, "n_samples": N_SAMPLES, "obs_dim": OBS_DIM,
            "hidden_dim": HIDDEN_DIM, "n_epochs": N_EPOCHS, "lr": LR,
            "anchor_weight": ANCHOR_WEIGHT,
            "beta_anneal_fraction": BETA_ANNEAL_FRACTION,
        },
    })
    print(f"Saved run metadata to: {os.path.join(run_dir, 'run.json')}")

    if render:
        render_run(run_dir)
