# topovae: Constructing VAE Latent Spaces with Prescribed Topology

This repository contains **topovae**, a library for giving a variational autoencoder a latent space with a topology you choose. A standard VAE puts its latent space in $\mathbb{R}^d$. When the data lives on a circle, a torus, or a Möbius strip, $\mathbb{R}^d$ cannot hold that shape without tearing it, and the learned coordinates inherit the tear. This library builds the manifold into the latent space instead, using a reparameterisable distribution per factor, a decoder that reads an embedding of the manifold, and an optional anchor loss that pins the otherwise-arbitrary choice of coordinates.

**📄 Paper:** [Constructing VAE Latent Spaces with Prescribed Topology](https://arxiv.org/abs/2606.07058)
(J. S. van Hulst, J. M. Tomczak, W. P. M. H. Heemels, and D. J. Antunes)

## Quick Start

```bash
git clone https://github.com/JvHulst/VAE-Topology.git
cd VAE-Topology
pip install -e ".[all]"

# See it work on synthetic cylinder data
python examples/quickstart.py

# Reproduce a paper experiment
python -m experiments torus
```

## Using the library

```python
from topovae import CylinderVAE, train_vae, get_latent_codes

model = CylinderVAE(input_dim=50)
train_vae(model, x_train, n_epochs=3000, beta=1.0)
z = get_latent_codes(model, x_test)      # {'theta': [N], 'h': [N]}
```

## Experiments

```bash
python -m experiments torus                      # train, then draw its figures
python -m experiments torus --seeds 30           # repeat, report the spread
python -m experiments torus --betas 0.05,0.1,1.0 # sweep beta, draw the trade-off
python -m experiments torus --render results/torus/<run>
```

| Experiment | Data | Prescribed latent |
|---|---|---|
| `cylinder` | generated on $S^1 \times [0,1]$ | $S^1 \times [0,1]$ |
| `torus` | generated on $T^2$ | $T^2$ |
| `mobius` | generated on $(S^1 \times [0,1]) / \mathbb{Z}_2$ | $(S^1 \times [0,1]) / \mathbb{Z}_2$ |
| `annulus` | generated on $S^1 \times [r_{\min}, r_{\max}]$ | $S^1 \times [r_{\min}, r_{\max}]$ |
| `rotated` | MNIST digits under rotation | $\mathbb{R}^k \times S^1$ |
| `shifted` | MNIST digits under cyclic shift | $\mathbb{R}^k \times T^2$ |

The first four generate observations from a manifold we pick, so the true latent variable is known outright. The two image experiments apply a group action to real digits. There the periodic factor is known because we applied it, while digit identity and handwriting style are unlabelled and left to the $\mathbb{R}^k$ coordinates. The right-hand column is the topology the model is given, not a claim about what the data's own latent space is.

Every experiment trains a Gaussian baseline against the topology-aware VAE with and without anchoring, at a $\beta$ tuned so the unweighted KL matches across models. `--seeds` varies model initialisation only, holding the dataset and the split fixed, so the spread it reports is optimisation variance. `--betas` walks the reconstruction/regularisation trade-off at a reduced setting and keeps only the metrics. Both are resumable, so reusing a `--run-name` skips finished work.

Training writes a self-contained run directory holding `run.json`, `checkpoints/`, `latents.npz`, `metrics.csv` and `figures/`. Passing `--render` reads one back and redraws without retraining.

### Metrics

`metrics.csv` holds train and test reconstruction RMS, the unweighted KL divergence, and the geodesic stress. The last is Kruskal's stress-1 between distances on the data manifold and distances in the model's own latent space, each measured with its own geodesic and normalised to unit mean. It is computed only over pairs whose true distance is known. For the image experiments that means pairs sharing a base digit, since those differ by the applied group element and by nothing else.

## Models

Each experiment above needed one model, and each was assembled from the same three pieces, a distribution per factor, an embedding for the decoder to read, and a per-factor KL. Writing a new one is meant to be short.

| Model | Latent space | Distributions |
|---|---|---|
| `GaussianVAE` | $\mathbb{R}^d$ (baseline) | Gaussian |
| `CylinderVAE` | $S^1 \times [0,1]$ | Wrapped Normal, Kumaraswamy |
| `TorusVAE` | $T^2 = S^1 \times S^1$ | Wrapped Normal × 2 |
| `AnnulusVAE` | $S^1 \times [r_{\min}, r_{\max}]$ | Wrapped Normal, Kumaraswamy |
| `MobiusVAE` | $(S^1 \times [0,1]) / \mathbb{Z}_2$ | Wrapped Normal, Kumaraswamy |
| `MixedCircleVAE` | $\mathbb{R}^k \times S^1$ | Gaussian, Wrapped Normal |
| `MixedTorusVAE` | $\mathbb{R}^k \times T^2$ | Gaussian, Wrapped Normal × 2 |

The `Mixed*` models take a 28×28 image through a CNN backbone, and the rest take a plain observation vector. All seven share one encode/decode contract, documented at the top of `topovae/models.py`.

## Adding a topology

1. If a factor needs a distribution beyond Gaussian, Wrapped Normal, or Kumaraswamy, add a class to `topovae/distributions.py` with `sample`, `kl_divergence`, and `mean`.
2. Add a model to `topovae/models.py` following the encode/decode contract. `encode` returns the posterior parameters, `decode` reads an embedding of the manifold, and `loss` sums a per-factor KL, which decouples across the product.
3. Add a row to the `SPECS` table of the matching experiment driver.

## Repository Structure

```
├── topovae/                    # The library
│   ├── distributions.py        # Wrapped Normal, Kumaraswamy
│   ├── layers.py               # Network builders, invariant features, manifold distances
│   ├── models.py               # One VAE per prescribed latent manifold
│   ├── train.py                # Training loops, anchor and repelling losses
│   └── evaluate.py             # Metrics and run-directory IO
├── experiments/                # The paper's experiments
│   ├── __main__.py             # Command-line entry point
│   ├── synthetic/              # Cylinder, torus, Möbius, annulus
│   └── mnist/                  # Rotated and shifted MNIST
└── examples/                   # quickstart.py
```

## Dependencies

- `numpy`, `torch`: the library
- `matplotlib`, `pandas`, `scipy`, `torchvision`: the experiments
- `umap-learn`, `scikit-learn`: the Gaussian latent-space panels of the image figures

## Citation

**If you use this code, please cite our paper:**

```bibtex
@article{hulst2026,
  title = {Constructing VAE Latent Spaces with Prescribed Topology},
  archivePrefix = {arXiv},
  arxivId = {2606.07058},
  eprint = {2606.07058},
  author = {van Hulst, J. S. and Tomczak, J. M. and Heemels, W P M H and Antunes, D J},
  url = {https://arxiv.org/abs/2606.07058},
  year = {2026}
}
```

Jilles van Hulst, `j.s.v.hulst@tue.nl`, Control Systems Technology, TU Eindhoven.
