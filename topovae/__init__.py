"""
topovae — variational autoencoders with a prescribed latent topology.

A standard VAE puts its latent space in ℝ^d. When the data really lives on a
circle, a torus or a Möbius strip, ℝ^d cannot represent that without tearing it,
and the learned coordinates inherit the tear. This library gives the latent
space the manifold instead: a reparameterisable distribution per factor, a
decoder that reads an embedding of the manifold rather than raw coordinates, and
an optional anchor loss that pins the otherwise arbitrary choice of coordinates.

Quickstart:

    import numpy as np
    from topovae import CylinderVAE, train_vae, get_latent_codes

    model = CylinderVAE(input_dim=50)
    train_vae(model, x_train, n_epochs=5000, beta=1.0)
    z = get_latent_codes(model, x_test)      # {'theta': [N], 'h': [N]}

To give a latent space a topology this library does not implement, add a class
to ``topovae.models`` following the encode/decode contract documented there,
and a distribution to ``topovae.distributions`` if the factor needs one.

Author: Jilles van Hulst
"""

from .distributions import KumaraswamyDistribution, WrappedNormalDistribution
from .layers import (
    canonicalize_to_mobius,
    circular_distance,
    mobius_invariant_features,
    mobius_z2_distance,
)
from .models import (
    AnnulusVAE,
    CylinderVAE,
    GaussianVAE,
    MixedCircleVAE,
    MixedTorusVAE,
    MNISTGaussianVAE,
    MobiusVAE,
    TorusVAE,
)
from .train import (
    angular_anchor_loss,
    bounded_anchor_loss,
    get_latent_codes,
    get_mnist_latent_codes,
    get_reconstruction,
    load_model_from_checkpoint,
    train_mnist_vae,
    train_vae,
)

__all__ = [
    # Models — one prescribed latent manifold each
    'GaussianVAE',
    'CylinderVAE',
    'TorusVAE',
    'AnnulusVAE',
    'MobiusVAE',
    'MNISTGaussianVAE',
    'MixedCircleVAE',
    'MixedTorusVAE',
    # Distributions — the reparameterisable factors the models are built from
    'WrappedNormalDistribution',
    'KumaraswamyDistribution',
    # Training
    'train_vae',
    'train_mnist_vae',
    'angular_anchor_loss',
    'bounded_anchor_loss',
    'load_model_from_checkpoint',
    # Inference
    'get_latent_codes',
    'get_mnist_latent_codes',
    'get_reconstruction',
    # Manifold geometry
    'canonicalize_to_mobius',
    'mobius_invariant_features',
    'circular_distance',
    'mobius_z2_distance',
]
