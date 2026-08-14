"""
Training a model, and getting things back out of it.

Two training loops, because the two families of experiment feed the model
differently: ``train_vae`` takes the whole synthetic dataset as one tensor,
``train_mnist_vae`` takes DataLoaders over tens of thousands of images.  Both
anneal beta from zero, both optionally apply an anchor loss, and both keep the
best validation checkpoint.

The anchor loss fixes the otherwise-arbitrary choice of coordinates on the
latent manifold.  Without it a topology-aware VAE learns the right *shape* but
places it at an arbitrary offset, so latent coordinates are not comparable
across runs.  Pinning a few samples to known coordinates removes that freedom.

``get_reconstruction`` and ``get_latent_codes`` read a trained model back out,
absorbing the fact that each model returns a different tuple from ``forward``.

Which samples to pin, and to what, is experiment specific and lives with the
experiment (``experiments/*/anchors.py``).

Author: Jilles van Hulst
"""

import os

import numpy as np
import torch
import torch.optim as optim

from .distributions import KumaraswamyDistribution
from .layers import circular_distance, mobius_z2_distance


# =============================================================================
# READING A TRAINED MODEL
# =============================================================================


# =============================================================================
# SYNTHETIC EXPERIMENTS
# =============================================================================


def get_reconstruction(model, x_data):
    """
    Reconstruct x by decoding the posterior mean.

    Decoding the mean rather than a sample from the posterior makes this
    deterministic, so a figure drawn from it reproduces exactly.  It is also
    the reconstruction the model is actually claiming: a draw from q(z|x)
    adds posterior noise that says nothing about reconstruction quality.

    Args:
        model: Trained VAE model (any type)
        x_data: Input data (numpy array or torch tensor)

    Returns:
        Reconstruction, as a numpy array if x_data was one
    """
    if isinstance(x_data, np.ndarray):
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
    else:
        x_tensor = x_data

    model.eval()
    with torch.no_grad():
        x_recon = _decode_posterior_mean(model, x_tensor)

    return x_recon.numpy() if isinstance(x_data, np.ndarray) else x_recon


def _decode_posterior_mean(model, x_tensor):
    """Encode x, then decode the mean of q(z|x) without sampling."""
    from .models import (AnnulusVAE, CylinderVAE, GaussianVAE, MobiusVAE,
                         MixedCircleVAE, MixedTorusVAE, MNISTGaussianVAE,
                         TorusVAE)

    if isinstance(model, (GaussianVAE, MNISTGaussianVAE)):
        mu, _ = model.encode(x_tensor)
        return model.decode(mu)

    if isinstance(model, (CylinderVAE, MobiusVAE)):
        mu_angle, _, a, b = model.encode(x_tensor)
        return model.decode(mu_angle, KumaraswamyDistribution.mean(a, b))

    if isinstance(model, AnnulusVAE):
        mu_angle, _, a_r, b_r = model.encode(x_tensor)
        r_unit = KumaraswamyDistribution.mean(a_r, b_r)
        return model.decode(mu_angle, model.r_min
                            + (model.r_max - model.r_min) * r_unit)

    if isinstance(model, TorusVAE):
        mu1, _, mu2, _ = model.encode(x_tensor)
        return model.decode(mu1, mu2)

    if isinstance(model, MixedCircleVAE):
        z_mu, _, mu_theta, _ = model.encode(x_tensor)
        return model.decode(z_mu, mu_theta)

    if isinstance(model, MixedTorusVAE):
        z_mu, _, mu1, _, mu2, _ = model.encode(x_tensor)
        return model.decode(z_mu, mu1, mu2)

    raise ValueError(f"Unknown model type: {type(model)}")


def get_latent_codes(model, x_data):
    """
    Extract latent codes using the mean (mu) for evaluation.
    
    Args:
        model: Trained VAE model
        x_data: Data to encode (numpy array)
    
    Returns:
        Dict with latent coordinates depending on model type
    """
    from .models import (GaussianVAE, CylinderVAE, TorusVAE, AnnulusVAE,
                         MobiusVAE)
    
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    model.eval()
    
    with torch.no_grad():
        if isinstance(model, GaussianVAE):
            mu, _ = model.encode(x_tensor)
            return {'z': mu.numpy()}

        elif isinstance(model, (CylinderVAE, MobiusVAE)):
            mu_angle, sigma, a, b = model.encode(x_tensor)
            h = KumaraswamyDistribution.mean(a, b)
            return {'theta': mu_angle.squeeze().numpy(), 'h': h.squeeze().numpy(),
                    'mu_angle': mu_angle.numpy(), 'sigma': sigma.numpy()}

        elif isinstance(model, TorusVAE):
            mu1, sigma1, mu2, sigma2 = model.encode(x_tensor)
            return {'theta1': mu1.squeeze().numpy(), 'theta2': mu2.squeeze().numpy()}

        elif isinstance(model, AnnulusVAE):
            mu_angle, sigma, a_r, b_r = model.encode(x_tensor)
            # Use Kumaraswamy mean for radius
            r_unit = KumaraswamyDistribution.mean(a_r, b_r)
            r = model.r_min + (model.r_max - model.r_min) * r_unit
            x_cart = r * torch.cos(mu_angle)
            y_cart = r * torch.sin(mu_angle)
            return {
                'z1': x_cart.squeeze().numpy(),
                'z2': y_cart.squeeze().numpy(),
                'theta': mu_angle.squeeze().numpy(),
                'r': r.squeeze().numpy()
            }

        else:
            raise ValueError(f"Unknown model type: {type(model)}")


# =============================================================================
# IMAGE EXPERIMENTS
# =============================================================================


def get_mnist_latent_codes(model, x_data, batch_size=256):
    """
    Extract latent distribution means from a trained MNIST VAE.

    Args:
        model:      Trained MNIST model (any of the MNIST* / Mixed* classes).
        x_data:     float32 numpy array [N, 784].
        batch_size: Mini-batch size for inference.

    Returns:
        dict whose keys depend on the model type:
          MNISTGaussianVAE:  {'z': [N, latent_dim]}
          MixedTorusVAE:     {'z_d': [N, k], 'theta1': [N], 'theta2': [N]}
          MixedCircleVAE:    {'z_d': [N, k], 'theta': [N]}
    """
    from .models import MNISTGaussianVAE, MixedCircleVAE, MixedTorusVAE

    model.eval()
    N = len(x_data)
    chunks = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            x_batch = torch.tensor(x_data[start:start + batch_size], dtype=torch.float32)
            enc = model.encode(x_batch)
            chunks.append([t.cpu().numpy() for t in enc])

    def _cat(i):
        return np.concatenate([c[i] for c in chunks], axis=0).squeeze(-1)

    if isinstance(model, MNISTGaussianVAE):
        return {'z': np.concatenate([c[0] for c in chunks], axis=0)}

    if isinstance(model, MixedTorusVAE):
        # encode returns (z_mu, z_logvar, mu1, sigma1, mu2, sigma2)
        z_d = np.concatenate([c[0] for c in chunks], axis=0)
        return {'z_d': z_d, 'theta1': _cat(2), 'theta2': _cat(4)}

    if isinstance(model, MixedCircleVAE):
        # encode returns (z_mu, z_logvar, mu_theta, sigma_theta)
        z_d = np.concatenate([c[0] for c in chunks], axis=0)
        return {'z_d': z_d, 'theta': _cat(2)}

    raise ValueError(f"Unsupported model type: {type(model)}")


# =============================================================================
# TRAINING
# =============================================================================


# =============================================================================
# ANCHOR LOSSES
# =============================================================================


def angular_anchor_loss(pred_angles, target_angles):
    """
    Compute anchor loss for angular coordinates with circular distance.
    
    Args:
        pred_angles: Predicted angles from encoder [N, 1]
        target_angles: Target anchor angles (list or tensor)
    
    Returns:
        Mean squared circular distance
    """
    if not isinstance(target_angles, torch.Tensor):
        target_angles = torch.tensor(target_angles, dtype=torch.float32)
    if target_angles.dim() == 1:
        target_angles = target_angles.reshape(-1, 1)
    
    angle_dist = circular_distance(pred_angles, target_angles)
    return torch.mean(angle_dist ** 2)


def bounded_anchor_loss(pred_values, target_values):
    """
    Compute anchor loss for bounded coordinates (simple MSE).
    
    Args:
        pred_values: Predicted values from encoder [N, 1]
        target_values: Target anchor values (list or tensor)
    
    Returns:
        Mean squared error
    """
    if not isinstance(target_values, torch.Tensor):
        target_values = torch.tensor(target_values, dtype=torch.float32)
    if target_values.dim() == 1:
        target_values = target_values.reshape(-1, 1)
    
    return torch.mean((pred_values - target_values) ** 2)


# =============================================================================
# SYNTHETIC EXPERIMENTS
# =============================================================================


def train_vae(model, x_data, n_epochs=50000, lr=3e-5, beta=1.0, print_every=300,
              anchor_indices=None, anchor_targets=None, anchor_weight=10.0,
              beta_anneal_epochs=None, repel_center=None, repel_xi=None,
              return_history=False, x_test=None, validation_freq=25):
    """
    Generic training loop with optional anchoring, repelling, KL annealing, and early stopping.
    
    Args:
        model: VAE model to train
        x_data: Training data (numpy array)
        n_epochs: Number of training epochs
        lr: Learning rate
        beta: Final KL weight (target β value after annealing)
        print_every: Print frequency
        anchor_indices: List of indices into x_data for anchor points
        anchor_targets: Dict mapping coordinate names to target values
        anchor_weight: Weight for anchor loss
        beta_anneal_epochs: Epochs to linearly anneal β from 0 to final value
        repel_center: Center point to repel latent codes from (for soft holes)
        repel_xi: Target radius of the soft hole. If provided, repelling weight is
                  automatically computed as: repel_weight = beta * xi^2 / 2
        return_history: If True, return (model, history_dict) instead of just model
        x_test: Test/validation data (numpy array). If provided, tracks best model on test loss
        validation_freq: Frequency (in epochs) to evaluate on test set
    
    Returns:
        If return_history=False: Trained model (at best test checkpoint if x_test provided)
        If return_history=True: (model, history) where history contains loss curves
    """

    if beta_anneal_epochs is None:
        beta_anneal_epochs = n_epochs // 4  # Default: anneal over first quarter of training

    # Import model types here to avoid circular imports
    from .models import (GaussianVAE, CylinderVAE, TorusVAE, AnnulusVAE,
                         MobiusVAE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    
    has_anchors = anchor_indices is not None and anchor_targets is not None
    if has_anchors:
        anchor_x = x_tensor[anchor_indices]
    
    # Compute repelling weight from xi and beta for meaningful hole control
    has_repel = repel_center is not None and repel_xi is not None
    if has_repel:
        repel_center_tensor = torch.tensor(repel_center, dtype=torch.float32)
        # Auto-compute repel_weight to create a hole of radius ~xi
        # From equilibrium analysis: r ≈ xi when repel_weight = beta * xi^2 / 2
        repel_weight = beta * (repel_xi ** 2) / 2.0
        print(f"Repelling anchor: xi={repel_xi:.3f}, auto-computed weight={repel_weight:.4f}")
    
    # History tracking
    history = {'loss': [], 'recon': [], 'kl': [], 'beta': [], 'anchor': [], 'repel': []}
    
    # Best model tracking (if test data provided)
    best_test_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    if x_test is not None:
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # KL annealing
        if epoch < beta_anneal_epochs:
            current_beta = beta * (epoch / beta_anneal_epochs)
        else:
            current_beta = beta
        
        anchor_loss = torch.tensor(0.0)
        repel_loss = torch.tensor(0.0)
        
        if isinstance(model, GaussianVAE):
            x_recon, mu, logvar, z = model(x_tensor)
            loss, recon, kl = model.loss(x_tensor, x_recon, mu, logvar, current_beta)
            
            if has_repel:
                repel_loss = model.repelling_loss(z, repel_center_tensor, xi=repel_xi)
                loss = loss + repel_weight * repel_loss
                
        elif isinstance(model, CylinderVAE):
            x_recon, mu_angle, sigma, a, b, theta, h = model(x_tensor)
            loss, recon, kl = model.loss(x_tensor, x_recon, mu_angle, sigma, a, b, current_beta)
            
            if has_anchors and 'theta' in anchor_targets:
                anchor_mu_angle, _, anchor_a, anchor_b = model.encode(anchor_x)
                anchor_h_mean = KumaraswamyDistribution.mean(anchor_a, anchor_b)
                anchor_loss = (angular_anchor_loss(anchor_mu_angle, anchor_targets['theta']) +
                               bounded_anchor_loss(anchor_h_mean, anchor_targets['h']))
                loss = loss + anchor_weight * anchor_loss
                
        elif isinstance(model, MobiusVAE):
            x_recon, mu_angle, sigma, a, b, theta, h = model(x_tensor)
            loss, recon, kl = model.loss(x_tensor, x_recon, mu_angle, sigma, a, b, theta, h, current_beta)
            
            if has_anchors and 'theta' in anchor_targets:
                anchor_mu_angle, _, anchor_a, anchor_b = model.encode(anchor_x)
                anchor_h_mean = KumaraswamyDistribution.mean(anchor_a, anchor_b)
                
                target_theta = torch.tensor(anchor_targets['theta'], dtype=torch.float32).unsqueeze(1)
                target_h = torch.tensor(anchor_targets['h'], dtype=torch.float32).unsqueeze(1)
                
                # Use ℤ₂-invariant distance for Möbius (squared for MSE-like loss)
                z2_dist = mobius_z2_distance(anchor_mu_angle, anchor_h_mean, target_theta, target_h)
                anchor_loss = torch.mean(z2_dist ** 2)
                loss = loss + anchor_weight * anchor_loss
                
        elif isinstance(model, TorusVAE):
            x_recon, mu1, sigma1, mu2, sigma2, theta1, theta2 = model(x_tensor)
            loss, recon, kl = model.loss(x_tensor, x_recon, mu1, sigma1, mu2, sigma2, current_beta)
            
            if has_anchors and 'theta1' in anchor_targets:
                anchor_mu1, _, anchor_mu2, _ = model.encode(anchor_x)
                anchor_loss = (angular_anchor_loss(anchor_mu1, anchor_targets['theta1']) +
                               angular_anchor_loss(anchor_mu2, anchor_targets['theta2']))
                loss = loss + anchor_weight * anchor_loss

        elif isinstance(model, AnnulusVAE):
            x_recon, mu_angle, sigma, a_r, b_r, z_cart, theta, r = model(x_tensor)
            loss, recon, kl = model.loss(x_tensor, x_recon, mu_angle, sigma, a_r, b_r, current_beta)

            if has_anchors and 'theta' in anchor_targets:
                anchor_mu_angle, _, anchor_a_r, anchor_b_r = model.encode(anchor_x)
                # Use Kumaraswamy mean for radius anchor
                anchor_r_mean = KumaraswamyDistribution.mean(anchor_a_r, anchor_b_r)

                # Convert target_r to unit interval [0, 1]
                target_r = torch.tensor(anchor_targets['r'], dtype=torch.float32).reshape(-1, 1)
                target_r_unit = (target_r - model.r_min) / (model.r_max - model.r_min)

                anchor_loss = (angular_anchor_loss(anchor_mu_angle, anchor_targets['theta']) +
                               bounded_anchor_loss(anchor_r_mean, target_r_unit))
                loss = loss + anchor_weight * anchor_loss

        else:
            raise ValueError(f"Unsupported model type in train_vae: {type(model)}")
        
        # Track history
        history['loss'].append(loss.item())
        history['recon'].append(recon.item())
        history['kl'].append(kl.item())
        history['beta'].append(current_beta)
        history['anchor'].append(anchor_loss.item() if has_anchors else 0.0)
        history['repel'].append(repel_loss.item() if has_repel else 0.0)
        
        loss.backward()
        optimizer.step()
        
        # Periodic validation on test set
        if x_test is not None and (epoch + 1) % validation_freq == 0 \
        and epoch >= beta_anneal_epochs:
            with torch.no_grad():
                # Compute test reconstruction loss
                x_recon_test = get_reconstruction(model, x_test)
                x_recon_test_tensor = torch.tensor(x_recon_test, dtype=torch.float32)
                test_recon_loss = torch.nn.functional.mse_loss(x_test_tensor, x_recon_test_tensor).item()
                
                # Track best model
                if test_recon_loss < best_test_loss:
                    best_test_loss = test_recon_loss
                    best_epoch = epoch + 1
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % print_every == 0:
            extra_str = ""
            if has_anchors and anchor_loss.item() > 0:
                extra_str += f", anchor={anchor_weight * anchor_loss.item():.4f}"
            if has_repel:
                extra_str += f", repel={repel_weight * repel_loss.item():.4f}"
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Recon={recon.item():.4f}, "
                  f"beta*KL={beta * kl.item():.4f}{extra_str}")
    
    # Restore best model if we tracked validation performance
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  Restored best model from epoch {best_epoch} (test recon loss: {best_test_loss:.6f})")
    
    if return_history:
        return model, history
    return model


# =============================================================================
# IMAGE EXPERIMENTS
# =============================================================================


def _mnist_forward_loss(model, x, beta):
    """Dispatch forward pass and loss for any supported MNIST model type."""
    from .models import MNISTGaussianVAE, MixedCircleVAE, MixedTorusVAE

    if isinstance(model, MNISTGaussianVAE):
        x_recon, mu, logvar, z = model(x)
        return model.loss(x, x_recon, mu, logvar, beta)

    if isinstance(model, MixedTorusVAE):
        x_recon, z_mu, z_logvar, mu1, sigma1, mu2, sigma2, z_d, t1, t2 = model(x)
        return model.loss(x, x_recon, z_mu, z_logvar, mu1, sigma1, mu2, sigma2, beta)

    if isinstance(model, MixedCircleVAE):
        x_recon, z_mu, z_logvar, mu_theta, sigma_theta, z_d, theta = model(x)
        return model.loss(x, x_recon, z_mu, z_logvar, mu_theta, sigma_theta, beta)

    raise ValueError(f"Unsupported model type: {type(model)}")


def _mnist_anchor_loss(model, anchor_x, anchor_targets):
    """
    Compute the anchor loss for the periodic coordinates only.

    The Gaussian subspace z_d of MixedTorusVAE and MixedCircleVAE is
    intentionally NOT anchored — digit organisation emerges from
    reconstruction pressure without any heuristics.
    """
    from .models import MixedCircleVAE, MixedTorusVAE

    if isinstance(model, MixedTorusVAE):
        _, _, mu1, _, mu2, _ = model.encode(anchor_x)

        t1 = anchor_targets['theta1']
        t2 = anchor_targets['theta2']
        if not isinstance(t1, torch.Tensor):
            t1 = torch.tensor(t1, dtype=torch.float32, device=anchor_x.device)
            t2 = torch.tensor(t2, dtype=torch.float32, device=anchor_x.device)
        return angular_anchor_loss(mu1, t1) + angular_anchor_loss(mu2, t2)

    if isinstance(model, MixedCircleVAE):
        _, _, mu_theta, _ = model.encode(anchor_x)

        t = anchor_targets['theta']
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=anchor_x.device)
        return angular_anchor_loss(mu_theta, t)

    return torch.tensor(0.0, device=anchor_x.device)


def train_mnist_vae(model, train_loader, val_loader,
                    n_epochs=100, lr=1e-4, beta=0.3,
                    beta_anneal_epochs=None, print_every=10,
                    anchor_x=None, anchor_targets=None, anchor_weight=10.0,
                    device=None, vae_progress=None, checkpoint_path=None):
    """
    Mini-batch training loop for CNN-based MNIST VAEs.

    Mirrors train_ronchigram_vae in structure.  All periodic anchor
    coordinates are optionally constrained via anchor_weight; Gaussian
    subspace coordinates are always left free.

    Args:
        model:             Any of the MNIST* / Mixed* classes.
        train_loader:      Training DataLoader (first element of each batch is x).
        val_loader:        Validation DataLoader.
        n_epochs:          Number of full passes over the training set.
        lr:                Adam learning rate.
        beta:              Final KL weight (β-VAE style).
        beta_anneal_epochs: Epochs over which β is linearly warmed up from 0.
                            Defaults to n_epochs // 4 (first quarter of training).
                            Validation records are tracked from this epoch onward.
        print_every:       Epoch print frequency.
        anchor_x:          Float32 tensor [K, 784] of anchor images (CPU).
        anchor_targets:    Dict mapping coordinate names to target values.
        anchor_weight:     Multiplicative weight for the anchor loss.
        device:            'cuda' or 'cpu'. Auto-detected if None.
        checkpoint_path:   Optional path to save model checkpoint whenever validation loss improves
                          (after annealing). Results in immediate saving of best state.

    Returns:
        Trained model (CPU, best validation checkpoint restored).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if beta_anneal_epochs is None:
        beta_anneal_epochs = n_epochs // 4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_steps = n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.1)

    has_anchors = anchor_x is not None and anchor_targets is not None
    anchor_x_dev = anchor_x.to(device) if has_anchors else None

    best_val = float('inf')
    best_state = None
    # Begin tracking validation records as soon as beta annealing is complete.
    # The model operates at full regularisation from this point onward, so any
    # improvement genuinely reflects better representation quality.
    checkpoint_start = beta_anneal_epochs

    for epoch in range(n_epochs):
        model.train()
        current_beta = beta * min(1.0, (epoch + 1) / max(beta_anneal_epochs, 1))

        e_loss, e_recon, e_kl, n_batches = 0., 0., 0., 0

        for batch in train_loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()

            loss, recon, kl = _mnist_forward_loss(model, x_batch, current_beta)

            if has_anchors:
                a_loss = _mnist_anchor_loss(model, anchor_x_dev, anchor_targets)
                loss = loss + anchor_weight * a_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            e_loss += loss.item()
            e_recon += recon.item()
            e_kl += kl.item()
            n_batches += 1

        # Validation at final beta (unbiased toward early low-β epochs)
        model.eval()
        val_total = 0.
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                vloss, _, _ = _mnist_forward_loss(model, x, beta)
                val_total += vloss.item()
            val_anchor = _mnist_anchor_loss(model, anchor_x_dev, anchor_targets).item() if has_anchors else 0.0

        val_avg = val_total / max(len(val_loader), 1) + anchor_weight * val_anchor
        if val_avg < best_val and epoch >= checkpoint_start:
            best_val = val_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save intermediate checkpoint if path provided
            if checkpoint_path is not None:
                os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                torch.save(best_state, checkpoint_path)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            ckpt_marker = '  [ckpt]' if val_avg < best_val + 1e-9 and epoch >= checkpoint_start else ''
            prefix = f'[{vae_progress}] ' if vae_progress else ''
            print(f"  {prefix}Epoch {epoch+1:4d}/{n_epochs}  "
                  f"loss={e_loss/n_batches:.4f}  "
                  f"recon={e_recon/n_batches:.4f}  "
                  f"kl={e_kl/n_batches:.4f}  "
                  f"val={val_avg:.4f}  beta={current_beta:.4f}{ckpt_marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to('cpu')
    print(f"  Training complete. Best val loss: {best_val:.4f}")
    return model


# =============================================================================
# CHECKPOINTS
# =============================================================================


def load_model_from_checkpoint(model, checkpoint_path):
    """
    Load a trained model state from a checkpoint file.

    Args:
        model: torch.nn.Module instance to load weights into.
        checkpoint_path: Path to the checkpoint file (.pt file containing state_dict).

    Returns:
        The model with loaded weights (on CPU).
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model
