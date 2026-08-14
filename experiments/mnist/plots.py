"""
The two MNIST figures.

Each ``plot_*_main_figure`` compares three latent spaces on the same data: the
Gaussian VAE baseline, the topology-aware VAE, and the same VAE with anchoring.
Where the synthetic figures can simply scatter the latent coordinates, these
have to *decode*, because the only way to see whether a latent coordinate means
what it should is to look at the image it generates.

So each figure carries, alongside the scatter panels:

- a mosaic sweeping the latent coordinate over its full range, next to the
  ground-truth sweep it should reproduce;
- geodesics between pairs of images, walked in the latent space and compared
  against the same walk taken in the true coordinates.

The Gaussian baseline has no periodic coordinate to sweep, so its latent space
is shown through a UMAP embedding coloured by the true angle. A torn circle
shows up as a colour discontinuity.

``plot_*_decoded_ring`` is a diagnostic rather than a paper figure: it decodes
one full turn of the latent circle for each digit, which is the quickest way to
see whether anchoring landed where it should.

Author: Jilles van Hulst
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from topovae.layers import circular_distance
from topovae.models import MixedCircleVAE, MixedTorusVAE, MNISTGaussianVAE

from .._panels import (
    _digit_rank_map,
    _draw_order,
    _interp_angle,
    _scatter_coords,
    _scatter_digits,
    _set_theta_axes_style,
    _show_image_grid,
    _umap_2d,
    image_grid_height,
    label_image_grid,
    make_image_aware_figure,
)
from .._style import (
    angle_to_color,
    apply_publication_style,
    save_figure,
    scale_fonts_to_page,
)


#: Shared layout for both MNIST grids, in inches.  ``LEFT_IN`` is wide enough
#: for the panels' y-axis labels *and* the rotated row labels beyond them (see
#: ``label_image_grid``); the two figures and the width estimate below all read
#: these so they cannot drift apart.
CW_IN = 4.0
GAP_COL_IN = 0.60
GAP_ROW_IN = 0.90
LEFT_IN = 0.95
RIGHT_IN = 0.15
TOP_IN = 0.55
BOTTOM_IN = 0.65


def _image_nominal_w_in(n_cols):
    """Nominal width of an image figure before the tight-bbox crop."""
    return LEFT_IN + n_cols * CW_IN + (n_cols - 1) * GAP_COL_IN + RIGHT_IN

apply_publication_style()


# =============================================================================
# ROTATED MNIST — S¹
# =============================================================================


def _rotated_select_endpoints_multi(data, geo_specs, k_nearest=5, seed=42):
    """Find nearest data-index pairs for a list of (digit, theta_a, theta_b) specs.

    Returns list of (digit, idx_a, idx_b).  Searches are restricted to samples of
    the specified digit for each spec.
    """
    rng = np.random.RandomState(seed)
    labels = data['labels']
    theta  = data['theta']
    result = []
    for digit, theta_a, theta_b in geo_specs:
        mask    = np.where(labels == digit)[0]
        if len(mask) == 0:
            raise ValueError(
                f"_rotated_select_endpoints_multi: no samples found for digit {digit!r} "
                f"in data['labels'] (dtype={labels.dtype}, unique values={np.unique(labels).tolist()}). "
                "Check that the digit list passed to plot_rotated_main_figure matches the dataset."
            )
        theta_d = theta[mask]

        def _pick(target, _mask=mask, _theta_d=theta_d):
            dists = circular_distance(_theta_d, target)
            top_k = np.argsort(dists)[:k_nearest]
            return int(_mask[top_k[rng.randint(0, len(top_k))]])

        result.append((digit, _pick(theta_a), _pick(theta_b)))
    return result


def _rotated_ground_truth_geodesic_grid(data, endpoint_specs, n_frames=9, pad=2):
    """Ground-truth geodesic grid from a list of (digit, idx_a, idx_b) specs.

    Each mosaic row shows the nearest real sample of the specified digit at each
    angle along the interpolated path from θ_a to θ_b.
    """
    theta  = data['theta']
    labels = data['labels']
    cell   = 28 + pad
    n_geo  = len(endpoint_specs)
    mosaic = np.ones((n_geo * cell - pad, n_frames * cell - pad), dtype=np.float32)
    for ri, (digit, idx_a, idx_b) in enumerate(endpoint_specs):
        interp_angles = _interp_angle(float(theta[idx_a]), float(theta[idx_b]), n_frames)
        mask    = np.where(labels == digit)[0]
        theta_d = theta[mask]
        for ci, ang in enumerate(interp_angles):
            dists = circular_distance(theta_d, ang)
            best  = mask[int(np.argmin(dists))]
            r0, c0 = ri * cell, ci * cell
            mosaic[r0:r0 + 28, c0:c0 + 28] = np.clip(data['x'][best].reshape(28, 28), 0, 1)
    return mosaic


def _rotated_model_geodesic_grid(model, x_data, endpoint_specs, n_frames=9, pad=2):
    """Model geodesic grid from a list of (digit, idx_a, idx_b) specs.

    Encodes the actual endpoint images, interpolates in latent space, and decodes.
    The MixedCircleVAE path is geodesic on S¹ for θ and linear for z_d;
    the Gaussian path is linear throughout.  Endpoints are genuine reconstructions.
    """
    cell  = 28 + pad
    n_geo = len(endpoint_specs)
    mosaic = np.ones((n_geo * cell - pad, n_frames * cell - pad), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_frames)
    model.eval()
    with torch.no_grad():
        for ri, (digit, idx_a, idx_b) in enumerate(endpoint_specs):
            xa = torch.tensor(x_data[idx_a:idx_a + 1], dtype=torch.float32)
            xb = torch.tensor(x_data[idx_b:idx_b + 1], dtype=torch.float32)
            if isinstance(model, MixedCircleVAE):
                z_mu_a, _, mu_th_a, _ = model.encode(xa)
                z_mu_b, _, mu_th_b, _ = model.encode(xb)
                z_interp = (
                    (1 - t)[:, None] * z_mu_a.cpu().numpy()
                    + t[:, None] * z_mu_b.cpu().numpy()
                )
                th_interp = _interp_angle(float(mu_th_a), float(mu_th_b), n_frames)
                imgs = model.decode(
                    torch.tensor(z_interp, dtype=torch.float32),
                    torch.tensor(th_interp[:, None], dtype=torch.float32),
                )
            elif isinstance(model, MNISTGaussianVAE):
                mu_a, _ = model.encode(xa)
                mu_b, _ = model.encode(xb)
                z_interp = (
                    (1 - t)[:, None] * mu_a.cpu().numpy()
                    + t[:, None] * mu_b.cpu().numpy()
                )
                imgs = model.decode(torch.tensor(z_interp, dtype=torch.float32))
            else:
                return None
            imgs_np = imgs.detach().cpu().numpy()
            for ci in range(n_frames):
                r0, c0 = ri * cell, ci * cell
                mosaic[r0:r0 + 28, c0:c0 + 28] = np.clip(imgs_np[ci].reshape(28, 28), 0, 1)
    return mosaic


def _rotated_ring_mosaic_gt(data, digits, n_angle=12, pad=2):
    """Ground-truth ring: ``n_digits × n_angle`` mosaic of nearest real samples.

    For each digit and uniformly-spaced target angle, the nearest training sample
    (by circular distance in the true θ space) is placed in the grid.  The result
    shows a 'filmstrip' of the true data manifold around the full S¹.
    """
    angles = np.linspace(-np.pi, np.pi, n_angle, endpoint=False)
    cell   = 28 + pad
    mosaic = np.ones((len(digits) * cell - pad, n_angle * cell - pad), dtype=np.float32)
    for ri, d in enumerate(digits):
        mask    = np.where(data['labels'] == d)[0]
        theta_d = data['theta'][mask]
        for ci, ang in enumerate(angles):
            dists  = circular_distance(theta_d, ang)
            best   = mask[int(np.argmin(dists))]
            r0, c0 = ri * cell, ci * cell
            mosaic[r0:r0 + 28, c0:c0 + 28] = np.clip(
                data['x'][best].reshape(28, 28), 0, 1)
    return mosaic


def _rotated_ring_mosaic_model(model, x_data, theta_encoded, labels, digits,
                                n_angle=12, pad=2):
    """Model ring: encode the nearest-angle sample, reconstruct, place in grid.

    ``theta_encoded`` defines the "angle" axis for this model:
    - For ``MixedCircleVAE``: use ``z_codes['theta']`` (the VAE's encoded S¹ coord).
    - For ``MNISTGaussianVAE``: use ``arctan2(z[:,1], z[:,0])`` (first-two-dim proxy).

    The nearest sample for each digit and target angle is encoded then decoded,
    so the output is the model's reconstruction (not raw pixels).  Columns form a
    ring around S¹; rows are digit classes.  If the model learned a clean circle,
    the ring will look smooth and digit-consistent.  If not, it will be noisy —
    that's the honest result.
    """
    angles  = np.linspace(-np.pi, np.pi, n_angle, endpoint=False)
    cell    = 28 + pad
    mosaic  = np.ones((len(digits) * cell - pad, n_angle * cell - pad), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for ri, d in enumerate(digits):
            mask    = np.where(labels == d)[0]
            theta_d = theta_encoded[mask]
            for ci, ang in enumerate(angles):
                dists = circular_distance(theta_d, ang)
                best  = mask[int(np.argmin(dists))]
                x = torch.tensor(x_data[best:best + 1], dtype=torch.float32)
                if isinstance(model, MixedCircleVAE):
                    z_d, _, mu_th, _ = model.encode(x)
                    recon = model.decode(z_d, mu_th)
                elif isinstance(model, MNISTGaussianVAE):
                    mu, _ = model.encode(x)
                    recon = model.decode(mu)
                else:
                    recon = x
                img = np.clip(recon.cpu().numpy().reshape(-1, 28, 28)[0], 0, 1)
                r0, c0 = ri * cell, ci * cell
                mosaic[r0:r0 + 28, c0:c0 + 28] = img
    return mosaic


def _slice_deg_label(c1, c2=None):
    """Legend text for a slice: one angle, or both angles as a pair on one line."""
    if c2 is None:
        return r'$%+d^\circ$' % int(round(c1))
    return r'$(%+d^\circ, %+d^\circ)$' % (int(round(c1)), int(round(c2)))


def _slice_quadrants(slice_coords):
    """The four ``(x0, y0, mask, label)`` quadrants of a sliced style panel.

    ``slice_coords`` is a list of one or two angle arrays (each in (-pi, pi]):

    - one angle (S^1): the circle is cut into four equal arcs, read left to
      right then top to bottom.
    - two angles (T^2): theta_1 sets the column (negative left, positive right)
      and theta_2 the row (positive top, negative bottom), so every quadrant
      differs in both angles.

    ``x0, y0`` are the inset's lower-left corner in axes fractions; ``mask``
    selects that slice's points; ``label`` is the slice angle(s) for the box.
    """
    # Inset corners in reading order: top-left, top-right, bottom-left, bottom-right.
    corners = [(0.0, 0.5), (0.5, 0.5), (0.0, 0.0), (0.5, 0.0)]
    if len(slice_coords) == 1:
        a = np.asarray(slice_coords[0])
        edges = np.linspace(-np.pi, np.pi, 5)
        quads = []
        for k, (x0, y0) in enumerate(corners):
            mask = (a >= edges[k]) & (a < edges[k + 1])
            mid = np.rad2deg(0.5 * (edges[k] + edges[k + 1]))
            quads.append((x0, y0, mask, _slice_deg_label(mid)))
        return quads

    t1, t2 = (np.asarray(c) for c in slice_coords)
    specs = [
        (0.0, 0.5, (t1 < 0),  (t2 >= 0), -90, +90),
        (0.5, 0.5, (t1 >= 0), (t2 >= 0), +90, +90),
        (0.0, 0.0, (t1 < 0),  (t2 < 0),  -90, -90),
        (0.5, 0.0, (t1 >= 0), (t2 < 0),  +90, -90),
    ]
    return [(x0, y0, m1 & m2, _slice_deg_label(c1, c2))
            for x0, y0, m1, m2, c1, c2 in specs]


def _scatter_digits_sliced(ax, z_style, digit_labels, slice_coords):
    """Fill ``ax`` with a 2x2 grid of angular-slice quadrants of the style space.

    Each quadrant scatters the digit-coloured style codes for one slice of the
    true angle(s) (see ``_slice_quadrants``), on shared axis limits, with the
    slice angle(s) in a legend-styled box at its top right. z_1 is labelled once
    under each column and z_2 once beside each row, at the panels' own padding.
    Colours follow the shared digit-to-tab10 map, so they agree with the digit
    legend for any digit set. The two digits separate within a fixed angle even
    where the scatter pooled over all angles does not.
    """
    emb = np.asarray(z_style)
    if emb.ndim == 2 and emb.shape[1] == 2:
        xlabel, ylabel = r'$z_1$', r'$z_2$'
    else:
        emb, idx = _umap_2d(emb)
        digit_labels = digit_labels[idx]
        slice_coords = [np.asarray(c)[idx] for c in slice_coords]
        xlabel, ylabel = 'UMAP 1', 'UMAP 2'

    # Shared limits centred on the data, with enough margin that the slices are
    # not clipped: a percentile radius about the median, not a hard min/max.
    center = np.median(emb, axis=0)
    half = np.percentile(np.abs(emb - center), 99, axis=0) * 1.1
    lo, hi = center - half, center + half

    digit_to_rank = _digit_rank_map(digit_labels)
    cmap = plt.cm.tab10

    ax.set_xticks([])
    ax.set_yticks([])
    for x0, y0, mask, label in _slice_quadrants(slice_coords):
        inset = ax.inset_axes([x0, y0, 0.5, 0.5])
        if mask.any():
            ranks = np.array([digit_to_rank[int(d)] for d in digit_labels[mask]])
            order = _draw_order(int(mask.sum()), seed=42)
            inset.scatter(emb[mask, 0][order], emb[mask, 1][order],
                          c=cmap(ranks[order]),
                          s=4, alpha=0.4, rasterized=True)
        inset.set_xlim(lo[0], hi[0])
        inset.set_ylim(lo[1], hi[1])
        inset.set_xticks([])
        inset.set_yticks([])
        # z_1 under each column, z_2 beside each row — bottom/left insets only.
        inset.set_xlabel(xlabel if y0 == 0.0 else '')
        inset.set_ylabel(ylabel if x0 == 0.0 else '')
        # Angle box styled as a standard legend, but with no handle or marker.
        inset.legend([Line2D([], [], linestyle='none', marker='none')], [label],
                     loc='upper right', handlelength=0, handletextpad=0,
                     borderaxespad=0.2, framealpha=0.9, fontsize=12)


def plot_rotated_main_figure(data, z_gauss, z_circ_noanch, z_circ,
                              model_gauss, model_circ_noanch, model_circ,
                              sliced_style=False, save_path=None):
    """Publication-quality rotated MNIST comparison figure.

    Columns: ground truth | Gaussian VAE | [Circle VAE] | Circle VAE (anchored)
    The unanchored Circle VAE column is omitted when ``z_circ_noanch`` is None.

    Rows
    ----
    0  Latent-space scatter coloured by true θ (HSV rainbow).
       • GT / Circle VAEs: unit-circle scatter of (cos θ, sin θ).
       • Gaussian VAE:     UMAP of full z, coloured by true θ.
    1  Style-space scatter coloured by digit class (tab10).
       • GT:               UMAP of raw pixels (only honest baseline available).
       • Gaussian VAE:     same UMAP as row 0, recoloured by digit.
       • Circle VAEs:      UMAP or direct scatter of z_d.
    2  Full-circle ring: n_digits × n_angle grid of nearest-angle reconstructions.
    3  Geodesic interpolation: 2 strips per digit, shared endpoint indices.
    """

    digits   = np.sort(np.unique(data['labels'])).tolist()
    n_digits = len(digits)
    colors   = angle_to_color(data['theta'])

    # ── Pre-compute expensive UMAPs once ──────────────────────────────────── #
    # Gaussian latent: one UMAP reused for both row-0 (θ-coloured) and
    # row-1 (digit-coloured) — same embedding, different colour array.
    gauss_umap = _umap_2d(np.asarray(z_gauss['z']))
    # GT style (row-1 only): UMAP of raw pixels, the only honest baseline.
    gt_pixel_umap = _umap_2d(data['x'])

    # ── Column definitions ─────────────────────────────────────────────────── #
    # Each entry: (label, theta, z_style, style_umap_cache, model, z_codes)
    #   theta           : S¹ coordinate for row-0 unit-circle scatter.
    #                     None for Gaussian VAE (uses gauss_umap instead).
    #   z_style         : coordinates for row-1 style scatter.
    #   style_umap_cache: pre-computed (emb, idx) for z_style, or None.
    #   model / z_codes : for geodesic and ring rows.
    z_sty_a = z_circ.get('z_d', z_circ.get('z', z_gauss['z']))
    cols = [
        ('True data',
         data['theta'], data['x'],  gt_pixel_umap,  None,        None,   False),
        ('Gaussian VAE',
         None,          z_gauss['z'], gauss_umap,   model_gauss, z_gauss, True),
    ]
    if z_circ_noanch is not None and model_circ_noanch is not None:
        z_sty_na = z_circ_noanch.get('z_d', z_circ_noanch.get('z', z_gauss['z']))
        cols.append(('Circle VAE',
                     z_circ_noanch['theta'], z_sty_na, None,
                     model_circ_noanch, z_circ_noanch, False))
    cols.append(('Circle VAE (anchored)',
                 z_circ['theta'], z_sty_a, None, model_circ, z_circ, False))
    n_cols = len(cols)

    # ── Layout constants ───────────────────────────────────────────────────── #
    _n_frames  = 9        # geodesic interpolation steps
    _n_angle   = 12       # ring: number of uniformly-spaced angles
    _pad       = 2        # pixel gap between images
    _cw        = CW_IN    # column content-width in inches

    # ── Two geodesics per digit ────────────────────────────────────────────── #
    # All arcs ≤ 90° (π/2). Digits alternate direction for visual variety.
    _P = np.pi
    _MAX_ARC =  _P / 2
    _templates = [
        [( 0.0,         +_MAX_ARC), ( _P / 3,     -_MAX_ARC)],
        [( 0.0,         -_MAX_ARC), (-_P / 3,      _MAX_ARC)],
        [( 0.0,         +_MAX_ARC), ( _P * 5 / 8, -_MAX_ARC)],
        [( 0.0,         -_MAX_ARC), (-_P * 5 / 8,  _MAX_ARC)],
    ]
    geo_specs = [
        (digit, theta_a, theta_a + arc)
        for di, digit in enumerate(digits)
        for theta_a, arc in _templates[di % len(_templates)]
    ]
    endpoint_specs = _rotated_select_endpoints_multi(data, geo_specs)
    n_geo = len(endpoint_specs)

    row_heights = [
        _cw,
        _cw,
        image_grid_height(_cw, n_digits, _n_angle,  28, 28, _pad),
        image_grid_height(_cw, n_geo,    _n_frames, 28, 28, _pad),
    ]
    scale_fonts_to_page(_image_nominal_w_in(n_cols))
    fig, axes = make_image_aware_figure(
        row_heights, n_cols,
        col_width_in=_cw, gap_row_in=GAP_ROW_IN, gap_col_in=GAP_COL_IN,
        left_in=LEFT_IN, right_in=RIGHT_IN, top_in=TOP_IN, bottom_in=BOTTOM_IN,
    )

    # ── Row 0: latent-space scatter coloured by true θ ────────────────────── #
    for c, (title, theta, z_style, style_cache, model, z_codes, is_gaussian) in enumerate(cols):
        ax = axes[0, c]
        if is_gaussian:
            _scatter_coords(ax, z_gauss['z'], colors, None,
                            xlabel='UMAP 1', ylabel='UMAP 2', umap_cache=gauss_umap)
        else:
            ring_order = _draw_order(len(theta), seed=42)
            ax.scatter(np.cos(np.asarray(theta))[ring_order],
                       np.sin(np.asarray(theta))[ring_order],
                       c=np.asarray(colors)[ring_order],
                       s=4, alpha=0.4, rasterized=True)
            ax.set_aspect('equal')
            ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
            # No ticks: the cos/sin axis labels name the axes, and the circle is
            # the point, not its coordinates.
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(r'$\cos\theta$' if c == 0 else r'$\cos z_\theta$')
            ax.set_ylabel(r'$\sin\theta$' if c == 0 else r'$\sin z_\theta$')

    # ── Row 1: style-space scatter coloured by digit class ───────────────────  #
    # Columns 0 (ground truth) and 1 (Gaussian VAE) keep the aggregated scatter;
    # the Circle VAE columns split into angular-slice quadrants when requested.
    for c, (title, theta, z_style, style_cache, model, z_codes, is_gaussian) in enumerate(cols):
        if sliced_style and c >= 2:
            _scatter_digits_sliced(axes[1, c], z_style, data['labels'], [data['theta']])
        else:
            _scatter_digits(axes[1, c], z_style, data['labels'],
                            umap_cache=style_cache, show_legend=(c == 0))

    # ── Row 2: full-circle ring ───────────────────────────────────────────── #
    # Gaussian VAE has no reliable angular coordinate → show GT nearest samples.
    for c, (title, theta, z_style, style_cache, model, z_codes, is_gaussian) in enumerate(cols):
        ax = axes[2, c]
        if model is None or is_gaussian:
            mosaic = _rotated_ring_mosaic_gt(data, digits, n_angle=_n_angle, pad=_pad)
        else:
            mosaic = _rotated_ring_mosaic_model(
                model, data['x'], theta, data['labels'], digits,
                n_angle=_n_angle, pad=_pad)
        _show_image_grid(ax, mosaic)

    # ── Row 3: geodesic interpolation ─────────────────────────────────────── #
    for c, (title, theta, z_style, style_cache, model, z_codes, is_gaussian) in enumerate(cols):
        ax = axes[3, c]
        mosaic = (
            _rotated_ground_truth_geodesic_grid(data, endpoint_specs, n_frames=_n_frames, pad=_pad)
            if model is None else
            _rotated_model_geodesic_grid(model, data['x'], endpoint_specs, n_frames=_n_frames, pad=_pad)
        )
        if mosaic is not None:
            _show_image_grid(ax, mosaic)

    label_image_grid(fig, axes, col_titles=[c[0] for c in cols],
                     row_titles=['Latent space', 'Style space', 'Ring', 'Geodesics'])

    if save_path is not None:
        save_figure(save_path)


def plot_rotated_decoded_ring(model_circ, z_circ, data, n_angle=12, save_path=None):

    labels = data['labels']
    digits = np.sort(np.unique(labels)).tolist()
    n_digits = len(digits)
    n_cols = n_angle + 1
    angles = np.linspace(-np.pi, np.pi, n_angle, endpoint=False)
    angles = np.append(angles, angles[0])   # wrap back to start

    fig, axes = plt.subplots(n_digits, n_cols, figsize=(n_cols * 0.9, n_digits * 1.0))
    # Ensure axes is always 2-D even when n_digits == 1
    if n_digits == 1:
        axes = axes[np.newaxis, :]
    z_d_all = z_circ['z_d']

    model_circ.eval()
    with torch.no_grad():
        for row, d in enumerate(digits):
            mask = labels == d
            zd_mean = torch.tensor(z_d_all[mask].mean(axis=0)[None, :], dtype=torch.float32)
            for j, ang in enumerate(angles):
                th = torch.tensor([[ang]], dtype=torch.float32)
                img = model_circ.decode(zd_mean, th).squeeze().numpy().reshape(28, 28)
                axes[row, j].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
                axes[row, j].axis('off')
            axes[row, 0].set_ylabel(str(d), rotation=0, labelpad=8, va='center', fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        save_figure(save_path)


# =============================================================================
# SHIFTED MNIST — T²
# =============================================================================


def _shifted_decode_torus_grid(model, n_grid=10, pad=2, z_d_ref=None,
                                z_d_pool=None, theta1_pool=None, theta2_pool=None):
    """Decode a n_grid × n_grid torus sweep for MixedTorusVAE.

    z_d selection priority (highest to lowest):
    1. ``z_d_pool`` + ``theta1_pool`` + ``theta2_pool``: for each grid cell at
       (t1, t2), look up the nearest encoded training sample in (θ₁, θ₂) space
       from the pool and use its z_d.  This keeps each decoded image inside the
       training manifold of the target digit class, avoiding centroid artifacts.
    2. ``z_d_ref``: fixed reference vector (e.g. near-unshifted exemplar).
    3. Fallback: zero vector (prior mean).

    Returns the mosaic float32 array, or ``None`` for unsupported models.
    """
    cell = 28 + pad
    side = n_grid * cell - pad
    mosaic = np.ones((side, side), dtype=np.float32)
    angles = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)

    # Pre-convert pool arrays for vectorised lookup
    if z_d_pool is not None:
        _zd_pool  = np.asarray(z_d_pool,   dtype=np.float32)
        _t1_pool  = np.asarray(theta1_pool, dtype=np.float32)
        _t2_pool  = np.asarray(theta2_pool, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i, t2 in enumerate(angles):
            for j, t1 in enumerate(angles):
                if isinstance(model, MixedTorusVAE):
                    if z_d_pool is not None:
                        # Nearest-neighbour lookup in encoded (θ₁, θ₂) space.
                        d = np.sqrt(circular_distance(_t1_pool, t1) ** 2 +
                                    circular_distance(_t2_pool, t2) ** 2)
                        nn = int(np.argmin(d))
                        zd = torch.tensor(_zd_pool[nn:nn + 1], dtype=torch.float32)
                    elif z_d_ref is not None:
                        zd = torch.tensor(z_d_ref[None, :], dtype=torch.float32)
                    else:
                        zd = torch.zeros(1, model.gaussian_dim, dtype=torch.float32)
                    xr = model.decode(
                        zd,
                        torch.tensor([[t1]], dtype=torch.float32),
                        torch.tensor([[t2]], dtype=torch.float32),
                    )
                else:
                    return None
                r0, c0 = i * cell, j * cell
                mosaic[r0:r0 + 28, c0:c0 + 28] = np.clip(
                    xr.squeeze().numpy().reshape(28, 28), 0, 1)
    return mosaic


def _shifted_ground_truth_torus_grid(data, n_grid=10, pad=2):
    orig_mask = (data['dx'] == 0) & (data['dy'] == 0)
    exemplar = data['x'][np.where(orig_mask)[0][0]].reshape(28, 28)
    cell = 28 + pad
    h = n_grid * cell - pad
    mosaic = np.ones((h, h), dtype=np.float32)
    # Use the same angle grid as the model decoded grid so that the cell at
    # (row i, col j) corresponds to the same (theta1, theta2) in both panels.
    # Inversion: theta = (2pi*d/28 + pi) % (2pi) - pi  =>  d = (28*theta/(2pi)) % 28
    # Verify: theta=0 -> d=0 (unshifted), theta=pi/2 -> d=7, theta=-pi -> d=14.
    angles = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    for i, theta2 in enumerate(angles):
        dy = int(round(28 * theta2 / (2 * np.pi))) % 28
        for j, theta1 in enumerate(angles):
            dx = int(round(28 * theta1 / (2 * np.pi))) % 28
            shifted = np.roll(np.roll(exemplar, dx, axis=0), dy, axis=1)
            r0, c0 = i * cell, j * cell
            mosaic[r0:r0 + 28, c0:c0 + 28] = np.clip(shifted, 0, 1)
    return mosaic


def _shifted_geodesic_data_images(data, idx_a, idx_b, n_frames=8, digit=None):
    """Ground-truth geodesic: pick the nearest real data sample at each interpolated angle.

    If ``digit`` is given, the nearest-neighbour search is restricted to samples
    of that digit class, ensuring the strip stays within one class.
    """
    theta1_path = _interp_angle(data['theta1'][idx_a], data['theta1'][idx_b], n_frames)
    theta2_path = _interp_angle(data['theta2'][idx_a], data['theta2'][idx_b], n_frames)

    if digit is not None and 'labels' in data:
        pool = np.where(data['labels'] == digit)[0]
    else:
        pool = np.arange(len(data['theta1']))

    pool_t1 = data['theta1'][pool]
    pool_t2 = data['theta2'][pool]

    images = []
    for t1, t2 in zip(theta1_path, theta2_path):
        dist = np.sqrt(circular_distance(pool_t1, t1) ** 2 + circular_distance(pool_t2, t2) ** 2)
        nearest = int(pool[np.argmin(dist)])
        images.append(data['x'][nearest].reshape(28, 28))
    return images


def _shifted_geodesic_model_images(model, z_codes, idx_a, idx_b, n_frames=8):
    """Decode images along the latent geodesic between two encoded endpoints.

    For ``MixedTorusVAE``: the style code z_d is **linearly interpolated** from
    the actual encoded z_d of endpoint A to that of endpoint B.  The torus angles
    (θ₁, θ₂) are interpolated geodesically on T².  This means:
    - Frame 0   → full reconstruction of endpoint A (correct digit, correct shift).
    - Frame N-1 → full reconstruction of endpoint B (correct digit, correct shift).
    - Interior  → smooth path in both style and angle.

    For ``MNISTGaussianVAE``: linearly interpolates the full latent vector.
    """
    model.eval()
    t = np.linspace(0.0, 1.0, n_frames)

    if isinstance(model, MixedTorusVAE):
        theta1_path = _interp_angle(z_codes['theta1'][idx_a], z_codes['theta1'][idx_b], n_frames)
        theta2_path = _interp_angle(z_codes['theta2'][idx_a], z_codes['theta2'][idx_b], n_frames)
        # Linearly interpolate z_d between the actual encoded
        # representations of the two endpoints.
        z_d_a = z_codes['z_d'][idx_a].astype(np.float32)
        z_d_b = z_codes['z_d'][idx_b].astype(np.float32)
        zd_path = z_d_a[None, :] + t[:, None] * (z_d_b - z_d_a)[None, :]
        with torch.no_grad():
            imgs = model.decode(
                torch.tensor(zd_path, dtype=torch.float32),
                torch.tensor(theta1_path[:, None], dtype=torch.float32),
                torch.tensor(theta2_path[:, None], dtype=torch.float32),
            ).detach().cpu().numpy()
        return [img.reshape(28, 28) for img in imgs]

    # MNISTGaussianVAE: linear interpolation of the full latent vector.
    z_a = np.asarray(z_codes['z'][idx_a], dtype=np.float32)
    z_b = np.asarray(z_codes['z'][idx_b], dtype=np.float32)
    z_path = z_a[None, :] + t[:, None] * (z_b - z_a)[None, :]
    with torch.no_grad():
        imgs = model.decode(torch.tensor(z_path, dtype=torch.float32)).detach().cpu().numpy()
    return [img.reshape(28, 28) for img in imgs]


def _shifted_decode_gaussian_grid(model, z_codes, labels, canonical_digit=3,
                                   n_grid=10, pad=2):
    """Decode a 2-D slice of the Gaussian latent, digit-consistent via nearest-neighbour.

    Sweeps z[:,0] (columns) and z[:,1] (rows) over the 5th–95th percentile range
    of the encoded codes for ``canonical_digit``.  For each cell the remaining
    dimensions (z[2:]) are taken from the nearest canonical-digit sample in the
    (z₁, z₂) plane, keeping every decoded image firmly inside the digit's training
    manifold rather than using a centroid that can lie between clusters.

    Demonstrates that the Gaussian latent is locally smooth in the (z₁, z₂) plane
    but cannot tile the torus boundary.
    """
    z = np.asarray(z_codes['z'])
    if z.shape[1] < 2:
        return None

    if labels is not None and canonical_digit is not None:
        mask = labels == canonical_digit
        if not mask.any():
            mask = np.ones(len(z), dtype=bool)
    else:
        mask = np.ones(len(z), dtype=bool)

    z_digit = z[mask]
    lo0, hi0 = np.percentile(z_digit[:, 0], [5, 95])
    lo1, hi1 = np.percentile(z_digit[:, 1], [5, 95])

    # Column direction: z[:,0] increases left→right
    # Row direction: z[:,1] increases top→bottom (matching imshow convention where
    # row 0 is the top — consistent with how the torus grid places θ=-π at the top)
    grid0 = np.linspace(lo0, hi0, n_grid)
    grid1 = np.linspace(lo1, hi1, n_grid)

    cell = 28 + pad
    side = n_grid * cell - pad
    mosaic = np.ones((side, side), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i, z1_val in enumerate(grid1):
            for j, z0_val in enumerate(grid0):
                # For each cell, find the nearest canonical-digit sample in the
                # (z₁, z₂) plane and use its z[2:] as the fixed tail dimensions.
                # This keeps every decoded image inside the digit's training manifold.
                d2 = (z_digit[:, 0] - z0_val) ** 2 + (z_digit[:, 1] - z1_val) ** 2
                nn  = int(np.argmin(d2))
                z_sample = z_digit[nn].copy().astype(np.float32)
                z_sample[0] = float(z0_val)
                z_sample[1] = float(z1_val)
                xr = model.decode(torch.tensor(z_sample[None, :], dtype=torch.float32))
                r0, c0 = i * cell, j * cell
                mosaic[r0:r0 + 28, c0:c0 + 28] = np.clip(
                    xr.squeeze().numpy().reshape(28, 28), 0, 1)
    return mosaic


def _shifted_select_geodesic_endpoints(data, geo_specs, k_nearest=5, seed=42):
    """Find nearest data-index pairs for a list of geodesic specifications.

    Parameters
    ----------
    data : dict
        Dataset dict with ``'labels'``, ``'theta1'``, ``'theta2'``.
    geo_specs : list of (digit, th1_a, th2_a, th1_b, th2_b)
        Each entry defines one geodesic strip: the digit class and the start/end
        torus angles in radians.  The search for each endpoint is restricted to
        samples of the specified digit.
    k_nearest : int
        Number of nearest neighbours to consider before picking one at random.
    seed : int
        Random seed for reproducible neighbour selection.

    Returns
    -------
    list of (digit, idx_a, idx_b)
    """
    rng = np.random.RandomState(seed)
    labels = data['labels']
    result = []
    for digit, th1_a, th2_a, th1_b, th2_b in geo_specs:
        mask = np.where(labels == digit)[0]
        t1_d = data['theta1'][mask]
        t2_d = data['theta2'][mask]

        def _pick(th1_t, th2_t):
            d = np.sqrt(circular_distance(t1_d, th1_t) ** 2 +
                        circular_distance(t2_d, th2_t) ** 2)
            top_k = np.argsort(d)[:k_nearest]
            return int(mask[top_k[rng.randint(0, len(top_k))]])

        result.append((digit, _pick(th1_a, th2_a), _pick(th1_b, th2_b)))
    return result


def _shifted_build_geodesic_mosaic_gt(data, endpoint_specs, n_frames=8, pad=2):
    """Stack ground-truth geodesic strips into one mosaic.

    Each strip shows the nearest real data sample at each interpolated torus angle,
    restricted to the digit class specified in the corresponding ``endpoint_specs``
    entry, so the GT column always shows the intended digit.
    """
    n_geo = len(endpoint_specs)
    cell = 28 + pad
    mosaic = np.ones((n_geo * cell - pad, n_frames * cell - pad), dtype=np.float32)
    for gi, (digit, idx_a, idx_b) in enumerate(endpoint_specs):
        images = _shifted_geodesic_data_images(
            data, idx_a, idx_b, n_frames=n_frames, digit=digit)
        r0 = gi * cell
        for ci, img in enumerate(images):
            mosaic[r0:r0 + 28, ci * cell:ci * cell + 28] = np.clip(
                img.reshape(28, 28), 0, 1)
    return mosaic


def _shifted_build_geodesic_mosaic_model(model, z_codes, endpoint_specs, n_frames=8, pad=2):
    """Stack model-decoded geodesic strips into one mosaic.

    Each strip uses the actual encoded latent coordinates of the two endpoints
    (see ``_shifted_geodesic_model_images``), so the first and last frames are
    genuine reconstructions of the corresponding ground-truth images.
    """
    n_geo = len(endpoint_specs)
    cell = 28 + pad
    mosaic = np.ones((n_geo * cell - pad, n_frames * cell - pad), dtype=np.float32)
    for gi, (digit, idx_a, idx_b) in enumerate(endpoint_specs):
        images = _shifted_geodesic_model_images(
            model, z_codes, idx_a, idx_b, n_frames=n_frames)
        r0 = gi * cell
        for ci, img in enumerate(images):
            mosaic[r0:r0 + 28, ci * cell:ci * cell + 28] = np.clip(
                img.reshape(28, 28), 0, 1)
    return mosaic


def plot_shifted_main_figure(data, model_entries, sliced_style=False, save_path=None):
    """Publication-quality shifted MNIST comparison figure.

    Layout: 4 rows × (1 + n_models) columns, using ``_make_image_aware_figure``
    for pixel-exact row heights and uniform inter-row gaps.

    Rows
    ----
    0  (θ₁, θ₂) scatter coloured by θ₁ (HSV rainbow).
    1  Style latent coloured by digit class (tab10 UMAP or direct 2-D scatter).
    2  Decoded grid: GT uses ``numpy.roll`` on the first unshifted exemplar of
       ``canonical_digit``; topology VAEs sweep (θ₁, θ₂) with z_d fixed at the
       per-canonical-digit centroid; Gaussian VAE shows a 2-D slice of z₁ × z₂
       at the canonical-digit centroid of the remaining dimensions.
    3  Four stacked geodesic strips (each 90°): pure θ₁, pure θ₂, diagonal,
       anti-diagonal.  GT column shows nearest real data samples; model columns
       decode along the latent geodesic with z_d fixed at the canonical-digit centroid.

    Columns: ground truth | Gaussian VAE | [MixedTorusVAE] | MixedTorusVAE (anchored)
    """

    n_models = len(model_entries)
    n_cols = 1 + n_models

    # ── Layout constants ───────────────────────────────────────────────────── #
    _n_grid   = 10        # decoded grid: n_grid × n_grid cells
    _n_frames = 8         # frames per geodesic strip
    _n_geo    = 6         # number of geodesic strips
    _pad      = 2         # pixel gap between images (within strips and grids)
    _cw       = CW_IN     # column content-width in inches

    # ── Canonical digit (smallest in dataset; typically 3) ────────────────── #
    canonical_digit = int(np.sort(np.unique(data['labels']))[0])

    # ── Six diverse geodesic specifications (digit, θ₁_a, θ₂_a, θ₁_b, θ₂_b) #
    # All shifts are in [90°, 135°].  The first pair is the simplest (digit 3,
    # pure θ₁ shift from rest).  The remaining 5 vary digit, starting angle,
    # and direction so the mosaic shows a broad cross-section of the torus.
    _P = np.pi
    geo_specs = [
        # digit,  start (θ₁, θ₂),         end (θ₁, θ₂)
        (3, 0.0,      0.0,       3*_P/4,  0.0),            # D3: 135° pure θ₁ from origin
        (3, _P/4,     0.0,       _P/4,    3*_P/4),         # D3: 135° pure θ₂ from θ₁=45°
        (3, _P/3,     _P/4,     -_P/3,    _P/4+_P*2/3),   # D3: diagonal, non-zero start, 120°
        (4, 0.0,      0.0,       0.0,     _P*3/4),         # D4: 135° pure θ₂ from origin
        (4, -_P/4,    _P/4,      _P/2,    _P/4),           # D4: pure θ₁, 135°, neg start
        (7, _P/2,     0.0,      -_P/4,   -_P*3/4),        # D7: neg diagonal, 90°-ish, shifted start
    ]

    endpoint_specs = _shifted_select_geodesic_endpoints(data, geo_specs)

    # ── Row heights (derived from pixel aspect ratios) ─────────────────────── #
    row_heights = [
        _cw,                                                              # row 0: square scatter
        _cw,                                                              # row 1: style space
        image_grid_height(_cw, _n_grid, _n_grid, 28, 28, _pad),         # row 2: decoded grid (square)
        image_grid_height(_cw, _n_geo,  _n_frames, 28, 28, _pad),       # row 3: geodesic mosaic
    ]

    scale_fonts_to_page(_image_nominal_w_in(n_cols))
    fig, axes = make_image_aware_figure(
        row_heights, n_cols,
        col_width_in=_cw,
        gap_row_in=GAP_ROW_IN,
        gap_col_in=GAP_COL_IN,
        left_in=LEFT_IN,
        right_in=RIGHT_IN,
        top_in=TOP_IN,
        bottom_in=BOTTOM_IN,
    )

    colors = angle_to_color(data['theta1'])
    labels = data['labels']

    mask_cd = labels == canonical_digit

    # ── Pre-compute expensive UMAPs once ──────────────────────────────────── #
    # Find the Gaussian model entry (if any) and compute its UMAP once so that
    # row 0 (θ₁-coloured) and row 1 (digit-coloured) share the same embedding.
    gauss_umap_cache = {}   # keyed by column index
    for col, (label, model, z) in enumerate(model_entries, start=1):
        if 'theta1' not in z:
            gauss_umap_cache[col] = _umap_2d(np.asarray(z['z']))
    # GT row-1: UMAP of raw pixels (the only honest style baseline for true data).
    gt_pixel_umap = _umap_2d(data['x'])

    # ── Row 0: (θ₁, θ₂) scatter, or UMAP coloured by true θ₁ for Gaussian ── #
    theta_order = _draw_order(len(data['theta1']), seed=42)
    axes[0, 0].scatter(np.asarray(data['theta1'])[theta_order],
                       np.asarray(data['theta2'])[theta_order],
                       c=np.asarray(colors)[theta_order],
                       s=4, alpha=0.4, rasterized=True)
    _set_theta_axes_style(axes[0, 0], hat=False)

    for col, (label, model, z) in enumerate(model_entries, start=1):
        ax = axes[0, col]
        if col in gauss_umap_cache:
            _scatter_coords(ax, z['z'], colors, None,
                            xlabel='UMAP 1', ylabel='UMAP 2',
                            umap_cache=gauss_umap_cache[col])
        else:
            ax.scatter(np.asarray(z['theta1'])[theta_order],
                       np.asarray(z['theta2'])[theta_order],
                       c=np.asarray(colors)[theta_order],
                       s=4, alpha=0.4, rasterized=True)
            _set_theta_axes_style(ax, hat=True)

    # ── Row 1: style-space scatter coloured by digit class ───────────────────  #
    # GT: UMAP of raw pixels — the only honest style baseline for true data.
    # Gaussian VAE: reuse the UMAP from row 0, recoloured by digit.
    # Topology VAEs: UMAP or direct 2-D scatter of z_d.
    _scatter_digits(axes[1, 0], data['x'], labels,
                    umap_cache=gt_pixel_umap, show_legend=True)
    for col, (label, model, z) in enumerate(model_entries, start=1):
        z_style = z.get('z_d', z.get('z'))
        if sliced_style and 'z_d' in z:
            _scatter_digits_sliced(axes[1, col], z_style, labels,
                                   [data['theta1'], data['theta2']])
        else:
            _scatter_digits(axes[1, col], z_style, labels,
                            umap_cache=gauss_umap_cache.get(col))

    # ── Row 2: decoded latent grid ────────────────────────────────────────── #
    gt_grid = _shifted_ground_truth_torus_grid(data, n_grid=_n_grid, pad=_pad)
    _show_image_grid(axes[2, 0], gt_grid,
                     xlabel=r'$\theta_1$', ylabel=r'$\theta_2$')

    # Pre-compute per-cell lookup pools restricted to canonical_digit.
    # Using the nearest encoded (θ₁, θ₂) sample's actual z_d (rather than a
    # fixed centroid or exemplar) keeps every decoded image inside the training
    # manifold of digit canonical_digit.
    for col, (label, model, z) in enumerate(model_entries, start=1):
        ax = axes[2, col]
        if isinstance(model, MixedTorusVAE):
            z_d_pool    = z['z_d'][mask_cd]
            theta1_pool = z['theta1'][mask_cd]
            theta2_pool = z['theta2'][mask_cd]
            grid = _shifted_decode_torus_grid(
                model, n_grid=_n_grid, pad=_pad,
                z_d_pool=z_d_pool,
                theta1_pool=theta1_pool,
                theta2_pool=theta2_pool)
            _show_image_grid(ax, grid,
                             xlabel=r'$z_{\theta_1}$', ylabel=r'$z_{\theta_2}$')
        elif isinstance(model, MNISTGaussianVAE):
            grid = _shifted_decode_gaussian_grid(
                model, z, labels, canonical_digit=canonical_digit,
                n_grid=_n_grid, pad=_pad)
            _show_image_grid(ax, grid, xlabel=r'$z_1$', ylabel=r'$z_2$')

    # ── Row 3: six stacked geodesic strips ────────────────────────────────── #
    gt_mosaic = _shifted_build_geodesic_mosaic_gt(
        data, endpoint_specs, n_frames=_n_frames, pad=_pad)
    _show_image_grid(axes[3, 0], gt_mosaic)

    for col, (label, model, z) in enumerate(model_entries, start=1):
        mosaic = _shifted_build_geodesic_mosaic_model(
            model, z, endpoint_specs, n_frames=_n_frames, pad=_pad)
        _show_image_grid(axes[3, col], mosaic)

    col_titles = ['True data'] + [label for label, _, _ in model_entries]
    label_image_grid(fig, axes, col_titles=col_titles,
                     row_titles=['Latent space', 'Style space', 'Decoded grid', 'Geodesics'])

    if save_path is not None:
        save_figure(save_path)


def plot_shifted_decoded_ring(model, z_codes, data, n_angle=14, save_path=None):
    """Decoded-ring figure for MixedTorusVAE: one row per digit class.

    Sweeps theta1 from -pi to pi while fixing theta2=0 and holding z_d at the
    per-class mean.  The first frame is repeated at the end to show wrap-around.
    """

    labels = data['labels']
    digits = np.sort(np.unique(labels)).tolist()
    n_digits = len(digits)
    n_cols = n_angle + 1
    angles = np.linspace(-np.pi, np.pi, n_angle, endpoint=False)
    angles = np.append(angles, angles[0])   # wrap back to start

    fig, axes = plt.subplots(n_digits, n_cols, figsize=(n_cols * 0.9, n_digits * 1.0))
    if n_digits == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for row, d in enumerate(digits):
            mask = labels == d
            if isinstance(model, MixedTorusVAE) and 'z_d' in z_codes:
                zd_mean = torch.tensor(
                    z_codes['z_d'][mask].mean(axis=0)[None, :], dtype=torch.float32)
            else:
                zd_mean = None

            for j, ang in enumerate(angles):
                t1 = torch.tensor([[ang]], dtype=torch.float32)
                t2 = torch.zeros(1, 1, dtype=torch.float32)
                if not isinstance(model, MixedTorusVAE):
                    break
                img = model.decode(zd_mean, t1, t2)
                axes[row, j].imshow(
                    np.clip(img.squeeze().numpy().reshape(28, 28), 0, 1),
                    cmap='gray', vmin=0, vmax=1)
                axes[row, j].axis('off')
            axes[row, 0].set_ylabel(str(d), rotation=0, labelpad=8, va='center', fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        save_figure(save_path)
