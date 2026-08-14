"""
Panel primitives shared by the figures.

Nothing here knows which experiment it is drawing.  The per-experiment figure
functions live in ``synthetic/plots.py`` and ``mnist/plots.py`` and are built
out of these.

``make_image_aware_figure`` is the one worth knowing about: a figure whose rows
mix scatter panels with strips of 28×28 images cannot use ``tight_layout``,
because an image panel's height is dictated by its pixel aspect ratio rather
than by the space available.  It lays out rows at explicit heights with uniform
gaps instead, and ``image_strip_height`` / ``image_grid_height`` compute the
height an image row needs at a given column width.

Author: Jilles van Hulst
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

from ._style import (
    BOX_ZOOM_3D,
    PROJ_TYPE,
    SHOW_3D_TICK_LABELS,
    VIEW_AZIM,
    VIEW_ELEV,
    label3d_pad,
    panel_title,
)


# =============================================================================
# FIGURE LAYOUT
# =============================================================================


def finalize_synthetic_row(fig, *, left=0.02, right=0.98, bottom=0.08,
                           top=0.90, wspace=0.40, shift_3d=-0.08):
    """Lay out a single row of synthetic panels and clear the 3-D/2-D label clash.

    A 3-D panel's z-axis label protrudes to the right of its cube, into the gap
    where the next panel's y-axis label sits, so the two collide.  Nudging every
    3-D cube left within its own cell pulls its z label back off the neighbour.

    The panel titles are detached first and redrawn at the fixed cell centres, so
    they stay aligned across the row no matter how far the cubes underneath them
    move — and so that a 3-D title and a 2-D title sit at the same height.

    Args:
        fig: The figure holding the row of panels.
        left, right, bottom, top: Outer margins passed to ``subplots_adjust``.
        wspace: Gap between panels, as a fraction of panel width.
        shift_3d: Sideways nudge of each 3-D panel, as a fraction of its width
            (negative moves it left).
    """
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace)
    axes = fig.get_axes()

    # Record each title at its cell centre, then clear it from the axes so the
    # nudge below cannot drag it out of column.
    titles = []
    for ax in axes:
        pos = ax.get_position()
        titles.append((ax.get_title(), pos.x0 + pos.width / 2, pos.y1))
        ax.set_title('')

    for ax in axes:
        if hasattr(ax, 'zaxis'):
            pos = ax.get_position()
            ax.set_position([pos.x0 + shift_3d * pos.width, pos.y0,
                             pos.width, pos.height])

    size = plt.rcParams['axes.titlesize']
    pad = plt.rcParams['axes.titlepad'] / 72.0 / fig.get_figheight()
    for text, x, y in titles:
        if text:
            fig.text(x, y + pad, text, ha='center', va='bottom', fontsize=size)


def make_image_aware_figure(
    row_heights_in,
    n_cols,
    col_width_in,
    gap_row_in=0.30,
    gap_col_in=0.12,
    left_in=0.55,
    right_in=0.10,
    top_in=0.35,
    bottom_in=0.50,
):
    """Create a figure whose axes have precisely specified content heights.

    Unlike ``plt.subplots`` with ``height_ratios`` + ``hspace``, this helper
    positions every axes cell in absolute inches so that:

    * each row i has content height ``row_heights_in[i]``,
    * every vertical gap between adjacent rows is exactly ``gap_row_in``,
    * every horizontal gap between adjacent columns is exactly ``gap_col_in``.

    Image pixels in any row are never stretched: if you pass ``aspect='equal'``
    to ``ax.imshow`` the image fills its cell exactly.

    Parameters
    ----------
    row_heights_in : list[float]
        Content height of each row in inches.  Row 0 is the topmost row.
    n_cols : int
        Number of columns.
    col_width_in : float
        Content width of each column in inches.
    gap_row_in : float
        Vertical gap between the bottom of one row's axes box and the top of
        the next, in inches.
    gap_col_in : float
        Horizontal gap between adjacent column axes boxes, in inches.
    left_in, right_in, top_in, bottom_in : float
        Outer margins (in inches) around the grid.  ``left_in`` should be wide
        enough for y-tick labels; ``bottom_in`` for x-axis labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray, shape ``(n_rows, n_cols)``, dtype object
        Row 0 is the topmost row.
    """
    n_rows = len(row_heights_in)

    fig_w = left_in + n_cols * col_width_in + (n_cols - 1) * gap_col_in + right_in
    fig_h = top_in + sum(row_heights_in) + (n_rows - 1) * gap_row_in + bottom_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = np.empty((n_rows, n_cols), dtype=object)

    # Top edge of each row in inches from the figure bottom.
    y_top = fig_h - top_in
    row_tops = []
    for h in row_heights_in:
        row_tops.append(y_top)
        y_top -= h + gap_row_in

    for i in range(n_rows):
        for j in range(n_cols):
            x_left = left_in + j * (col_width_in + gap_col_in)
            b = (row_tops[i] - row_heights_in[i]) / fig_h
            axes[i, j] = fig.add_axes([
                x_left / fig_w,
                b,
                col_width_in / fig_w,
                row_heights_in[i] / fig_h,
            ])

    return fig, axes


def image_strip_height(col_width_in, n_frames, img_h=28, img_w=28, pad=2):
    """Height in inches of a horizontal strip of ``n_frames`` images.

    Each image occupies ``(img_w + pad)`` pixels horizontally (no trailing pad
    on the last image) and ``img_h`` pixels vertically.  Given a column width
    of ``col_width_in`` inches the strip height in inches is::

        col_width_in * img_h / (n_frames * (img_w + pad) - pad)
    """
    strip_w_px = n_frames * (img_w + pad) - pad
    return col_width_in * img_h / strip_w_px


def image_grid_height(col_width_in, n_rows_img, n_cols_img, img_h=28, img_w=28, pad=2):
    """Height in inches of an ``n_rows_img`` × ``n_cols_img`` decoded image grid."""
    grid_w_px = n_cols_img * (img_w + pad) - pad
    grid_h_px = n_rows_img * (img_h + pad) - pad
    return col_width_in * grid_h_px / grid_w_px


# =============================================================================
# SCATTER PANELS
# =============================================================================


def _plot_3d_panel(ax, x, y, z, colors, xlabel, ylabel, zlabel, title, aspect=[1, 1, 1], 
                   anchor_indices=None):
    """
    Plot a 3D scatter panel with consistent styling and optional anchor points.
    
    Args:
        ax: Matplotlib 3D axes
        x, y, z: Data coordinates
        colors: Color array for each point
        xlabel, ylabel, zlabel: Axis labels
        title: Panel title
        aspect: Box aspect ratio [x, y, z]
        anchor_indices: Optional indices of anchor points to highlight
    """
    ax.scatter(x, y, z, c=colors, s=4, alpha=0.4, rasterized=True)
    
    if anchor_indices is not None:
        anchor_colors = colors[anchor_indices]
        ax.scatter(x[anchor_indices], y[anchor_indices], z[anchor_indices],
                   c=anchor_colors, s=100, marker='*',
                   edgecolors='black', linewidths=1.0, zorder=10)
    
    # mplot3d places a label at (labelpad + 21 pt) from its axis regardless of
    # rcParams, so a small positive labelpad still prints far from the cube
    # once the figure is shrunk onto the page; label3d_pad() compensates.
    pad = label3d_pad()
    ax.set_xlabel(xlabel, labelpad=pad)
    ax.set_ylabel(ylabel, labelpad=pad)
    # mplot3d rotates an axis label to follow its projected axis, which fires
    # for a two-glyph label like $z_h$ but not for a single glyph like $h$.
    # Pin the height label upright so it reads the same in every panel.
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlabel, labelpad=pad, rotation=0)
    ax.set_title(title)

    # Clean 3D axis styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))

    if not SHOW_3D_TICK_LABELS:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Claw back mplot3d's generous margin by enlarging the cube inside its axes box.
    ax.set_box_aspect(aspect, zoom=BOX_ZOOM_3D)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_proj_type(PROJ_TYPE)


def _plot_2d_panel(ax, x, y, colors, xlabel, ylabel, title, xlim=None, ylim=None,
                   anchor_indices=None):
    """
    Plot a 2D scatter panel with consistent styling and optional anchor points.
    
    Args:
        ax: Matplotlib axes
        x, y: Data coordinates
        colors: Color array for each point
        xlabel, ylabel: Axis labels
        title: Panel title
        xlim, ylim: Optional axis limits (tuples)
        anchor_indices: Optional indices of anchor points to highlight
    """
    ax.scatter(x, y, c=colors, s=4, alpha=0.4, rasterized=True)
    
    if anchor_indices is not None:
        anchor_colors = colors[anchor_indices]
        ax.scatter(x[anchor_indices], y[anchor_indices],
                   c=anchor_colors, s=100, marker='*',
                   edgecolors='black', linewidths=1.0, zorder=10)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')
    
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_gaussian_vae_panel(ax, z_gaussian, colors, title_letter='b'):
    """
    Plot standard 2D Gaussian VAE panel (used in all experiments).
    
    Args:
        ax: Matplotlib axes
        z_gaussian: Dict with 'z' key containing 2D latent codes
        colors: Color array for each point
        title_letter: Panel label letter (default: 'b')
    """
    z = z_gaussian['z']
    _plot_2d_panel(ax, z[:, 0], z[:, 1], colors,
                   xlabel=r'$z_1$', ylabel=r'$z_2$',
                   title=panel_title(title_letter, 'Gaussian VAE'),
                   xlim=(-3, 3), ylim=(-3, 3))


def _plot_annulus_boundaries(ax, r_min, r_max):
    """
    Plot inner and outer boundary circles for annulus plots.
    
    Args:
        ax: Matplotlib axes
        r_min: Inner radius
        r_max: Outer radius
    """
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(r_min * np.cos(theta_circle), r_min * np.sin(theta_circle), 
            'k--', linewidth=1, alpha=0.5)
    ax.plot(r_max * np.cos(theta_circle), r_max * np.sin(theta_circle), 
            'k--', linewidth=1, alpha=0.5)


def _add_repel_panel(fig, n_panels, panel_idx, z_gaussian_repel, colors):
    """Add the Gaussian VAE + repelling anchor panel (shared by cylinder & annulus)."""
    ax = fig.add_subplot(1, n_panels, panel_idx)
    z = z_gaussian_repel['z']
    _plot_2d_panel(ax, z[:, 0], z[:, 1], colors,
                   xlabel=r'$z_1$', ylabel=r'$z_2$',
                   title='Gaussian VAE + repel',
                   xlim=(-3, 3), ylim=(-3, 3))
    ax.scatter([0], [0], c='red', s=100, marker='x', linewidths=2, zorder=10)


def _scatter_coords(ax, coords, colors, title, xlabel, ylabel,
                    umap_cache=None, n_sub=3000, seed=42, **scatter_kw):
    """Scatter ``coords`` on ``ax``, falling back to UMAP when ``coords`` is >2-D.

    Parameters
    ----------
    coords : array [N, D]
        The coordinates to plot.  If D == 2 they are used directly; otherwise
        a 2-D UMAP is computed (or ``umap_cache`` is reused).
    colors : array [N] or [N, 4]
        Per-point colour values passed to ``ax.scatter``.
    title : str or None
        Panel title.  Pass ``None`` when the panel's purpose is already given
        by a shared column header / row label (see ``label_image_grid``).
    umap_cache : (emb, idx) or None
        Pre-computed UMAP result from ``_umap_2d``.  Reuse this to avoid
        re-fitting when the same high-D space is plotted with different colours.
    **scatter_kw
        Extra keyword arguments forwarded to ``ax.scatter``.
    """
    coords = np.asarray(coords)
    scatter_defaults = dict(s=4, alpha=0.4, rasterized=True)
    scatter_defaults.update(scatter_kw)

    if coords.ndim == 2 and coords.shape[1] == 2:
        emb, idx = coords, np.arange(len(coords))
        x_label, y_label = xlabel, ylabel
    else:
        if umap_cache is not None:
            emb, idx = umap_cache
        else:
            emb, idx = _umap_2d(coords, n_sub=n_sub, seed=seed)
        x_label, y_label = 'UMAP 1', 'UMAP 2'

    c = colors[idx] if np.asarray(colors).ndim >= 1 and len(np.asarray(colors)) == len(coords) else colors
    order = _draw_order(len(emb), seed)
    ax.scatter(emb[order, 0], emb[order, 1],
               c=c[order] if np.ndim(c) >= 1 and len(np.asarray(c)) == len(emb) else c,
               **scatter_defaults)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def _draw_order(n, seed):
    """A shuffled plotting order for ``n`` points.

    The image datasets are laid out one digit class after another, so scattering
    them in array order paints the last class on top of every earlier one and
    hides whatever lies beneath it.  Drawing in a random order interleaves the
    classes, so overlapping regions show the mixture that is really there.  The
    order is seeded, so a re-render reproduces the same figure.
    """
    return np.random.default_rng(seed).permutation(n)


def _digit_rank_map(digit_labels):
    """Map each digit class to its tab10 colour index.

    The digits present are sorted and take tab10 slots 0, 1, 2, ... in turn, so
    the same digit gets the same colour in the scatter, the legend, and the
    sliced quadrants, whichever digits a run happens to use.
    """
    unique = np.sort(np.unique(digit_labels))
    return {int(d): i % plt.cm.tab10.N for i, d in enumerate(unique)}


def _scatter_digits(ax, coords, digit_labels, title=None,
                    umap_cache=None, n_sub=3000, seed=42,
                    show_legend=False):
    """Scatter ``coords`` coloured by digit class (tab10).

    The digits present are sorted and take tab10 colours 0, 1, 2, ... in turn,
    so every run opens on blue, orange, green whichever digits it happens to
    use, and the scatter and legend cannot disagree.  Falls back to UMAP when
    ``coords`` is >2-D, or reuses ``umap_cache`` when provided.

    ``title``, if given, is set on the panel directly (no row-label prefix):
    the row's purpose is expected to already be given by a shared row label,
    see ``label_image_grid``.
    """

    digit_to_rank = _digit_rank_map(digit_labels)
    unique_digits = np.sort(np.unique(digit_labels))
    rank_labels   = np.array([digit_to_rank[int(d)] for d in digit_labels])
    cmap = plt.cm.tab10

    coords = np.asarray(coords)
    if coords.ndim == 2 and coords.shape[1] == 2:
        emb, idx = coords, np.arange(len(coords))
        xlabel, ylabel = r'$z_1$', r'$z_2$'
    else:
        if umap_cache is not None:
            emb, idx = umap_cache
        else:
            emb, idx = _umap_2d(coords, n_sub=n_sub, seed=seed)
        xlabel, ylabel = 'UMAP 1', 'UMAP 2'

    order = _draw_order(len(emb), seed)
    ax.scatter(emb[order, 0], emb[order, 1], c=cmap(rank_labels[idx][order]),
               s=4, alpha=0.4, rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    if show_legend:
        legend_elements = [
            Patch(facecolor=cmap(digit_to_rank[int(d)]), label=str(int(d)))
            for d in unique_digits
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  framealpha=0.9, ncol=1)


# =============================================================================
# IMAGE PANELS
# =============================================================================


def _show_image_grid(ax, mosaic, xlabel='', ylabel='', title=''):
    """Display a float32 grayscale mosaic with minimal axis decoration."""
    ax.imshow(mosaic, cmap='gray', vmin=0, vmax=1,
              aspect='equal', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def label_image_grid(fig, axes, col_titles, row_titles,
                     header_pad_in=0.06, row_label_x_in=0.28):
    """Label a grid of panels with one header per column, one label per row.

    The column name goes in a single header above each column and the row's
    purpose in a single rotated label at the left, rather than a title on every
    panel.  At a legible font size, repeating the column name on every row just
    burns space, and the row's purpose reads better as one label than as a
    prefix on four panels.

    The row labels sit at a fixed distance from the figure's left edge, so they
    line up in a column near the edge; the panels' own y-axis labels sit to
    their right.  The figure's ``left_in`` must be wide enough to hold the row
    labels, the y-axis labels, and a gap between them (``left_in`` around 0.95 in
    at the sizes here).

    Args:
        fig: The figure the axes belong to.
        axes: The (n_rows, n_cols) axes array from ``make_image_aware_figure``.
        col_titles: One heading per column, left to right.
        row_titles: One label per row, top to bottom.
        header_pad_in: Gap, in inches, between the top row and its headers.
        row_label_x_in: Distance, in inches, from the figure's left edge to the
            centre of the rotated row labels.
    """
    fig_w, fig_h = fig.get_size_inches()
    size = plt.rcParams['axes.titlesize']

    for ax, text in zip(axes[0, :], col_titles):
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, pos.y1 + header_pad_in / fig_h,
                 text, ha='center', va='bottom', fontsize=size)

    for ax, text in zip(axes[:, 0], row_titles):
        pos = ax.get_position()
        fig.text(row_label_x_in / fig_w, pos.y0 + pos.height / 2,
                 text, ha='center', va='center', rotation=90, fontsize=size)


# =============================================================================
# ANGLES AND EMBEDDINGS
# =============================================================================


def _set_theta_axes_style(ax, hat=False):
    """Style a (theta1, theta2) scatter: equal aspect, no ticks, z-subscript labels.

    The axes span one full period in each angle and carry no tick labels, so the
    panel reads as a torus chart rather than a numeric plot.
    """
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    if hat:
        ax.set_xlabel(r'$z_{\theta_1}$')
        ax.set_ylabel(r'$z_{\theta_2}$')
    else:
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')


def _interp_angle(a, b, n_frames):
    a = float(a)
    b = float(b)
    delta = np.arctan2(np.sin(b - a), np.cos(b - a))
    t = np.linspace(0.0, 1.0, n_frames)
    return a + t * delta


def _umap_2d(z, n_sub=3000, seed=42):
    """Fit a 2-D UMAP on (a subsample of) ``z``.

    Returns ``(emb, idx)`` where ``emb`` is the N'×2 embedding and ``idx``
    is the array of original row indices that were embedded.  Call once and
    pass the result to ``_scatter_coords`` or ``_scatter_digits`` to reuse
    the same embedding with different colour arrays.
    """
    from umap import UMAP
    rng = np.random.RandomState(seed)
    n   = len(z)
    idx = rng.choice(n, min(n_sub, n), replace=False) if n > n_sub else np.arange(n)
    emb = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
               random_state=seed).fit_transform(z[idx])
    return emb, idx
