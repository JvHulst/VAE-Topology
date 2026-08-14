r"""
The figure style every figure in the paper is drawn in.

Conventions, enforced throughout:

* Titles: centred, no panel-letter prefix.  Panel letters were dropped in
  favour of row/column references in the captions; ``panel_title`` keeps its
  ``letter`` argument only so call sites did not all have to change.
* Axis labels.  A VAE coordinate is written with a z subscript ($z_\theta$,
  $z_{\theta_1}$, $z_h$, $z_\psi$, $z_s$); the corresponding ground-truth axis
  uses the bare symbol ($\theta$, $h$, ...).  A single symbol takes no
  parentheses.
* Angular axes: values are stored in radians and ticked in degrees.
* Colour: the cyclic HSV rainbow (``angle_to_color``) for periodic coordinates,
  viridis for bounded non-periodic ones, tab10 for a categorical digit class.
* Scatter: ``s=4, alpha=0.4, rasterized=True``.
* Images: ``aspect='equal', interpolation='nearest'`` everywhere.
* Saving: ``save_figure`` writes a 300 dpi PDF and a 150 dpi PNG side by side.

``text.usetex`` is on, so drawing any figure needs a working LaTeX install.

Author: Jilles van Hulst
"""

import os

import matplotlib
matplotlib.use('Agg')          # every figure is written to file, never shown
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


PUBLICATION_RC_PARAMS = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}


def _latex_available() -> bool:
    """Whether matplotlib can actually run LaTeX to typeset a label.

    Checked by rendering rather than by looking for a binary: a `latex` on
    PATH that is missing the packages in the preamble fails just as surely as
    no LaTeX at all, and the failure would otherwise land in the middle of a
    figure the caller has already spent minutes computing.
    """
    from matplotlib.texmanager import TexManager
    try:
        TexManager().make_dvi(r"$z_\theta$", 10)
        return True
    except Exception:
        return False


#: Camera for every 3-D panel.  The defaults are isometric: equal
#: foreshortening along all three axes, under an orthographic projection so
#: parallel edges of the cube stay parallel.
VIEW_ELEV = float(np.degrees(np.arctan(1 / np.sqrt(2))))  # 35.264 deg
VIEW_AZIM = -45.0
PROJ_TYPE = 'ortho'


#: Enlargement of the cube inside its axes box.  mplot3d leaves a wide margin
#: by default, which wastes a large part of every 3-D panel.
BOX_ZOOM_3D = 1.15


#: Numeric tick labels on 3-D panels.  Off by default: three axes of numbers
#: around a small cube crowd each other, and at the scale the manuscript prints
#: these panels the numbers arrive at roughly 3 pt, which is decorative rather
#: than readable.  The tick marks stay either way.
SHOW_3D_TICK_LABELS = False


_STYLE_APPLIED = False


def apply_publication_style(force_usetex: bool | None = None) -> None:
    """Apply the shared publication figure style.

    The paper's figures are typeset with LaTeX.  A fresh clone without a LaTeX
    install falls back to matplotlib's own mathtext, which renders every label
    in these figures acceptably — the maths is all subscripts and Greek — but
    not identically, so figures for the paper must be drawn with LaTeX present.

    Args:
        force_usetex: skip the probe and demand (True) or refuse (False) LaTeX.
    """
    global _STYLE_APPLIED
    params = dict(PUBLICATION_RC_PARAMS)

    usetex = force_usetex
    if usetex is None:
        usetex = _latex_available()
        if not usetex and not _STYLE_APPLIED:
            print("topovae: no working LaTeX install found, falling back to "
                  "mathtext. Figures will render but will not match the paper.")

    if not usetex:
        params["text.usetex"] = False
        params["font.family"] = "serif"
        params["font.serif"] = ["DejaVu Serif"]
        params["mathtext.fontset"] = "dejavuserif"

    rcParams.update(params)
    _STYLE_APPLIED = True


def panel_title(letter: str, title: str) -> str:
    """Return the panel title string.

    Figures identify panels by row and column in their captions rather than by
    letter, so ``letter`` is accepted to keep one call signature across panel
    builders and does not appear in the rendered title.
    """
    return title


apply_publication_style()


# =============================================================================
# PRINT SCALE
# =============================================================================
#
# Every figure here is drawn many inches wide, at whatever size gives its
# panels room to work with, then shrunk onto the page by `\includegraphics`.
# That shrink applies to every native length in the figure equally — so a
# 12 pt label meant for an 18 in canvas prints at roughly 12 * (7.14/18),
# under 5 pt.  `scale_fonts_to_page` sets the native font sizes so that after
# that same shrink, text prints at the PRINT_*_PT targets below.


#: Full text width of a page in this paper: IEEEtran, journal mode, US letter
#: paper.  Every synthetic and MNIST figure is placed at (approximately) this
#: width in a double-column `figure*`, via `\includegraphics[width=\textwidth]`
#: or an equivalent `scale=`.  Set in IEEEtran.cls as `\textwidth 43pc`; a pica
#: is 12 pt and a (TeX) point is 1/72.27 in.
PAGE_TEXT_WIDTH_IN = 43 * 12 / 72.27

#: Point sizes as they should read *on the printed page*.
PRINT_TITLE_PT = 10.0
PRINT_LABEL_PT = 9.0
PRINT_TICK_PT = 8.0
PRINT_LEGEND_PT = 8.0
PRINT_TITLE_PAD_PT = 4.0
PRINT_LABEL_PAD_PT = 3.0

#: Shrink factor of the figure most recently sized by `scale_fonts_to_page`,
#: i.e. `PAGE_TEXT_WIDTH_IN / nominal_width_in`.  Kept around for `label3d_pad`,
#: below: mplot3d's axis-label offset is fixed in native points and so does
#: not follow the rcParams font rescale that everything else here gets.
CURRENT_SHRINK = 1.0


def scale_fonts_to_page(nominal_width_in, *, title_pt=PRINT_TITLE_PT):
    """Set native font sizes (and paddings) so they print at the PRINT_*_PT targets.

    Call once per figure, before building it, with that figure's own known
    nominal width in inches (its `figsize` width, or close to it). Padding
    (title pad, label pad) is rescaled the same way as the fonts themselves,
    so that whitespace stays in proportion to the text once both are shrunk.

    Returns the shrink factor, for callers that also need to size a 3-D label
    pad (`label3d_pad`) or an absolute-inch margin for the same figure.
    """
    global CURRENT_SHRINK
    CURRENT_SHRINK = PAGE_TEXT_WIDTH_IN / nominal_width_in
    plt.rcParams.update({
        "axes.labelsize": PRINT_LABEL_PT / CURRENT_SHRINK,
        "axes.titlesize": title_pt / CURRENT_SHRINK,
        "xtick.labelsize": PRINT_TICK_PT / CURRENT_SHRINK,
        "ytick.labelsize": PRINT_TICK_PT / CURRENT_SHRINK,
        "legend.fontsize": PRINT_LEGEND_PT / CURRENT_SHRINK,
        "axes.titlepad": PRINT_TITLE_PAD_PT / CURRENT_SHRINK,
        "axes.labelpad": PRINT_LABEL_PAD_PT / CURRENT_SHRINK,
    })
    return CURRENT_SHRINK


#: mplot3d's fixed part of the label offset (`default_offset` in
#: `axis3d.Axis.draw`): a 3-D axis label is always placed
#: `(labelpad + MPL3D_DEFAULT_OFFSET)` points from its axis, regardless of
#: rcParams.  Even `labelpad=0` therefore prints conspicuously far from the
#: cube once the figure is shrunk onto the page — the 21 pt constant does not
#: shrink with everything else, since it is not a font size.
MPL3D_DEFAULT_OFFSET = 21.0

#: Target gap between a 3-D axis and its label, in points on the printed page.
LABEL_3D_GAP_PT = 4.0


def label3d_pad(gap_pt=LABEL_3D_GAP_PT):
    """The `labelpad` that leaves `gap_pt` between a 3-D axis and its label.

    Solves `(labelpad + MPL3D_DEFAULT_OFFSET) * CURRENT_SHRINK = gap_pt` for
    `labelpad` — negative enough to cancel most of mplot3d's fixed offset.
    Reads `CURRENT_SHRINK` from the most recent `scale_fonts_to_page` call, so
    that call must come first.
    """
    return gap_pt / CURRENT_SHRINK - MPL3D_DEFAULT_OFFSET


# =============================================================================
# COLOUR
# =============================================================================


#: Cyclic colormap for angular coordinates.  'hsv' is the rainbow used
#: throughout the manuscript; 'twilight' is perceptually uniform and reads
#: better in greyscale, at the cost of a light band through its middle that
#: makes anchor markers harder to pick out.
CYCLIC_CMAP = 'hsv'


def angle_to_color(angles):
    """Convert angles in (-pi, pi] to colors on the cyclic colormap."""
    normalized = (np.asarray(angles) + np.pi) / (2 * np.pi)
    return plt.get_cmap(CYCLIC_CMAP)(normalized)


# =============================================================================
# SAVING
# =============================================================================


def save_figure(save_path, pad_inches=0.2):
    """
    Save figure as PDF and PNG with consistent settings.

    Args:
        save_path: Path to save the PDF file (PNG is saved alongside).
        pad_inches: Tight-layout padding in inches (default: 0.2).

    Note: bbox_inches='tight' is used instead of tight_layout() because
    tight_layout does not account for 3-D axis labels correctly.
    """
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.splitext(save_path)[0] + '.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=pad_inches)
    plt.savefig(png_path,   dpi=150, bbox_inches='tight', pad_inches=pad_inches)
    print(f"Saved figure to {save_path}")
    plt.close()
