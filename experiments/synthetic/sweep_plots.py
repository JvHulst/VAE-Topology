"""
The KL-sweep figure: test reconstruction error against unweighted KL divergence.

    python -m experiments torus --render results/kl_sweep/<run>

Both axes decrease toward the lower left, so a curve lying below and to the left
is better. Each point is the median over the sweep's seeds at one beta, with the
individual seeds drawn faintly behind it.

This is separate from plots.py so that the single-run figure code carries none of
the sweep's aggregation.

Author: Jilles van Hulst
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from topovae.evaluate import get_figures_dir, load_run_info
from experiments._style import apply_publication_style, save_figure

apply_publication_style()


# =============================================================================
# BETA SWEEP
# =============================================================================


def _model_color(model: str) -> str:
    mapping = {
        "Gaussian": "tab:blue",
        "Cylinder": "tab:orange",
        "Cylinder+Anchor": "tab:green",
        "Torus": "tab:orange",
        "Torus+Anchor": "tab:green",
        "Mobius": "tab:orange",
        "Mobius+Anchor": "tab:green",
        "Möbius": "tab:orange",
        "Möbius+Anchor": "tab:green",
    }
    return mapping.get(model, "tab:gray")


#: Marker and dash pattern per model, so the three curves in a panel stay
#: separable when the figure is printed in greyscale or read by someone with a
#: colour vision deficiency.  Colour alone does not carry that.
_MODEL_STYLES = {
    "Gaussian": ("o", "-"),
    "Cylinder": ("^", "--"),
    "Torus": ("^", "--"),
    "Mobius": ("^", "--"),
    "Möbius": ("^", "--"),
    "Cylinder+Anchor": ("D", "-."),
    "Torus+Anchor": ("D", "-."),
    "Mobius+Anchor": ("D", "-."),
    "Möbius+Anchor": ("D", "-."),
}


def _model_style(model: str) -> tuple:
    """Return the ``(marker, linestyle)`` pair for a model key."""
    return _MODEL_STYLES.get(model, ("o", "-"))


def _aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["experiment", "model", "beta"]
    for (experiment, model, beta), g in df.groupby(group_cols):
        rows.append(
            {
                "experiment": experiment,
                "model": model,
                "beta": beta,
                "n": int(len(g)),
                "kl_median": float(np.median(g["kl_divergence"])),
                "kl_q25": float(np.percentile(g["kl_divergence"], 25)),
                "kl_q75": float(np.percentile(g["kl_divergence"], 75)),
                "test_rms_median": float(np.median(g["test_rms"])),
                "test_rms_q25": float(np.percentile(g["test_rms"], 25)),
                "test_rms_q75": float(np.percentile(g["test_rms"], 75)),
            }
        )
    return pd.DataFrame(rows).sort_values(["experiment", "model", "beta"]).reset_index(drop=True)


#: Width of one panel on the printed page.  Each topology is its own figure,
#: included at this width, so the fonts are sized against it directly.
PANEL_WIDTH_MM = 100.0

#: Height/width of a panel.  The tradeoff curve is a shallow L, so it reads
#: better wide than tall, and a short panel leaves room for all three figures in
#: a column.
PANEL_ASPECT = 0.52

#: Font sizes in points, as they appear on the printed page.  The figure is
#: drawn at its final size and included without a width override, so these are
#: literal: nothing rescales them afterwards.
FONT_LABEL_PT = 10.0
FONT_TICK_PT = 9.0


def plot_kl_vs_reconstruction(
    df: pd.DataFrame,
    save_path: str,
    xscale: str = "linear",
    yscale: str = "linear",
    plot_error_bars: bool = True,
    plot_individual_points: bool = True,
    shared_x_axis: bool = True,
    panel_width_mm: float = PANEL_WIDTH_MM,
):
    """Draw the reconstruction/KL tradeoff, one figure per topology.

    Each topology is written to its own file, named after ``save_path`` with the
    experiment key appended: ``fig_kl_sweep.pdf`` gives ``fig_kl_sweep_cylinder.pdf``
    and so on.  The panels are drawn at their final printed size, so no
    ``\\includegraphics`` shrink intervenes and the fonts land where they are set.

    With ``shared_x_axis`` the KL range is computed across every topology and
    applied to all of them, so curves stay comparable between the separate
    figures.  Which topology a figure shows, and which curve is which model, are
    stated in the caption; a title and legend at this size would cost data area.

    Returns the list of paths written.
    """
    experiment_order = ["cylinder", "torus", "mobius"]
    experiments = [exp for exp in experiment_order if exp in df["experiment"].unique()]

    if not experiments:
        raise ValueError("No recognized experiments found in metrics.csv")

    fig_w = panel_width_mm / 25.4
    fig_h = fig_w * PANEL_ASPECT

    # Sizes apply as written: the figure is drawn at the width it is printed at,
    # so it takes no `\includegraphics` shrink for the fonts to compensate for.
    plt.rcParams.update({
        "axes.labelsize": FONT_LABEL_PT,
        "xtick.labelsize": FONT_TICK_PT,
        "ytick.labelsize": FONT_TICK_PT,
        "axes.labelpad": 2.0,
    })

    # One shared KL range across the separate figures, from the same spread of
    # points that gets drawn.  Without this each figure would autoscale to its
    # own topology and the panels could not be read against each other.
    if shared_x_axis:
        span = df["kl_divergence"]
        margin = 0.04 * (span.max() - span.min())
        xlim = (span.min() - margin, span.max() + margin)

    stem, ext = os.path.splitext(save_path)
    written = []

    for exp_name in experiments:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        exp_df = df[df["experiment"] == exp_name].copy()

        for model in exp_df["model"].unique():
            model_df = exp_df[exp_df["model"] == model]
            color = _model_color(model)
            marker, linestyle = _model_style(model)

            if plot_individual_points:
                ax.scatter(
                    model_df["kl_divergence"],
                    model_df["test_rms"],
                    color=color,
                    marker=marker,
                    s=10,
                    alpha=0.15,
                    linewidths=0,
                    zorder=2,
                    rasterized=True,
                )

            beta_values = sorted(model_df["beta"].unique())
            kl_med, kl_lo, kl_hi = [], [], []
            rec_med, rec_lo, rec_hi = [], [], []

            for beta in beta_values:
                g = model_df[model_df["beta"] == beta]
                kl_vals = g["kl_divergence"].values
                rec_vals = g["test_rms"].values

                kl_med.append(np.median(kl_vals))
                kl_lo.append(np.percentile(kl_vals, 25))
                kl_hi.append(np.percentile(kl_vals, 75))

                rec_med.append(np.median(rec_vals))
                rec_lo.append(np.percentile(rec_vals, 25))
                rec_hi.append(np.percentile(rec_vals, 75))

            kl_med = np.array(kl_med)
            kl_lo = np.array(kl_lo)
            kl_hi = np.array(kl_hi)
            rec_med = np.array(rec_med)
            rec_lo = np.array(rec_lo)
            rec_hi = np.array(rec_hi)

            order = np.argsort(kl_med)
            kl_med = kl_med[order]
            kl_lo = kl_lo[order]
            kl_hi = kl_hi[order]
            rec_med = rec_med[order]
            rec_lo = rec_lo[order]
            rec_hi = rec_hi[order]

            ax.plot(
                kl_med,
                rec_med,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                zorder=4,
                marker=marker,
                markersize=5,
                markerfacecolor=color,
            )

            if plot_error_bars:
                ax.errorbar(
                    kl_med,
                    rec_med,
                    xerr=[kl_med - kl_lo, kl_hi - kl_med],
                    yerr=[rec_med - rec_lo, rec_hi - rec_med],
                    fmt=marker,
                    color=color,
                    markersize=5,
                    elinewidth=1.2,
                    capsize=3,
                    capthick=1.2,
                    zorder=5,
                )

        ax.set_xlabel(r"Unweighted KL divergence")
        ax.set_ylabel("Test RMS error")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if shared_x_axis:
            ax.set_xlim(*xlim)

        if yscale == "log":
            # The reconstruction errors span well under a decade, where the
            # default log ticks would label a single power of ten.  Label the
            # 1-2-3-5-7 steps instead, and write them as plain numbers.
            ax.yaxis.set_major_locator(
                LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=12))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, alpha=0.3, which="both")

        exp_path = f"{stem}_{exp_name}{ext}"
        fig.tight_layout(pad=0.2)
        # save_figure closes the figure, so nothing is left open between panels.
        save_figure(exp_path, pad_inches=0.02)
        written.append(exp_path)

    return written

# =============================================================================
# RENDER
# =============================================================================


def render_run(
    run_dir: str,
    xscale: str = "linear",
    yscale: str = "log",
    plot_error_bars: bool = False,
    plot_individual_points: bool = True,
    shared_x_axis: bool = True,
    panel_width_mm: float = PANEL_WIDTH_MM,
) -> None:
    _ = load_run_info(run_dir)  # Validate run.json exists and is readable

    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Expected metrics.csv at {metrics_path}")

    df = pd.read_csv(metrics_path)
    required_cols = {"experiment", "model", "beta", "seed", "kl_divergence", "test_rms"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metrics.csv is missing required columns: {sorted(missing)}")

    summary = _aggregate_summary(df)
    summary.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    fig_dir = get_figures_dir(run_dir)
    plot_kl_vs_reconstruction(
        df=df,
        save_path=os.path.join(fig_dir, "fig_kl_sweep.pdf"),
        xscale=xscale,
        yscale=yscale,
        plot_error_bars=plot_error_bars,
        plot_individual_points=plot_individual_points,
        shared_x_axis=shared_x_axis,
        panel_width_mm=panel_width_mm,
    )


