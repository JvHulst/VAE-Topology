"""
One command for every experiment in the paper.

    python -m experiments torus                     # train it, then draw its figures
    python -m experiments torus --seeds 30          # repeat it, report the spread
    python -m experiments torus --betas 0.05,10.0   # sweep beta, draw the trade-off
    python -m experiments torus --render results/torus/<run>

Pick an experiment, then choose how many times to run it. With neither `--seeds`
nor `--betas` it trains once at the operating point in the experiment's spec and
saves a full run directory. `--seeds` repeats that at fixed beta and summarises
the spread across model initialisations. `--betas` walks the reconstruction /
regularisation trade-off instead, training many cheap models and keeping only
their metrics.

`--render` skips training and redraws a saved run. A run directory says in its
own run.json which kind of run it is, so the same flag works for all of them.

The experiments themselves live one directory down: `synthetic/experiment.py` for
the manifolds with known ground truth, `mnist/experiment.py` for the images. Each
holds a SPECS table with one row per experiment, and that row is the whole
description of what makes it that experiment.

Author: Jilles van Hulst
"""

import argparse

from topovae.evaluate import load_run_info

from experiments import _seeds, _sweep
from experiments.mnist import experiment as image
from experiments.synthetic import experiment as synthetic
from experiments.synthetic import sweep_plots

#: Every family of experiments, searched in order to resolve an experiment name.
FAMILIES = (synthetic, image)


def _family(name):
    """The family module that owns an experiment name."""
    for family in FAMILIES:
        if name in family.SPECS:
            return family
    raise ValueError(f"Unknown experiment: {name}")


def _experiment_names():
    return sorted(name for family in FAMILIES for name in family.SPECS)


def _parse_betas(text):
    betas = [float(x) for x in text.split(",") if x.strip()]
    if not betas:
        raise ValueError("--betas needs at least one value")
    return betas


def main():
    parser = argparse.ArgumentParser(
        prog="python -m experiments",
        description="Train, repeat, sweep or redraw one of the paper's experiments.")
    parser.add_argument("experiment", choices=_experiment_names())
    parser.add_argument("--seeds", type=int, metavar="N",
                        help="repeat over N model initialisations and summarise the spread")
    parser.add_argument("--betas", metavar="LIST",
                        help="comma-separated beta values to sweep, e.g. 0.05,0.2,1.0,5.0")
    parser.add_argument("--render", metavar="RUN_DIR",
                        help="redraw a saved run instead of training")
    parser.add_argument("--recompute-metrics", action="store_true",
                        help="with --render, also recompute the metrics table")
    parser.add_argument("--run-name", default=None,
                        help="name the run directory; reuse a name to resume a sweep")
    parser.add_argument("--base-seed", type=int, default=1000,
                        help="first seed used by --seeds and --betas")
    args = parser.parse_args()

    family = _family(args.experiment)

    if args.render:
        if load_run_info(args.render).get("experiment") == "kl_sweep":
            sweep_plots.render_run(args.render)
        else:
            family.render(args.experiment, args.render,
                          recompute_metrics=args.recompute_metrics)

    elif args.betas:
        if family is not synthetic:
            parser.error("--betas is only available for the synthetic experiments; "
                         "the image experiments are too expensive to sweep")
        _sweep.run(args.experiment, betas=_parse_betas(args.betas),
                   n_seeds=args.seeds or _sweep.DEFAULT_N_SEEDS,
                   base_seed=args.base_seed, run_name=args.run_name)

    elif args.seeds:
        _seeds.run(family, args.experiment, n_seeds=args.seeds,
                   base_seed=args.base_seed, group_name=args.run_name)

    else:
        family.train(args.experiment, run_name=args.run_name)


if __name__ == "__main__":
    main()
