"""
Repeat one experiment across seeds and summarize the spread.

    python -m experiments cylinder --seeds 30

Each seed is an ordinary training run, written to

    results/<experiment>/seeds_<timestamp>/seed_<seed>/

so it carries its own checkpoints, latents and figures. The seed reported at the
end as representative is therefore ready to use for the paper figure with no
extra training: its figures are already in that directory.

The dataset and the train/test split are the same in every seed; only model
initialisation changes. The spread across seeds is optimisation variance at a
fixed dataset, which is what separates a reproducible effect from one seed
landing well. Beta comes from the experiment's spec, tuned so the unweighted KL
is comparable across the models being compared, so the medians are read at
matched KL. The `KL` column reports how well that matching held.

Reusing a `--run-name` skips seeds that already finished, so an interrupted set
resumes where it stopped.

Author: Jilles van Hulst
"""

import csv
import os
from datetime import datetime

import numpy as np

#: Metrics summarized, in the order they appear in the printed and LaTeX tables.
#: `beta` is a training setting rather than an outcome, but it belongs in the
#: table: the comparison is only fair at matched KL, and beta is what buys that.
METRICS = ["beta", "train_rms", "test_rms", "kl_divergence", "geodesic_stress"]

#: Metrics whose across-seed spread is reported alongside the median. These are
#: the ones where a single run can be misleading: a model that usually holds the
#: manifold but loses it on some seeds shows a wide interval here and a normal
#: median.
SPREAD_METRICS = ["test_rms", "geodesic_stress"]


def _read_metrics(run_dir):
    """A run's metrics.csv as {model: {metric: value}}."""
    out = {}
    with open(os.path.join(run_dir, "metrics.csv"), newline="") as f:
        for row in csv.DictReader(f):
            model, metric = row["metric"].rsplit(".", 1)
            out.setdefault(model, {})[metric] = float(row["value"])
    return out


def _stats(per_seed, model, metric):
    """Median and quartiles of one model's metric across seeds."""
    vals = np.array([s[model][metric] for s in per_seed])
    return np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75)


def _representative_seed(per_seed, seeds):
    """The seed sitting closest to the across-seed median.

    Each metric is scaled by its own interquartile range before distances are
    summed, so metrics on different scales count comparably. A metric that does
    not vary carries no information about which seed is typical and is skipped.
    """
    score = np.zeros(len(per_seed))
    for model in per_seed[0]:
        for metric in METRICS:
            vals = np.array([s[model][metric] for s in per_seed])
            median, q25, q75 = _stats(per_seed, model, metric)
            if q75 - q25 <= 0:
                continue
            score += np.abs(vals - median) / (q75 - q25)
    return seeds[int(np.argmin(score))]


def _write_summary(per_seed, models, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "n_seeds", "median", "q25", "q75"])
        for model in models:
            for metric in METRICS:
                median, q25, q75 = _stats(per_seed, model, metric)
                writer.writerow([model, metric, len(per_seed), median, q25, q75])


def _write_latex(per_seed, models, path):
    """Table body, one row per model, for pasting into the manuscript."""
    with open(path, "w", encoding="utf-8") as f:
        for model in models:
            cells = []
            for metric in METRICS:
                median, q25, q75 = _stats(per_seed, model, metric)
                digits = 2 if metric in ("kl_divergence", "beta") else 3
                cell = f"{median:.{digits}f}"
                if metric in SPREAD_METRICS:
                    cell += f" [{q25:.{digits}f}, {q75:.{digits}f}]"
                cells.append(cell)
            f.write(f"{model.replace('_', ' ')} & " + " & ".join(cells) + r" \\" + "\n")


def _print_table(per_seed, models):
    head = f"{'model':24}" + "".join(f"{m:>26}" for m in METRICS)
    print("\n" + head)
    print("-" * len(head))
    for model in models:
        line = f"{model:24}"
        for metric in METRICS:
            median, q25, q75 = _stats(per_seed, model, metric)
            line += f"{f'{median:.3f} [{q25:.3f},{q75:.3f}]':>26}"
        print(line)


def run(family, experiment, n_seeds=15, base_seed=1000, group_name=None):
    """Train `experiment` once per seed, then write and print the summary.

    Args:
        family: the module owning the experiment, exposing `train` and `results_dir`.
        experiment: which row of that family's SPECS to run.
        n_seeds: how many model initialisations to try.
        base_seed: the first seed; the rest follow consecutively.
        group_name: name of the directory holding the seeds. Reusing one resumes it.
    """
    group = group_name or f"seeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = family.results_dir(experiment)
    group_dir = os.path.join(results_dir, group)
    seeds = [base_seed + i for i in range(n_seeds)]

    per_seed = []
    for i, seed in enumerate(seeds):
        run_name = os.path.join(group, f"seed_{seed}")
        run_dir = os.path.join(results_dir, run_name)
        print("\n" + "=" * 70 + f"\nSEED {seed}  ({i + 1}/{len(seeds)})\n" + "=" * 70)
        if os.path.exists(os.path.join(run_dir, "metrics.csv")):
            print("Already trained; reusing.")
        else:
            family.train(experiment, run_name=run_name, seed=seed)
        per_seed.append(_read_metrics(run_dir))

    models = list(per_seed[0])
    _print_table(per_seed, models)

    os.makedirs(group_dir, exist_ok=True)
    _write_summary(per_seed, models, os.path.join(group_dir, "summary.csv"))
    _write_latex(per_seed, models, os.path.join(group_dir, "table.tex"))

    best = _representative_seed(per_seed, seeds)
    print(f"\nMedian and [q25, q75] over {len(seeds)} seeds -> {group_dir}")
    print(f"Representative seed: {best}")
    print(f"Figures for it: {os.path.join(group_dir, f'seed_{best}', 'figures')}")
