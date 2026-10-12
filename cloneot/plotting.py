"""Plotting functions of CloneOT."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

SOURCE, TARGET, PREDICTION, RAY, WRONG = '#1f2a6b', '#f08080', '#7ec8e3', '#9a9a9a', '#d62728'


def transport(source, target, prediction, correct: Optional[Sequence[bool]] = None, ax=None, rays: int = 4,
              title: Optional[str] = None, legend: bool = True):
    """Two-dimensional transport plot: source cells, target cells, predictions, and rays from every `rays`-th source
    cell to its prediction (red if `correct` is False for that cell)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    source, target, prediction = map(np.asarray, (source, target, prediction))
    correct = np.ones(len(source), bool) if correct is None else np.asarray(correct)
    ax.scatter(*target.T, s=3, c=TARGET, lw=0, zorder=1, label='Target')
    ax.scatter(*source.T, s=3, c=SOURCE, lw=0, zorder=2, label='Source')
    for i in range(0, len(source), rays):
        ax.plot(*np.c_[source[i], prediction[i]], c=RAY if correct[i] else WRONG, lw=.35, alpha=.8, zorder=3)
    ax.scatter(*prediction.T, s=3, c=PREDICTION, lw=0, zorder=4, label='Prediction')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=False, markerscale=3, loc='best')
    return ax


def clone_deviations(predicted, observed, clone_key: str, genes: Optional[Sequence[str]] = None, ax=None,
                     color: str = '#51A9D6'):
    """Scatter of predicted against observed clone-specific deviations (clone mean minus the mean over clones) for
    every clone and gene, annotated with the clone-specific correlation r."""
    import matplotlib.pyplot as plt
    import pandas as pd
    ax = ax or plt.gca()
    genes = list(genes) if genes is not None else [g for g in predicted.var_names if g in set(observed.var_names)]
    pc = predicted.obs[clone_key].astype(object)
    yc = observed.obs[clone_key].astype(object)
    clones = sorted(set(pc[pd.notna(pc)]) & set(yc[pd.notna(yc)]), key=str)
    dense = lambda a: np.asarray(a.toarray() if hasattr(a, 'toarray') else a)
    P = np.stack([dense(predicted[(pc == c).to_numpy(), genes].X).mean(0) for c in clones])
    Y = np.stack([dense(observed[(yc == c).to_numpy(), genes].X).mean(0) for c in clones])
    dp, dy = (P-P.mean(0)).ravel(), (Y-Y.mean(0)).ravel()
    r = np.corrcoef(dp, dy)[0, 1]
    ax.scatter(dy, dp, s=4, c=color, lw=0, alpha=.6)
    lim = np.abs(np.r_[dp, dy]).max()*1.05
    ax.plot([-lim, lim], [-lim, lim], ls='--', lw=.6, c='#777777')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Observed deviation')
    ax.set_ylabel('Predicted deviation')
    ax.text(.04, .96, f'r = {r:.2f}\n{len(clones)} clones', transform=ax.transAxes, va='top')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return ax


METRIC_LABELS = {'l2': r'$\ell_2$ error', 'r2': r'$r^2$', 'energy': 'energy score'}


def paired_scores(baseline, scores, metric: str = 'l2', labels: Sequence[str] = ('Baseline', 'CloneOT'), ax=None,
                  color: str = '#51A9D6', other: str = '#C7BBAF'):
    """Per-clone comparison of two methods on the same clones: the metric of `baseline` (x-axis) against that of
    `scores` (y-axis), one point per clone, from data frames returned by :func:`cloneot.metrics.evaluate`.

    Clones on which the second method is better are colored, and the panel reports their fraction and the p-value of
    the two-sided Wilcoxon signed-rank test.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    ax = ax or plt.gca()
    joined = baseline[['clone', metric]].merge(scores[['clone', metric]], on='clone', suffixes=('_x', '_y')).dropna()
    x, y = joined[metric+'_x'].to_numpy(), joined[metric+'_y'].to_numpy()
    better = y > x if metric == 'r2' else y < x
    ax.scatter(x[~better], y[~better], s=10, c=other, lw=0, alpha=.8)
    ax.scatter(x[better], y[better], s=10, c=color, lw=0, alpha=.8)
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    pad = .04*(high-low)
    ax.plot([low-pad, high+pad], [low-pad, high+pad], ls='--', lw=.6, c='#777777')
    ax.set_xlim(low-pad, high+pad)
    ax.set_ylim(low-pad, high+pad)
    ax.set_aspect('equal')
    name = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel(f'{labels[0]}, {name}')
    ax.set_ylabel(f'{labels[1]}, {name}')
    p = wilcoxon(y-x).pvalue
    ax.text(.04, .96, f'{labels[1]} better for {better.mean():.0%}\nof {len(x)} clones (p = {p:.0e})',
            transform=ax.transAxes, va='top', fontsize=9, color=color if better.mean() > .5 else 'black')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return ax
