"""Fig. 2 of the paper and the numbers reported in Section 3.1, from the results of run.py (three training seeds).

    python figure.py                              # results/simulations/fig2_examples.{pdf,png} and summary.csv
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cloneot.datasets import rotated_moons, t_shape

RESULTS = Path(__file__).resolve().parents[1]/'results/simulations'
TASKS = {'tshape': ('T-shape', t_shape), 'rotated': ('Rotated moons', rotated_moons)}
METHODS = {'NeuralOT': 'Neural OT', 'OT-SI': 'OT-SI', 'CloneOT': 'CloneOT'}
SEEDS = (0, 1, 2)
SOURCE, TARGET, MOVED, RAY, WRONG = '#1f2a6b', '#f08080', '#7ec8e3', '#9a9a9a', '#d62728'


def own_class(task, prediction, source_class, seed=0):
    reference = TASKS[task][1](4000, seed=200+seed)
    nearest = ((prediction[:, None]-reference['target'][None])**2).sum(-1).argmin(1)
    return reference['target_class'][nearest] == source_class


def panel(ax, source, target, moved, correct, title=None):
    ax.scatter(*target.T, s=3, c=TARGET, lw=0, zorder=1)
    ax.scatter(*source.T, s=3, c=SOURCE, lw=0, zorder=2)
    for i in range(0, len(source), 4):
        ax.plot(*np.c_[source[i], moved[i]], c=RAY if correct[i] else WRONG, lw=.35, alpha=.8, zorder=3)
    ax.scatter(*moved.T, s=3, c=MOVED, lw=0, zorder=4)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9.5)


def figure():
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.5))
    for r, (row, (task, (label, make))) in enumerate(zip(axes, TASKS.items())):
        test = make(400, seed=100)
        truth = test['fate'](test['source'])
        panel(row[0], test['source'], test['target'], truth, np.ones(len(truth), bool), 'Ground truth' if r == 0 else None)
        row[0].text(-.06, .5, label, transform=row[0].transAxes, rotation=90, ha='right', va='center', fontsize=9.5)
        for ax, (method, title) in zip(row[1:], METHODS.items()):
            z = np.load(RESULTS/task/f'{method}_s0.npz')
            panel(ax, z['source'], z['target'], z['prediction'], own_class(task, z['prediction'], z['source_class']),
                  title if r == 0 else None)
        lims = np.r_[test['source'], test['target']]
        pad = .08*(lims.max(0)-lims.min(0))
        for ax in row:
            ax.set_xlim(lims[:, 0].min()-pad[0], lims[:, 0].max()+pad[0])
            ax.set_ylim(lims[:, 1].min()-pad[1], lims[:, 1].max()+pad[1])
        fig.text(0, row[0].get_position().y1, 'ab'[r], fontsize=13, fontweight='bold', va='bottom')
    handles = [plt.Line2D([], [], ls='', marker='o', ms=3, c=c) for c in (SOURCE, TARGET, MOVED)]
    handles += [plt.Line2D([], [], c=RAY, lw=.8), plt.Line2D([], [], c=WRONG, lw=.8)]
    fig.legend(handles, ['Source', 'Target', 'Prediction', 'Transport', 'Transport to the other class'],
               loc='lower center', ncol=5, fontsize=8, frameon=False)
    fig.subplots_adjust(left=.05, right=.995, top=.93, bottom=.08, wspace=.05, hspace=.08)
    for ext in ('pdf', 'png'):
        fig.savefig(RESULTS/f'fig2_examples.{ext}', dpi=300)


def summary():
    rows = [json.loads((RESULTS/task/f'{method}_s{s}.json').read_text()) for task in TASKS for method in METHODS
            for s in SEEDS]
    table = pd.DataFrame(rows).groupby(['task', 'method']).agg(
        class_accuracy=('class_accuracy', 'mean'), accuracy_min=('class_accuracy', 'min'),
        accuracy_max=('class_accuracy', 'max'), fate_error=('fate_error', 'mean'))
    table.to_csv(RESULTS/'summary.csv')
    return table


if __name__ == '__main__':
    figure()
    print(summary().round(3).to_string())
