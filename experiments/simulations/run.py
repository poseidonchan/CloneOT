"""The two 2D datasets of the paper (Fig. 2): the T-shape dataset and the rotated-moons dataset.

Methods: neural W2OT with the squared Euclidean cost (from the w1ot package, the solver underlying CellOT), OT-SI and
CloneOT, given the same correspondences. In the T-shape, the correspondences of a random 20% of the subsets of two
source and two target points are given; in the rotated moons, the correspondence between each source moon and its
rotated moon. Metrics on 400 new source points: the fraction of predictions that land on the target of their own class
(class of the nearest of 4,000 reference target points) and the mean distance to the true target position.

    python run.py --task tshape --seed 0          # results/simulations/<task>/<method>_s<seed>.{json,npz}
"""
from pathlib import Path
import argparse
import json
import tempfile
import time

import numpy as np
import torch

from cloneot import CloneOT
from cloneot.datasets import label_fraction, rotated_moons, t_shape
from otsi import OTSI, OTSIConfig

OUT = Path(__file__).resolve().parents[1]/'results/simulations'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TASKS = {'tshape': t_shape, 'rotated': rotated_moons}


def lineage(data, task, seed):
    """Source and target clone labels (-1: unlabeled)."""
    if task == 'rotated':
        return data['source_clone'], data['target_clone']
    return label_fraction(data['source_clone'], data['target_clone'], 0.2, seed)


def fit(method, source, target, labels, seed):
    start = time.time()
    if method == 'CloneOT':
        model = CloneOT(seed=seed, device=DEVICE).fit(source, target, *labels).fit_map()
        predict = model.transport
    elif method == 'OT-SI':
        model = OTSI(OTSIConfig(seed=seed), device=DEVICE).fit(source, target, *labels)
        predict = lambda s: model.predict_mean(s).numpy()
    else:                                                           # neural W2OT (squared Euclidean cost)
        from w1ot import w2ot
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = w2ot(source=source.astype(np.float32), target=target.astype(np.float32), positive=False,
                     validation_size=.1, device=DEVICE, path=tempfile.mkdtemp(prefix='w2ot_'))
        model.fit_potential_function(flavor='custom', hidden_sizes=[64]*4, kernel_init=None, batch_size=256,
                                     num_iters=10000, lr_init=1e-4, lr_min=1e-4, betas=(0.5, 0.9))
        predict = lambda s: np.asarray(model.transport(s.astype(np.float32)))
    return predict, time.time()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=list(TASKS), required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--methods', nargs='*', default=['NeuralOT', 'OT-SI', 'CloneOT'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = OUT/args.task
    folder.mkdir(parents=True, exist_ok=True)
    make = TASKS[args.task]
    data, test, reference = make(400, seed=args.seed), make(400, seed=100+args.seed), make(4000, seed=200+args.seed)
    labels = lineage(data, args.task, args.seed)
    for method in args.methods:
        predict, seconds = fit(method, data['source'], data['target'], labels, args.seed)
        prediction = predict(test['source'])
        nearest = ((prediction[:, None]-reference['target'][None])**2).sum(-1).argmin(1)
        summary = dict(task=args.task, method=method, seed=args.seed, fit_seconds=seconds,
                       class_accuracy=float((reference['target_class'][nearest] == test['source_class']).mean()),
                       fate_error=float(np.linalg.norm(prediction-test['fate'](test['source']), axis=1).mean()))
        stem = f'{method}_s{args.seed}'
        np.savez_compressed(folder/f'{stem}.npz', source=test['source'], target=test['target'], prediction=prediction,
                            source_class=test['source_class'], target_class=test['target_class'])
        (folder/f'{stem}.json').write_text(json.dumps(summary, indent=1)+'\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
