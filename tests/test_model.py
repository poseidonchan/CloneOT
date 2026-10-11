import numpy as np
import pytest

from cloneot import CloneOT
from cloneot.datasets import class_accuracy, label_fraction, t_shape
from cloneot.model import encode_clones

FAST = dict(warmup=50, steps=100, polish=100, map_steps=200, device='cpu')


@pytest.fixture(scope='module')
def toy():
    data = t_shape(200, seed=0)
    labels = label_fraction(data['source_clone'], data['target_clone'], .5, seed=0)
    model = CloneOT(**FAST).fit(data['source'], data['target'], *labels).fit_map()
    return data, model


def test_outputs(toy):
    data, model = toy
    new = t_shape(50, seed=1)['source']
    n = len(new)
    assert model.transport(new).shape == (n, 2)
    pi = model.conditional(new)
    assert pi.shape == (n, len(data['target'])) and np.allclose(pi.sum(1), 1, atol=1e-4)
    assert model.cost(new, data['target']).shape == (n, len(data['target']))
    assert model.learned and len(model.history) > 0


def test_save_and_load(toy, tmp_path):
    data, model = toy
    model.save(tmp_path/'model.pt')
    loaded = CloneOT.load(tmp_path/'model.pt', device='cpu')
    new = t_shape(20, seed=2)['source']
    assert np.allclose(model.transport(new), loaded.transport(new))
    assert np.allclose(model.conditional(new), loaded.conditional(new), atol=1e-6)


def test_without_labels_is_euclidean_ot():
    data = t_shape(100, seed=0)
    model = CloneOT(**FAST).fit(data['source'], data['target']).fit_map()
    assert not model.learned
    reference = t_shape(400, seed=3)
    # squared Euclidean OT sends the cells of the T-shape to the wrong bar
    assert class_accuracy(model.transport(data['source']), data['source_class'], reference) < .6


def test_seed_reproducibility():
    data = t_shape(100, seed=0)
    labels = label_fraction(data['source_clone'], data['target_clone'], .5, seed=0)
    a = CloneOT(**FAST).fit(data['source'], data['target'], *labels).fit_map().transport(data['source'])
    b = CloneOT(**FAST).fit(data['source'], data['target'], *labels).fit_map().transport(data['source'])
    assert np.allclose(a, b)


def test_encode_clones():
    s, t = encode_clones(['a', None, 'b', 'c'], ['b', 'a', np.nan, ''], 4, 4)
    assert (s == [0, -1, 1, 2]).all() and (t == [1, 0, -1, -1]).all()
    s, t = encode_clones(None, None, 3, 2)
    assert (s == -1).all() and (t == -1).all()
    with pytest.raises(ValueError):
        encode_clones([1, 2], [1], 3, 1)
