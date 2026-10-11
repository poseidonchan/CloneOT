import numpy as np

from cloneot.metrics import clone_scores, clone_specific_correlation, energy_score


def test_perfect_prediction():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(60, 10))+np.repeat(rng.normal(size=(3, 10)), 20, 0)
    clones = np.repeat(['a', 'b', 'c'], 20)
    scores = clone_scores(y, y, clones, clones)
    assert np.allclose(scores.l2, 0) and np.allclose(scores.r2, 1)
    assert np.isclose(clone_specific_correlation(y, y, clones, clones), 1)


def test_energy_score_prefers_the_true_distribution():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(300, 2))
    assert energy_score(rng.normal(size=(300, 2)), y) < energy_score(rng.normal(size=(300, 2))+1, y)


def test_shared_response_has_no_clone_specific_signal():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(40, 5))+np.repeat(rng.normal(size=(4, 5)), 10, 0)
    clones = np.repeat(list('abcd'), 10)
    constant = np.repeat(y.mean(0, keepdims=True), 40, 0)+1e-9*rng.normal(size=(40, 5))
    assert abs(clone_specific_correlation(constant, y, clones, clones)) < .5
