import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import cloneot as cot


@pytest.fixture(scope='module')
def adata():
    rng = np.random.default_rng(0)
    n, genes, clones = 400, 120, 20
    clone = rng.integers(clones, size=n)
    condition = np.where(np.arange(n) < n//2, 'before', 'after')
    rate = np.exp(rng.normal(size=(clones, genes)))[clone]*np.where(condition == 'after', 2., 1.)[:, None]
    counts = sparse.csr_matrix(rng.poisson(rate).astype(np.float32))
    obs = pd.DataFrame({'condition': condition, 'clone': [f'c{c}' if rng.random() < .8 else np.nan for c in clone]},
                       index=[f'cell{i}' for i in range(n)])
    return ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f'g{i}' for i in range(genes)]))


def test_train_predict_save_load(adata, tmp_path):
    model = cot.CloneOTModel(adata, condition_key='condition', source='before', target='after', clone_key='clone',
                             n_top_genes=50, config=cot.CloneOTConfig(warmup=20, steps=30, polish=30, map_steps=60),
                             vae_config=cot.VAEConfig(steps=100, eval_every=50, latent_dim=8, hidden=[32]), device='cpu')
    model.train(verbose=False)
    predicted = model.predict()
    assert predicted.shape == (200, 50) and (predicted.X >= 0).all()
    scores, r = cot.metrics.evaluate(predicted, model.observed(), 'clone')
    assert len(scores) > 0 and np.isfinite(r)
    model.save(tmp_path/'model')
    loaded = cot.CloneOTModel.load(tmp_path/'model', device='cpu')
    assert np.allclose(loaded.predict(adata).X, predicted.X, atol=1e-5)


def test_missing_condition(adata):
    with pytest.raises(ValueError):
        cot.CloneOTModel(adata, condition_key='condition', source='before', target='nope')
