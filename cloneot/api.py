"""AnnData interface of CloneOT.

:class:`CloneOTModel` takes an AnnData object with cells measured before (source) and after (target) a perturbation and
the clone labels of lineage-traced cells. It selects highly variable genes, learns a VAE representation of both
populations, learns the transport cost from the labeled clones, and predicts the response of every source cell.

Example
-------
>>> import cloneot as cot
>>> model = cot.CloneOTModel(adata, condition_key='condition', source='control', target='treated', clone_key='clone')
>>> model.train()
>>> predicted = model.predict()            # AnnData: predicted expression of every source cell
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .metrics import clone_scores, response_genes
from .model import CloneOT, CloneOTConfig
from .vae import VAE, VAEConfig


def is_counts(X) -> bool:
    """Whether a matrix holds raw counts (non-negative integers)."""
    values = X.data if hasattr(X, 'data') else np.asarray(X).ravel()
    values = values[:100000]
    return bool(len(values) == 0 or (values.min() >= 0 and np.allclose(values, np.round(values))))


def _labels(values) -> np.ndarray:
    """Clone labels as objects, with None for cells without a label."""
    values = pd.Series(values).astype(object)
    return np.array([None if (v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ('', 'nan', 'None'))
                     else str(v) for v in values], dtype=object)


class CloneOTModel:
    """CloneOT on an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Cells of both conditions. ``adata.X`` (or ``adata.layers[layer]``) holds raw counts or log-normalized expression.
    condition_key : str
        Column of ``adata.obs`` with the condition of every cell.
    source, target : str
        Values of ``condition_key`` for the cells before and after the perturbation.
    clone_key : str, optional
        Column of ``adata.obs`` with the clone of every lineage-traced cell (NaN or empty for cells without a label).
        Without clone labels, CloneOT is entropic optimal transport with the squared Euclidean cost.
    layer : str, optional
        Layer to use instead of ``adata.X``.
    n_top_genes : int
        Highly variable genes to model (Seurat v3 selection on counts, treating the two conditions as batches).
    config, vae_config : optional
        Settings of CloneOT and of the VAE (defaults: the settings of the paper).
    device : str, optional
        Torch device.
    """

    def __init__(self, adata, condition_key: str, source: str, target: str, clone_key: Optional[str] = None,
                 layer: Optional[str] = None, n_top_genes: int = 2000, config: Optional[CloneOTConfig] = None,
                 vae_config: Optional[VAEConfig] = None, device: Optional[str] = None):
        self.condition_key, self.source, self.target = condition_key, str(source), str(target)
        self.clone_key, self.layer, self.n_top_genes = clone_key, layer, n_top_genes
        self.config, self.vae_config = config or CloneOTConfig(), vae_config or VAEConfig()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        condition = adata.obs[condition_key].astype(str)
        if not (condition == self.source).any() or not (condition == self.target).any():
            raise ValueError(f'no cells with {condition_key} == {source!r} or {target!r}')
        self.adata = adata[condition.isin([self.source, self.target]).to_numpy()]
        self.genes = None
        self.vae = self.ot = None

    # ----------------------------------------------------------------------------------------------- preprocessing
    def _matrix(self, adata):
        return adata.layers[self.layer] if self.layer is not None else adata.X

    def _select_genes(self):
        import scanpy as sc
        data = self.adata.copy()
        data.X = self._matrix(data)
        data.obs['_domain'] = data.obs[self.condition_key].astype(str).to_numpy()
        if is_counts(data.X):
            sc.pp.highly_variable_genes(data, n_top_genes=self.n_top_genes, flavor='seurat_v3', batch_key='_domain')
        else:
            sc.pp.highly_variable_genes(data, n_top_genes=self.n_top_genes, batch_key='_domain')
        return list(data.var_names[data.var.highly_variable])

    def log_expression(self, adata) -> np.ndarray:
        """Log-normalized expression of the model genes (normalization uses all genes of `adata`)."""
        X = self._matrix(adata)
        if is_counts(X):
            columns = adata.var_names.get_indexer(self.genes)
            if (columns < 0).any():
                raise ValueError('adata lacks some genes of the model')
            total = np.asarray(X.sum(1)).reshape(-1, 1)          # library size over all genes (sparse friendly)
            X = X[:, columns]
            X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
            return np.log1p(X/np.maximum(total, 1e-12)*1e4).astype(np.float32)
        X = adata[:, self.genes].X if self.layer is None else adata[:, self.genes].layers[self.layer]
        return np.asarray(X.toarray() if hasattr(X, 'toarray') else X, dtype=np.float32)

    def _split(self, adata):
        condition = adata.obs[self.condition_key].astype(str).to_numpy()
        return adata[condition == self.source], adata[condition == self.target]

    def latent(self, adata) -> np.ndarray:
        """VAE latent representation (posterior means) of the cells of `adata`."""
        x = torch.as_tensor(self.log_expression(adata), device=self.device)
        return self.vae.latent(x).cpu().numpy()

    # -------------------------------------------------------------------------------------------------- training
    def train(self, validation=None, representation: Optional['CloneOTModel'] = None, n_response_genes: int = 50,
              verbose: bool = True) -> 'CloneOTModel':
        """Learn the representation, the transport cost and the neural map.

        Parameters
        ----------
        representation : CloneOTModel, optional
            A trained model whose genes and VAE are reused, so that only the transport cost and the map are learned
            (e.g. to compare CloneOT with and without lineage on the same representation).
        validation : AnnData, optional
            Held-out source and target cells with clone labels (e.g. held-out cells of training clones). If given, the
            map checkpoint with the best validation criterion is kept: the mean over validation clones of l2, 1 - r^2
            and the energy score on the response genes, each relative to predicting every clone by the mean training
            target cell.
        n_response_genes : int
            Number of response genes used for validation.
        verbose : bool
            Show progress bars.
        """
        if representation is not None:
            self.genes, self.vae, self.vae_config = representation.genes, representation.vae, representation.vae_config
        else:
            self.genes = self._select_genes()
        source, target = self._split(self.adata)
        expression = torch.as_tensor(np.concatenate([self.log_expression(source), self.log_expression(target)]),
                                     device=self.device)
        if representation is None:
            domain = np.r_[np.zeros(source.n_obs, int), np.ones(target.n_obs, int)]
            torch.manual_seed(self.vae_config.seed)
            self.vae = VAE(len(self.genes), self.vae_config).to(self.device).fit(expression, domain, verbose=verbose)
            self.vae.requires_grad_(False)
        latent = self.vae.latent(expression).cpu().numpy()
        source_clones = target_clones = None
        if self.clone_key is not None:
            source_clones, target_clones = _labels(source.obs[self.clone_key]), _labels(target.obs[self.clone_key])
        self.ot = CloneOT(self.config, device=str(self.device)).fit(latent[:source.n_obs], latent[source.n_obs:],
                                                                      source_clones, target_clones, verbose=verbose)
        evaluate = None
        if validation is not None:
            evaluate = self._selection(validation, expression[source.n_obs:], n_response_genes)
        self.ot.fit_map(decoder=self.vae.decode, evaluate=evaluate, verbose=verbose)
        return self

    def response_genes(self, n_genes: int = 50) -> list[str]:
        """The `n_genes` model genes with the largest t-statistics between the training target and source cells (the
        evaluation genes of the paper)."""
        import anndata as ad
        source, target = self._split(self.adata)
        var = pd.DataFrame(index=self.genes)
        return response_genes(ad.AnnData(self.log_expression(source), var=var),
                              ad.AnnData(self.log_expression(target), var=var), n_genes)

    def _selection(self, validation, target_expression, n_genes):
        """Validation criterion of the paper (lower is better)."""
        source, target = self._split(validation)
        if self.clone_key is None:
            raise ValueError('validation requires clone labels')
        sx = torch.as_tensor(self.log_expression(source), device=self.device)
        z = self.vae.latent(sx).cpu().numpy()
        genes = self.response_genes(n_genes)
        columns = [self.genes.index(g) for g in genes]
        observed = self.log_expression(target)[:, columns]
        sc, tc = _labels(source.obs[self.clone_key]), _labels(target.obs[self.clone_key])
        keep_s, keep_t = sc != None, tc != None   # noqa: E711
        constant = np.repeat(target_expression.mean(0, keepdim=True).cpu().numpy()[:, columns], len(z), 0)
        reference = clone_scores(constant[keep_s], observed[keep_t], sc[keep_s], tc[keep_t])[['l2', 'r2', 'energy']].mean()

        def criterion(transport):
            predicted = self.vae.reconstruct(torch.as_tensor(transport(z), device=self.device)).cpu().numpy()[:, columns]
            m = clone_scores(predicted[keep_s], observed[keep_t], sc[keep_s], tc[keep_t])[['l2', 'r2', 'energy']].mean()
            return (m.l2/reference.l2+(1-m.r2)/(1-reference.r2)+m.energy/reference.energy)/3
        return criterion

    # ------------------------------------------------------------------------------------------------- prediction
    def predict(self, adata=None, latent: bool = False):
        """Predicted response of every source cell of `adata` (default: the training source cells).

        Returns an AnnData with the predicted log-normalized expression of the model genes and the ``obs`` of the
        source cells, or, with ``latent=True``, the predicted latent states as an array.
        """
        import anndata as ad
        if self.ot is None:
            raise RuntimeError('call train() first')
        adata = self.adata if adata is None else adata
        if self.condition_key in adata.obs:
            adata = self._split(adata)[0]
        z = self.ot.transport(self.latent(adata))
        if latent:
            return z
        X = self.vae.reconstruct(torch.as_tensor(z, device=self.device)).cpu().numpy()
        predicted = ad.AnnData(X, obs=adata.obs.copy(), var=pd.DataFrame(index=pd.Index(self.genes, name=None)))
        predicted.obsm['X_cloneot_latent'] = z
        predicted.uns['cloneot'] = dict(source=self.source, target=self.target, condition_key=self.condition_key)
        return predicted

    def observed(self, adata=None):
        """Log-normalized expression of the model genes of the target cells of `adata` (default: training cells), as an
        AnnData to compare with :meth:`predict`."""
        import anndata as ad
        adata = self.adata if adata is None else adata
        target = self._split(adata)[1]
        return ad.AnnData(self.log_expression(target), obs=target.obs.copy(), var=pd.DataFrame(index=self.genes))

    # ---------------------------------------------------------------------------------------------- save and load
    def save(self, folder) -> None:
        """Save the model (genes, VAE and CloneOT) to `folder`; the training AnnData is not saved."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.vae.state(), folder/'vae.pt')
        self.ot.save(folder/'cloneot.pt')
        meta = dict(condition_key=self.condition_key, source=self.source, target=self.target, clone_key=self.clone_key,
                    layer=self.layer, n_top_genes=self.n_top_genes, genes=self.genes, vae_config=asdict(self.vae_config))
        (folder/'model.json').write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, folder, adata=None, device: Optional[str] = None) -> 'CloneOTModel':
        """Load a saved model. `adata` (optional) is only needed to predict the training source cells by default."""
        folder = Path(folder)
        meta = json.loads((folder/'model.json').read_text())
        model = cls.__new__(cls)
        model.condition_key, model.source, model.target = meta['condition_key'], meta['source'], meta['target']
        model.clone_key, model.layer, model.n_top_genes, model.genes = meta['clone_key'], meta['layer'], meta['n_top_genes'], meta['genes']
        model.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        model.vae_config = VAEConfig(**meta['vae_config'])
        model.vae = VAE.from_state(torch.load(folder/'vae.pt', map_location='cpu', weights_only=False)).to(model.device)
        model.ot = CloneOT.load(folder/'cloneot.pt', device=str(model.device))
        model.config = model.ot.config
        model.adata = adata
        return model
