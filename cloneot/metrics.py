"""Clone-level evaluation of predicted perturbation responses.

For every clone with predicted profiles {p_i} (one per source cell) and observed target cells {y_j}:

* ``l2``: distance between the predicted and the observed clone-mean expression;
* ``r2``: squared Pearson correlation between the two clone means across genes;
* ``energy``: the energy score ``mean ||p - y|| - 1/2 mean ||p - p'||``, a strictly proper scoring rule of the
  predicted distribution.

The clone-specific correlation removes the response that all clones share: it correlates the predicted and observed
deviations of every clone mean from the average over clones, across all clones and genes.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _dense(x) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, 'toarray') else x, dtype=np.float64)


def energy_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Energy score ``mean ||p - y|| - 1/2 mean ||p - p'||`` of the predicted profiles (rows) against the observed cells
    (lower is better)."""
    return float(cdist(predicted, observed).mean()-.5*cdist(predicted, predicted).mean())


def clone_scores(predicted, observed, predicted_clones: Sequence, observed_clones: Sequence,
                 min_cells: int = 1) -> pd.DataFrame:
    """Per-clone l2, r2 and energy score.

    Parameters
    ----------
    predicted : array (n_predictions, n_genes)
        Predicted profile of every source cell.
    observed : array (n_observed, n_genes)
        Observed target cells.
    predicted_clones, observed_clones : sequence
        Clone of every predicted and observed cell. Clones present on both sides with at least `min_cells` cells on
        each side are scored.
    """
    predicted, observed = _dense(predicted), _dense(observed)
    predicted_clones, observed_clones = np.asarray(predicted_clones), np.asarray(observed_clones)
    rows = []
    for clone in sorted(set(predicted_clones.tolist()) & set(observed_clones.tolist()), key=str):
        p, y = predicted[predicted_clones == clone], observed[observed_clones == clone]
        if len(p) < min_cells or len(y) < min_cells:
            continue
        pm, ym = p.mean(0), y.mean(0)
        r = np.corrcoef(pm, ym)[0, 1] if pm.std() > 0 and ym.std() > 0 else np.nan
        rows.append(dict(clone=clone, n_predicted=len(p), n_observed=len(y), l2=float(np.linalg.norm(pm-ym)),
                         r2=float(r**2), energy=energy_score(p, y)))
    return pd.DataFrame(rows)


def clone_specific_correlation(predicted, observed, predicted_clones: Sequence, observed_clones: Sequence) -> float:
    """Pearson correlation between predicted and observed clone-specific deviations over all clones and genes."""
    predicted, observed = _dense(predicted), _dense(observed)
    predicted_clones, observed_clones = np.asarray(predicted_clones), np.asarray(observed_clones)
    clones = sorted(set(predicted_clones.tolist()) & set(observed_clones.tolist()), key=str)
    if len(clones) < 2:
        raise ValueError('at least two clones observed on both sides are required')
    P = np.stack([predicted[predicted_clones == c].mean(0) for c in clones])
    Y = np.stack([observed[observed_clones == c].mean(0) for c in clones])
    return float(np.corrcoef((P-P.mean(0)).ravel(), (Y-Y.mean(0)).ravel())[0, 1])


def evaluate(predicted, observed, clone_key: str, genes: Optional[Sequence[str]] = None,
             min_cells: int = 1) -> tuple[pd.DataFrame, float]:
    """Clone-level evaluation of AnnData objects: per-clone scores and the clone-specific correlation.

    Parameters
    ----------
    predicted : AnnData
        Predictions (e.g. from :meth:`cloneot.CloneOTModel.predict`), with the clone of every source cell in
        ``obs[clone_key]``.
    observed : AnnData
        Observed target cells (log-normalized expression) with ``obs[clone_key]``.
    genes : sequence of str, optional
        Genes to evaluate (e.g. the differentially expressed genes from :func:`cloneot.metrics.response_genes`). By
        default, all genes shared by the two objects.
    """
    genes = list(genes) if genes is not None else [g for g in predicted.var_names if g in set(observed.var_names)]
    p, y = predicted[:, genes], observed[:, genes]
    pc, yc = p.obs[clone_key].astype(object).to_numpy(), y.obs[clone_key].astype(object).to_numpy()
    keep_p, keep_y = pd.notna(pc) & (pc != ''), pd.notna(yc) & (yc != '')
    scores = clone_scores(p.X[keep_p], y.X[keep_y], pc[keep_p], yc[keep_y], min_cells)
    r = clone_specific_correlation(p.X[keep_p], y.X[keep_y], pc[keep_p], yc[keep_y]) if len(scores) > 1 else np.nan
    return scores, r


def response_genes(source, target, n_genes: int = 50) -> list[str]:
    """The `n_genes` genes with the largest t-statistics between target and source cells (log-normalized AnnData), the
    evaluation genes of the paper."""
    import anndata as ad
    import scanpy as sc
    both = ad.concat({'source': source, 'target': target}, label='_domain', index_unique='-')
    both.uns['log1p'] = {'base': None}
    sc.tl.rank_genes_groups(both, '_domain', groups=['target'], reference='source', method='t-test', use_raw=False,
                            n_genes=n_genes)
    return list(both.uns['rank_genes_groups']['names']['target'][:n_genes])
