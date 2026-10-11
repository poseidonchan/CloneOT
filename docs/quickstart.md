# Quick start

## On an AnnData object

Your data need a column with the condition of every cell (before or after the perturbation) and a column with the clone
of every lineage-traced cell. Cells without a lineage label have `NaN` (or an empty string) in the clone column. Raw
counts or log-normalized expression are both accepted.

```python
import cloneot as cot

model = cot.CloneOTModel(
    adata,
    condition_key="condition",   # column of adata.obs with the condition of every cell
    source="control",            # cells before the perturbation
    target="treated",            # cells after the perturbation
    clone_key="clone",           # clone of every lineage-traced cell, NaN otherwise
)
model.train()                    # genes, VAE, transport cost and neural map

predicted = model.predict(new_cells)       # AnnData: predicted expression of every source cell
observed = model.observed(new_cells)       # AnnData: observed target cells, to compare with
genes = model.response_genes(50)           # the 50 genes that change most between the conditions
scores, r = cot.metrics.evaluate(predicted, observed, clone_key="clone", genes=genes)
```

`scores` holds the $\ell_2$ distance and $r^2$ between the predicted and observed clone means and the energy score of
every clone, and `r` is the clone-specific correlation, which measures how well the predictions capture the differences
between clones. Save and reload a trained model with `model.save("folder")` and `cot.CloneOTModel.load("folder")`.

## On arrays

The solver works on any arrays of cell states, such as a precomputed latent representation:

```python
from cloneot import CloneOT

model = CloneOT()                                   # the settings of the paper; e.g. CloneOT(seed=1) to override
model.fit(source, target, source_clones, target_clones)    # clone labels, -1 or None for unlabeled cells
model.fit_map()
prediction = model.transport(new_source)            # predicted response of every new source cell
coupling = model.conditional(new_source)            # probability of every training target cell as descendant
cost = model.cost(new_source, target)               # the learned transport cost
```

Without clone labels, CloneOT is entropic optimal transport with the squared Euclidean cost, the solver underlying
CellOT, which makes it a natural baseline.

## Runtime and memory

CloneOT keeps the cost matrix between all training source and target cells, so its memory and time grow with the
product of their numbers. A scenario with a few thousand cells on each side trains in one to four minutes on a single
GPU, and the Watermelon tutorial, with about 6,000 source and 12,000 target cells, takes several minutes per model. For
much larger populations, subsample the cells without lineage labels first, since the labeled clones carry the
supervision.
