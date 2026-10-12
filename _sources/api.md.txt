# API

```{eval-rst}
.. currentmodule:: cloneot
```

## AnnData interface

```{eval-rst}
.. autoclass:: cloneot.CloneOTModel
   :members: train, predict, observed, latent, response_genes, save, load
```

## Solver

```{eval-rst}
.. autoclass:: cloneot.CloneOT
   :members: fit, fit_map, transport, conditional, barycentric_projection, cost, save, load

.. autoclass:: cloneot.CloneOTConfig
   :no-members:
```

The fields of `CloneOTConfig` and their defaults (the settings of the paper):

```{eval-rst}
.. literalinclude:: ../cloneot/model.py
   :language: python
   :pyobject: CloneOTConfig
```

## Representation

```{eval-rst}
.. autoclass:: cloneot.VAE
   :members: fit, latent, reconstruct

.. autoclass:: cloneot.VAEConfig
   :no-members:
```

## Evaluation

```{eval-rst}
.. automodule:: cloneot.metrics
   :members: evaluate, clone_scores, clone_specific_correlation, energy_score, response_genes
```

## Datasets

```{eval-rst}
.. automodule:: cloneot.datasets
   :members: watermelon, holdout, t_shape, rotated_moons, label_fraction, class_accuracy
```

## Plotting

```{eval-rst}
.. automodule:: cloneot.plotting
   :members: transport, clone_deviations, paired_scores
```
