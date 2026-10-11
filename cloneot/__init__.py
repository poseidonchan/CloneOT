"""CloneOT: predicting clone-specific perturbation responses with lineage-supervised neural inverse optimal transport.

* :class:`CloneOTModel` -- AnnData interface: representation, cost learning, prediction.
* :class:`CloneOT` -- the solver on arrays of cell states.
* :mod:`cloneot.metrics` -- clone-level evaluation.
* :mod:`cloneot.datasets` -- simulated examples and public lineage-tracing data.
* :mod:`cloneot.plotting` (``cloneot.pl``) -- plots.
"""
from . import datasets, metrics
from . import plotting as pl
from .api import CloneOTModel
from .model import CloneOT, CloneOTConfig
from .vae import VAE, VAEConfig

__version__ = '0.1.0'
__all__ = ['CloneOT', 'CloneOTConfig', 'CloneOTModel', 'VAE', 'VAEConfig', 'datasets', 'metrics', 'pl']
