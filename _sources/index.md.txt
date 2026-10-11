# CloneOT

**Predicting clone-specific perturbation responses with lineage-supervised neural inverse optimal transport.**

Single-cell perturbation experiments measure two unpaired populations, cells before and cells after the perturbation.
Optimal transport (OT) methods such as CellOT predict the response of a cell by transporting it at minimal cost, but the
cost, usually the squared Euclidean distance, is fixed in advance and may connect cells with the wrong responses.
Lineage tracing links the cells of the same clone before and after the perturbation. **CloneOT** uses these links to
learn the transport cost itself, and a neural map then predicts the response of any new cell from its expression alone.

```{image} _static/overview.png
:alt: Overview of CloneOT
:width: 100%
```

**Highlights**

- Learns a neural transport cost $c(x,y)=h(\varphi(x)-\psi(y))$ from partially lineage-traced data, with neural ODE
  flows $\varphi,\psi$ and a symmetric input convex network $h$.
- Needs lineage only for training. Predictions for new cells and for clones that were never labeled use their
  expression alone.
- Works directly on AnnData: gene selection, VAE representation, cost learning, prediction and clone-level evaluation.
- Trains one model in about 1.6 minutes on a single GPU.

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
quickstart
method
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/simulations
tutorials/watermelon
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
reproducibility
```

Questions and issues are welcome on [GitHub](https://github.com/poseidonchan/CloneOT/issues) or by email to
cys@umd.edu.
