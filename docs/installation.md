# Installation

CloneOT requires Python 3.10 or later and PyTorch. We recommend installing PyTorch with the right compute platform
(CUDA or CPU) from the [PyTorch website](https://pytorch.org) first. CloneOT runs on a CPU, but a GPU makes training
much faster.

Install CloneOT from GitHub:

```bash
pip install git+https://github.com/poseidonchan/CloneOT.git
```

or from a local clone, together with the test and documentation tools:

```bash
git clone https://github.com/poseidonchan/CloneOT.git
cd CloneOT
pip install -e ".[test,docs]"
pytest
```

The dependencies are NumPy, SciPy, pandas, PyTorch, AnnData, Scanpy, scikit-misc (for Seurat v3 gene selection),
scikit-learn, Matplotlib and tqdm. The scripts that reproduce the paper need a few more packages, which are installed
with `pip install -e ".[experiments]"`.
