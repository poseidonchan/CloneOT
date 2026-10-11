# Experiments

## Simulated examples (Fig. 2)

`simulations/run.py` trains neural OT with the squared Euclidean cost (the solver underlying CellOT, from our
[w1ot](https://github.com/poseidonchan/w1ot) package), OT-SI (`simulations/otsi.py`) and CloneOT on the T-shape and
rotated-moons examples, and `simulations/figure.py` draws Fig. 2 and summarizes the three training seeds.

```bash
pip install -e ".[experiments]"
cd experiments/simulations
for task in tshape rotated; do for seed in 0 1 2; do python run.py --task $task --seed $seed; done; done
python figure.py          # results/simulations/fig2_examples.pdf and summary.csv
```

CloneOT uses the default settings of the package, which are the settings of the paper for all simulations and
single-cell tasks.

## Single-cell benchmark

The single-cell experiments use the public lineage-tracing data of the Watermelon (GSE150949), ClonMapper (GSE291678)
and ReSisTrace (GSE223003) studies, with the settings listed in the supplement of the paper. The
[tutorial](../docs/tutorials/watermelon.ipynb) shows the complete workflow on the Watermelon data, from the download to
the clone-level evaluation of seen and unseen clones.
