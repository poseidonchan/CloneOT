# Reproducing the paper

The [`experiments/`](https://github.com/poseidonchan/CloneOT/tree/main/experiments) folder of the repository contains
the scripts of the simulated examples of the paper (Fig. 2): neural OT with the squared Euclidean cost, OT-SI and
CloneOT on the T-shape and rotated-moons examples, each with three training seeds.

```bash
pip install -e ".[experiments]"
cd experiments/simulations
for task in tshape rotated; do for seed in 0 1 2; do python run.py --task $task --seed $seed; done; done
python figure.py
```

The default settings of CloneOT are the settings of the paper, shared by all simulations and single-cell tasks. The
[Watermelon tutorial](tutorials/watermelon) shows the complete single-cell workflow, from the download of the public
data to the clone-level evaluation of seen and unseen clones. The single-cell datasets of the paper are public:
Watermelon (GEO GSE150949), ClonMapper (GSE291678) and ReSisTrace (GSE223003).
